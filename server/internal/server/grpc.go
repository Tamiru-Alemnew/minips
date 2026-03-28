package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/teim/minips/server/internal/codec"
	"github.com/teim/minips/server/internal/consistency"
	"github.com/teim/minips/server/internal/metrics"
	"github.com/teim/minips/server/internal/params"
	"github.com/teim/minips/server/internal/pb"
	"github.com/teim/minips/server/internal/shard"
)

const ChunkBytes = 4 << 20

type Config struct {
	Workers     int
	Shards      int
	Mode        string
	Staleness   int
	LR          float64
	WarmupSteps int64
	TotalSteps  int64
	Optim       shard.Optim
}

type Server struct {
	pb.UnimplementedParameterServerServer
	cfg     Config
	group   *shard.Group
	policy  consistency.Policy
	Metrics *metrics.Registry
	numel   map[string]int
}

func New(cfg Config, tensors []params.Tensor) (*Server, error) {
	if cfg.Workers < 1 || cfg.Shards < 1 {
		return nil, fmt.Errorf("need at least one worker and one shard")
	}
	s := &Server{cfg: cfg, Metrics: metrics.New(cfg.Workers), numel: map[string]int{}}
	s.group = shard.NewGroup(cfg.Shards, tensors, cfg.Optim)
	for _, t := range tensors {
		s.numel[t.Name] = t.Numel()
	}
	apply := func(c int64) {
		s.Metrics.Updates.Add(int64(s.group.Flush(1/float32(cfg.Workers), s.lr(c))))
	}
	p, err := consistency.New(cfg.Mode, cfg.Workers, cfg.Staleness, apply)
	if err != nil {
		return nil, err
	}
	s.policy = p
	return s, nil
}

func (s *Server) Close() { s.group.Close() }

func (s *Server) lr(c int64) float32 {
	return float32(LR(s.cfg.LR, c, s.cfg.WarmupSteps, s.cfg.TotalSteps))
}

// LR is linear warm-up for warmup clocks, then cosine decay to zero at total.
func LR(peak float64, c, warmup, total int64) float64 {
	if c < warmup {
		return peak * float64(c+1) / float64(warmup)
	}
	if total <= warmup {
		return peak
	}
	t := min(float64(c-warmup)/float64(total-warmup), 1)
	return 0.5 * peak * (1 + math.Cos(math.Pi*t))
}

func (s *Server) worker(id int32) (*metrics.Worker, error) {
	w := s.Metrics.Worker(int(id))
	if w == nil {
		return nil, status.Errorf(codes.InvalidArgument, "worker %d out of range [0,%d)", id, s.cfg.Workers)
	}
	return w, nil
}

func (s *Server) Register(_ context.Context, r *pb.RegisterRequest) (*pb.RegisterReply, error) {
	if _, err := s.worker(r.WorkerId); err != nil {
		return nil, err
	}
	out := &pb.RegisterReply{NumWorkers: int32(s.cfg.Workers), Mode: s.policy.Name(), Staleness: int32(s.cfg.Staleness)}
	for _, sp := range s.group.Specs {
		out.Params = append(out.Params, &pb.ParamSpec{Name: sp.Name, Shape: sp.Shape, Dtype: sp.Dtype})
	}
	log.Printf("worker %d registered", r.WorkerId)
	return out, nil
}

func (s *Server) Pull(r *pb.PullRequest, stream pb.ParameterServer_PullServer) error {
	w, err := s.worker(r.WorkerId)
	if err != nil {
		return err
	}
	minDone, err := s.policy.WaitRead(stream.Context(), int(r.WorkerId), r.Clock)
	if err != nil {
		return status.FromContextError(err).Err()
	}
	w.Pulls.Add(1)
	for _, sp := range s.group.Specs {
		data, err := s.group.Read(sp.Name)
		if err != nil {
			return status.Error(codes.Internal, err.Error())
		}
		for off := 0; ; off += ChunkBytes {
			end := min(off+ChunkBytes, len(data))
			msg := &pb.ParamChunk{Name: sp.Name, Offset: int64(off), Data: data[off:end], MinClock: minDone}
			if err := stream.Send(msg); err != nil {
				return err
			}
			if end == len(data) {
				break
			}
		}
	}
	return nil
}

func (s *Server) Push(stream pb.ParameterServer_PushServer) error {
	type pending struct {
		enc  pb.Encoding
		data []byte
	}
	var (
		worker int32 = -1
		clock  int64
		order  []string
		parts  = map[string]*pending{}
	)
	for {
		m, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if worker == -1 {
			worker, clock = m.WorkerId, m.Clock
		} else if m.WorkerId != worker || m.Clock != clock {
			return status.Error(codes.InvalidArgument, "one push carries one worker and one clock")
		}
		p, ok := parts[m.Name]
		if !ok {
			p = &pending{enc: m.Encoding}
			parts[m.Name] = p
			order = append(order, m.Name)
		}
		p.data = append(p.data, m.Data...)
	}
	if worker == -1 {
		return stream.SendAndClose(&pb.PushReply{})
	}
	w, err := s.worker(worker)
	if err != nil {
		return err
	}
	scale, lr := 1/float32(s.cfg.Workers), s.lr(clock)
	for _, name := range order {
		n, ok := s.numel[name]
		if !ok {
			return status.Errorf(codes.InvalidArgument, "unknown tensor %q", name)
		}
		grad, err := codec.Decode(parts[name].enc, parts[name].data, n)
		if err != nil {
			return status.Errorf(codes.InvalidArgument, "%s: %v", name, err)
		}
		if s.policy.Aggregates() {
			err = s.group.Accumulate(name, grad)
		} else {
			err = s.group.Apply(name, grad, scale, lr)
			s.Metrics.Updates.Add(1)
		}
		if err != nil {
			return status.Error(codes.Internal, err.Error())
		}
	}
	w.Pushes.Add(1)
	return stream.SendAndClose(&pb.PushReply{})
}

func (s *Server) Clock(_ context.Context, r *pb.ClockRequest) (*pb.ClockReply, error) {
	if _, err := s.worker(r.WorkerId); err != nil {
		return nil, err
	}
	if err := s.policy.Finish(int(r.WorkerId), r.Clock); err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	return &pb.ClockReply{}, nil
}

func (s *Server) Stats(context.Context, *pb.StatsRequest) (*pb.StatsReply, error) {
	ps := s.policy.Stats()
	out := &pb.StatsReply{StalenessHist: ps.Staleness, UpdatesApplied: s.Metrics.Updates.Load()}
	for i := 0; i < s.Metrics.N(); i++ {
		w := s.Metrics.Worker(i)
		out.Workers = append(out.Workers, &pb.WorkerStats{
			WorkerId:    int32(i),
			BytesIn:     w.BytesIn.Load(),
			BytesOut:    w.BytesOut.Load(),
			WaitSeconds: ps.WaitSeconds[i],
			Pulls:       w.Pulls.Load(),
			Pushes:      w.Pushes.Load(),
		})
	}
	return out, nil
}
