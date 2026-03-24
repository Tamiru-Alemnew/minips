package metrics

import (
	"context"
	"strconv"
	"sync/atomic"

	"google.golang.org/grpc/stats"
)

const WorkerHeader = "worker-id"

type Worker struct {
	BytesIn, BytesOut, Pulls, Pushes atomic.Int64
}

// Registry counts wire bytes per worker. It is installed as a gRPC stats
// handler, so the numbers are the payload bytes gRPC actually framed and
// sent, not tensor sizes.
type Registry struct {
	workers []Worker
	Updates atomic.Int64
}

func New(n int) *Registry { return &Registry{workers: make([]Worker, n)} }

func (r *Registry) Worker(w int) *Worker {
	if w < 0 || w >= len(r.workers) {
		return nil
	}
	return &r.workers[w]
}

func (r *Registry) N() int { return len(r.workers) }

type rpcKey struct{}

type rpcState struct{ w *Worker }

func (r *Registry) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context {
	return context.WithValue(ctx, rpcKey{}, &rpcState{})
}

func (r *Registry) HandleRPC(ctx context.Context, s stats.RPCStats) {
	st, _ := ctx.Value(rpcKey{}).(*rpcState)
	if st == nil {
		return
	}
	switch e := s.(type) {
	case *stats.InHeader:
		if v := e.Header.Get(WorkerHeader); len(v) == 1 {
			if id, err := strconv.Atoi(v[0]); err == nil {
				st.w = r.Worker(id)
			}
		}
	case *stats.InPayload:
		if st.w != nil {
			st.w.BytesIn.Add(int64(e.WireLength))
		}
	case *stats.OutPayload:
		if st.w != nil {
			st.w.BytesOut.Add(int64(e.WireLength))
		}
	}
}

func (r *Registry) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context { return ctx }
func (r *Registry) HandleConn(context.Context, stats.ConnStats)                       {}
