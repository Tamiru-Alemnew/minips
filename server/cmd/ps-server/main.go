package main

import (
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"

	"github.com/teim/minips/server/internal/params"
	"github.com/teim/minips/server/internal/pb"
	"github.com/teim/minips/server/internal/server"
	"github.com/teim/minips/server/internal/shard"
)

func main() {
	var (
		addr      = flag.String("addr", "127.0.0.1:50051", "listen address")
		initPath  = flag.String("init", "", "initial weights written by python -m minips.init")
		shards    = flag.Int("shards", 4, "number of parameter shards")
		workers   = flag.Int("workers", 8, "number of workers")
		mode      = flag.String("mode", "bsp", "bsp, ssp or asp")
		staleness = flag.Int("staleness", 0, "staleness bound for ssp")
		lr        = flag.Float64("lr", 0.1, "peak learning rate")
		steps     = flag.Int64("total-steps", 0, "clocks in the cosine schedule (0 = constant lr)")
		momentum  = flag.Float64("momentum", 0.9, "SGD momentum")
		wd        = flag.Float64("weight-decay", 5e-4, "weight decay on weight matrices")
	)
	flag.Parse()
	if *initPath == "" {
		log.Fatal("--init is required")
	}
	tensors, err := params.Load(*initPath)
	if err != nil {
		log.Fatalf("load %s: %v", *initPath, err)
	}
	ps, err := server.New(server.Config{
		Workers: *workers, Shards: *shards, Mode: *mode, Staleness: *staleness,
		LR: *lr, TotalSteps: *steps,
		Optim: shard.Optim{Momentum: float32(*momentum), WeightDecay: float32(*wd)},
	}, tensors)
	if err != nil {
		log.Fatal(err)
	}
	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatal(err)
	}
	g := grpc.NewServer(
		grpc.StatsHandler(ps.Metrics),
		grpc.MaxRecvMsgSize(server.ChunkBytes+(1<<20)),
	)
	pb.RegisterParameterServerServer(g, ps)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		g.Stop()
	}()
	log.Printf("serving %d tensors on %s: mode=%s staleness=%d workers=%d shards=%d",
		len(tensors), *addr, *mode, *staleness, *workers, *shards)
	if err := g.Serve(lis); err != nil {
		log.Fatal(err)
	}
	ps.Close()
}
