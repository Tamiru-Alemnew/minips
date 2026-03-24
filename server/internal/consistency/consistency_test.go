package consistency

import (
	"context"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// run drives n workers for iters iterations each against p, with random
// per-step delays, and calls observe with the slowest worker's progress
// as seen by every read.
func run(t *testing.T, p Policy, n int, iters int64, observe func(w int, c, minDone int64)) {
	t.Helper()
	var wg sync.WaitGroup
	for w := 0; w < n; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(w)))
			for c := int64(0); c < iters; c++ {
				m, err := p.WaitRead(context.Background(), w, c)
				if err != nil {
					t.Error(err)
					return
				}
				observe(w, c, m)
				time.Sleep(time.Duration(rng.Intn(200*(w+1))) * time.Microsecond)
				if err := p.Finish(w, c); err != nil {
					t.Error(err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

func TestSSPBoundsGap(t *testing.T) {
	for _, s := range []int64{0, 1, 2, 4} {
		var mu sync.Mutex
		maxGap := int64(0)
		run(t, NewSSP(6, int(s)), 6, 60, func(_ int, c, minDone int64) {
			mu.Lock()
			maxGap = max(maxGap, c-minDone)
			mu.Unlock()
		})
		if maxGap > s {
			t.Fatalf("s=%d: fastest worker read %d clocks ahead of the slowest", s, maxGap)
		}
	}
}

func TestSSPBlocksFastWorker(t *testing.T) {
	p := NewSSP(2, 1)
	for c := int64(0); c < 2; c++ {
		if _, err := p.WaitRead(context.Background(), 0, c); err != nil {
			t.Fatal(err)
		}
		if err := p.Finish(0, c); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := p.WaitRead(ctx, 0, 2); err == nil {
		t.Fatal("worker 0 read clock 2 while worker 1 had finished nothing (s=1)")
	}
	released := make(chan struct{})
	go func() {
		if _, err := p.WaitRead(context.Background(), 0, 2); err == nil {
			close(released)
		}
	}()
	if err := p.Finish(1, 0); err != nil {
		t.Fatal(err)
	}
	select {
	case <-released:
	case <-time.After(time.Second):
		t.Fatal("worker 0 not released after the slowest worker advanced")
	}
}

func TestBSPAppliesOncePerClock(t *testing.T) {
	const n, iters = 5, 40
	applied := map[int64]int{}
	var mu sync.Mutex
	var reads atomic.Int64
	p := NewBSP(n, func(c int64) {
		mu.Lock()
		applied[c]++
		mu.Unlock()
	})
	run(t, p, n, iters, func(w int, c, _ int64) {
		mu.Lock()
		defer mu.Unlock()
		if c > 0 && applied[c-1] != 1 {
			t.Errorf("worker %d read clock %d before clock %d was applied", w, c, c-1)
		}
		reads.Add(1)
	})
	for c := int64(0); c < iters; c++ {
		if applied[c] != 1 {
			t.Fatalf("clock %d applied %d times", c, applied[c])
		}
	}
	if got := reads.Load(); got != n*iters {
		t.Fatalf("reads = %d, want %d", got, n*iters)
	}
}

func TestASPNeverBlocks(t *testing.T) {
	p := NewASP(3)
	for c := int64(0); c < 100; c++ {
		if _, err := p.WaitRead(context.Background(), 0, c); err != nil {
			t.Fatal(err)
		}
		if err := p.Finish(0, c); err != nil {
			t.Fatal(err)
		}
	}
	if got := p.Stats().Staleness[99]; got != 1 {
		t.Fatalf("staleness histogram at 99 = %d, want 1", got)
	}
}

func TestFinishRejectsOutOfOrder(t *testing.T) {
	p := NewSSP(2, 1)
	if err := p.Finish(0, 3); err == nil {
		t.Fatal("expected error for skipped clocks")
	}
	if err := p.Finish(5, 0); err == nil {
		t.Fatal("expected error for unknown worker")
	}
}
