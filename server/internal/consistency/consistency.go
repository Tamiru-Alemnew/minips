package consistency

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Policy decides when a worker may read the weights for a given clock.
// A worker's clock c is the iteration it is about to start; Finish(w, c)
// marks iteration c of worker w as complete.
type Policy interface {
	Name() string
	Aggregates() bool
	WaitRead(ctx context.Context, w int, c int64) (minDone int64, err error)
	Finish(w int, c int64) error
	Stats() Stats
}

type Stats struct {
	WaitSeconds []float64
	Staleness   map[int64]int64
}

func New(mode string, n, staleness int, apply func(c int64)) (Policy, error) {
	switch mode {
	case "bsp":
		return NewBSP(n, apply), nil
	case "ssp":
		return NewSSP(n, staleness), nil
	case "asp":
		return NewASP(n), nil
	}
	return nil, fmt.Errorf("unknown mode %q (want bsp, ssp or asp)", mode)
}

// clocks is the shared vector clock: done[w] is the number of iterations
// worker w has finished.
type clocks struct {
	mu        sync.Mutex
	cond      *sync.Cond
	done      []int64
	wait      []time.Duration
	staleness map[int64]int64
}

func newClocks(n int) *clocks {
	c := &clocks{done: make([]int64, n), wait: make([]time.Duration, n), staleness: map[int64]int64{}}
	c.cond = sync.NewCond(&c.mu)
	return c
}

func (c *clocks) min() int64 {
	m := c.done[0]
	for _, d := range c.done[1:] {
		m = min(m, d)
	}
	return m
}

// waitUntil blocks with c.mu held until ok() is true or ctx is cancelled.
func (c *clocks) waitUntil(ctx context.Context, w int, ok func() bool) error {
	if ok() {
		return nil
	}
	start := time.Now()
	stop := context.AfterFunc(ctx, func() {
		c.mu.Lock()
		c.cond.Broadcast()
		c.mu.Unlock()
	})
	defer stop()
	for !ok() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		c.cond.Wait()
	}
	c.wait[w] += time.Since(start)
	return nil
}

func (c *clocks) finish(w int, it int64) error {
	if w < 0 || w >= len(c.done) {
		return fmt.Errorf("worker %d out of range", w)
	}
	if it != c.done[w] {
		return fmt.Errorf("worker %d finished clock %d, expected %d", w, it, c.done[w])
	}
	c.done[w]++
	c.cond.Broadcast()
	return nil
}

func (c *clocks) check(w int) error {
	if w < 0 || w >= len(c.done) {
		return fmt.Errorf("worker %d out of range", w)
	}
	return nil
}

func (c *clocks) stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := Stats{WaitSeconds: make([]float64, len(c.wait)), Staleness: map[int64]int64{}}
	for i, d := range c.wait {
		s.WaitSeconds[i] = d.Seconds()
	}
	for k, v := range c.staleness {
		s.Staleness[k] = v
	}
	return s
}
