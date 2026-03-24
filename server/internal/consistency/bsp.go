package consistency

import "context"

// BSP accumulates every worker's gradient for clock c, applies the average
// once, and only then lets anyone read clock c+1.
type BSP struct {
	*clocks
	n       int
	arrived map[int64]int
	applied int64
	apply   func(c int64)
}

func NewBSP(n int, apply func(c int64)) *BSP {
	return &BSP{clocks: newClocks(n), n: n, arrived: map[int64]int{}, apply: apply}
}

func (b *BSP) Name() string     { return "bsp" }
func (b *BSP) Aggregates() bool { return true }
func (b *BSP) Stats() Stats     { return b.stats() }

func (b *BSP) WaitRead(ctx context.Context, w int, c int64) (int64, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.check(w); err != nil {
		return 0, err
	}
	if err := b.waitUntil(ctx, w, func() bool { return b.applied >= c }); err != nil {
		return 0, err
	}
	b.staleness[c-b.applied]++
	return b.applied, nil
}

func (b *BSP) Finish(w int, c int64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.finish(w, c); err != nil {
		return err
	}
	b.arrived[c]++
	if b.arrived[c] == b.n {
		delete(b.arrived, c)
		b.apply(c)
		b.applied = c + 1
		b.cond.Broadcast()
	}
	return nil
}
