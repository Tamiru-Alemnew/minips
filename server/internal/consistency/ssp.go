package consistency

import "context"

// SSP applies updates on arrival and lets a worker at clock c read only
// while c - min(done) <= s, so the fastest worker is never more than s
// iterations ahead of the slowest.
type SSP struct {
	*clocks
	s int64
}

func NewSSP(n, s int) *SSP { return &SSP{clocks: newClocks(n), s: int64(s)} }

func (p *SSP) Name() string     { return "ssp" }
func (p *SSP) Aggregates() bool { return false }
func (p *SSP) Stats() Stats     { return p.stats() }

func (p *SSP) WaitRead(ctx context.Context, w int, c int64) (int64, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.check(w); err != nil {
		return 0, err
	}
	if err := p.waitUntil(ctx, w, func() bool { return c-p.min() <= p.s }); err != nil {
		return 0, err
	}
	m := p.min()
	p.staleness[c-m]++
	return m, nil
}

func (p *SSP) Finish(w int, c int64) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.finish(w, c)
}
