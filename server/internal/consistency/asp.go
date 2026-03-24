package consistency

import "context"

// ASP never blocks: updates are applied on arrival and reads are served
// immediately, however stale.
type ASP struct{ *clocks }

func NewASP(n int) *ASP { return &ASP{newClocks(n)} }

func (a *ASP) Name() string     { return "asp" }
func (a *ASP) Aggregates() bool { return false }
func (a *ASP) Stats() Stats     { return a.stats() }

func (a *ASP) WaitRead(_ context.Context, w int, c int64) (int64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.check(w); err != nil {
		return 0, err
	}
	m := a.min()
	a.staleness[c-m]++
	return m, nil
}

func (a *ASP) Finish(w int, c int64) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.finish(w, c)
}
