package server

import (
	"math"
	"testing"
)

func TestLR(t *testing.T) {
	cases := []struct {
		c, warmup, total int64
		want             float64
	}{
		{0, 4, 104, 0.25},
		{3, 4, 104, 1},
		{4, 4, 104, 1},
		{54, 4, 104, 0.5},
		{104, 4, 104, 0},
		{500, 0, 0, 1},
	}
	for _, c := range cases {
		if got := LR(1, c.c, c.warmup, c.total); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("LR(c=%d, warmup=%d, total=%d) = %v, want %v", c.c, c.warmup, c.total, got, c.want)
		}
	}
}
