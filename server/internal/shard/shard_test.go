package shard

import (
	"math"
	"testing"

	"github.com/Tamiru-Alemnew/minips/server/internal/params"
)

func tensors() []params.Tensor {
	return []params.Tensor{
		{Spec: params.Spec{Name: "w", Shape: []int64{2, 2}, Dtype: "float32"}, Data: []float32{1, 1, 1, 1}},
		{Spec: params.Spec{Name: "b", Shape: []int64{2}, Dtype: "float32"}, Data: []float32{0, 0}},
	}
}

func read(t *testing.T, g *Group, name string) []float32 {
	t.Helper()
	b, err := g.Read(name)
	if err != nil {
		t.Fatal(err)
	}
	return params.DecodeF32(b)
}

func near(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }

func TestMomentumAndDecay(t *testing.T) {
	g := NewGroup(3, tensors(), Optim{Momentum: 0.9, WeightDecay: 0.1})
	defer g.Close()
	for i := 0; i < 2; i++ {
		if err := g.Apply("w", []float32{1, 0, 0, 0}, 1, 0.5); err != nil {
			t.Fatal(err)
		}
		if err := g.Apply("b", []float32{1, 1}, 1, 0.5); err != nil {
			t.Fatal(err)
		}
	}
	// w[0]: d=1.1 -> m=1.1, w=0.45; d=1+0.045=1.045 -> m=0.99+1.045=2.035, w=0.45-1.0175
	if w := read(t, g, "w"); !near(w[0], 0.45-1.0175) || !near(w[1], 1-0.05-0.5*(0.09+0.1*0.95)) {
		t.Fatalf("w = %v", w)
	}
	// biases get no weight decay: m=1 then 1.9; b = -0.5 - 0.95
	if b := read(t, g, "b"); !near(b[0], -1.45) {
		t.Fatalf("b = %v", b)
	}
}

func TestAccumulateFlushAveragesOnce(t *testing.T) {
	g := NewGroup(2, tensors(), Optim{})
	defer g.Close()
	for _, v := range []float32{1, 3} {
		if err := g.Accumulate("b", []float32{v, v}); err != nil {
			t.Fatal(err)
		}
	}
	if n := g.Flush(0.5, 1); n != 1 {
		t.Fatalf("flushed %d tensors, want 1", n)
	}
	if b := read(t, g, "b"); !near(b[0], -2) {
		t.Fatalf("b = %v, want -2", b)
	}
	if n := g.Flush(0.5, 1); n != 0 {
		t.Fatalf("second flush applied %d tensors", n)
	}
}

func TestRejectsBadInput(t *testing.T) {
	g := NewGroup(1, tensors(), Optim{})
	defer g.Close()
	if err := g.Apply("nope", []float32{1}, 1, 1); err == nil {
		t.Fatal("expected unknown tensor error")
	}
	if err := g.Apply("b", []float32{1, 2, 3}, 1, 1); err == nil {
		t.Fatal("expected size error")
	}
}

func TestStableSharding(t *testing.T) {
	if Index("layer1.0.conv1.weight", 4) != Index("layer1.0.conv1.weight", 4) {
		t.Fatal("hash not stable")
	}
}
