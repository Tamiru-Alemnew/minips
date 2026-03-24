package shard

import (
	"fmt"
	"hash/fnv"

	"github.com/teim/minips/server/internal/params"
)

type Optim struct {
	Momentum    float32
	WeightDecay float32
}

type param struct {
	w, mom, acc []float32
	decay       bool
	accN        int
}

type opKind int

const (
	opApply opKind = iota
	opAccumulate
	opFlush
	opRead
)

type op struct {
	kind  opKind
	name  string
	grad  []float32
	scale float32
	lr    float32
	done  chan result
}

type result struct {
	data    []byte
	applied int
	err     error
}

// Shard owns a subset of the weights. All access goes through its goroutine,
// so the update path needs no locks.
type Shard struct {
	ops    chan op
	params map[string]*param
	opt    Optim
}

func New(opt Optim) *Shard {
	return &Shard{ops: make(chan op, 256), params: map[string]*param{}, opt: opt}
}

func (s *Shard) add(t params.Tensor) {
	s.params[t.Name] = &param{
		w:     append([]float32(nil), t.Data...),
		mom:   make([]float32, len(t.Data)),
		decay: len(t.Shape) > 1,
	}
}

func (s *Shard) Run() {
	for o := range s.ops {
		o.done <- s.handle(o)
	}
}

func (s *Shard) Close() { close(s.ops) }

func (s *Shard) do(o op) result {
	o.done = make(chan result, 1)
	s.ops <- o
	return <-o.done
}

func (s *Shard) handle(o op) result {
	if o.kind == opFlush {
		n := 0
		for _, p := range s.params {
			if p.accN > 0 {
				s.step(p, p.acc, o.scale, o.lr)
				clear(p.acc)
				p.accN = 0
				n++
			}
		}
		return result{applied: n}
	}
	p, ok := s.params[o.name]
	if !ok {
		return result{err: fmt.Errorf("unknown tensor %q", o.name)}
	}
	if o.grad != nil && len(o.grad) != len(p.w) {
		return result{err: fmt.Errorf("%s: grad has %d elements, want %d", o.name, len(o.grad), len(p.w))}
	}
	switch o.kind {
	case opApply:
		s.step(p, o.grad, o.scale, o.lr)
		return result{applied: 1}
	case opAccumulate:
		if p.acc == nil {
			p.acc = make([]float32, len(p.w))
		}
		for i, g := range o.grad {
			p.acc[i] += g
		}
		p.accN++
		return result{}
	case opRead:
		return result{data: params.EncodeF32(p.w)}
	}
	return result{err: fmt.Errorf("bad op %d", o.kind)}
}

// step is SGD with momentum; weight decay applies to weight matrices only.
func (s *Shard) step(p *param, grad []float32, scale, lr float32) {
	mu, wd := s.opt.Momentum, s.opt.WeightDecay
	if !p.decay {
		wd = 0
	}
	for i, g := range grad {
		d := g*scale + wd*p.w[i]
		p.mom[i] = mu*p.mom[i] + d
		p.w[i] -= lr * p.mom[i]
	}
}

// Group routes tensors to shards by a stable hash of their name.
type Group struct {
	shards []*Shard
	owner  map[string]*Shard
	Specs  []params.Spec
}

func NewGroup(n int, tensors []params.Tensor, opt Optim) *Group {
	g := &Group{owner: map[string]*Shard{}}
	for i := 0; i < n; i++ {
		s := New(opt)
		g.shards = append(g.shards, s)
	}
	for _, t := range tensors {
		s := g.shards[Index(t.Name, n)]
		s.add(t)
		g.owner[t.Name] = s
		g.Specs = append(g.Specs, t.Spec)
	}
	for _, s := range g.shards {
		go s.Run()
	}
	return g
}

func Index(name string, n int) int {
	h := fnv.New32a()
	h.Write([]byte(name))
	return int(h.Sum32() % uint32(n))
}

func (g *Group) shard(name string) (*Shard, error) {
	s, ok := g.owner[name]
	if !ok {
		return nil, fmt.Errorf("unknown tensor %q", name)
	}
	return s, nil
}

func (g *Group) Apply(name string, grad []float32, scale, lr float32) error {
	s, err := g.shard(name)
	if err != nil {
		return err
	}
	return s.do(op{kind: opApply, name: name, grad: grad, scale: scale, lr: lr}).err
}

func (g *Group) Accumulate(name string, grad []float32) error {
	s, err := g.shard(name)
	if err != nil {
		return err
	}
	return s.do(op{kind: opAccumulate, name: name, grad: grad}).err
}

// Flush applies every accumulated gradient once, scaled by scale, and returns
// the number of tensors updated.
func (g *Group) Flush(scale, lr float32) int {
	done := make(chan result, len(g.shards))
	for _, s := range g.shards {
		s.ops <- op{kind: opFlush, scale: scale, lr: lr, done: done}
	}
	n := 0
	for range g.shards {
		n += (<-done).applied
	}
	return n
}

func (g *Group) Read(name string) ([]byte, error) {
	s, err := g.shard(name)
	if err != nil {
		return nil, err
	}
	r := s.do(op{kind: opRead, name: name})
	return r.data, r.err
}

func (g *Group) Close() {
	for _, s := range g.shards {
		s.Close()
	}
}
