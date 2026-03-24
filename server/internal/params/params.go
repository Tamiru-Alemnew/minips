package params

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

const magic = "MPS1"

type Spec struct {
	Name  string  `json:"name"`
	Shape []int64 `json:"shape"`
	Dtype string  `json:"dtype"`
}

func (s Spec) Numel() int {
	n := 1
	for _, d := range s.Shape {
		n *= int(d)
	}
	return n
}

type Tensor struct {
	Spec
	Data []float32
}

// Load reads the initial weights written by `python -m minips.init`:
// "MPS1" | uint32 header length | JSON []Spec | float32 little-endian data in spec order.
func Load(path string) ([]Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

func Read(r io.Reader) ([]Tensor, error) {
	head := make([]byte, 8)
	if _, err := io.ReadFull(r, head); err != nil {
		return nil, err
	}
	if string(head[:4]) != magic {
		return nil, fmt.Errorf("bad magic %q", head[:4])
	}
	hdr := make([]byte, binary.LittleEndian.Uint32(head[4:]))
	if _, err := io.ReadFull(r, hdr); err != nil {
		return nil, err
	}
	var specs []Spec
	if err := json.Unmarshal(hdr, &specs); err != nil {
		return nil, err
	}
	out := make([]Tensor, len(specs))
	for i, s := range specs {
		if s.Dtype != "float32" {
			return nil, fmt.Errorf("%s: unsupported dtype %s", s.Name, s.Dtype)
		}
		buf := make([]byte, 4*s.Numel())
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("%s: %w", s.Name, err)
		}
		out[i] = Tensor{Spec: s, Data: DecodeF32(buf)}
	}
	return out, nil
}

func DecodeF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func EncodeF32(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}
