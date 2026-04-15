package codec

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Tamiru-Alemnew/minips/server/internal/params"
	"github.com/Tamiru-Alemnew/minips/server/internal/pb"
)

func TestFloat16(t *testing.T) {
	cases := map[uint16]float32{
		0x0000: 0, 0x3c00: 1, 0xc000: -2, 0x3555: 0.333251953125,
		0x7bff: 65504, 0x0001: 5.960464477539063e-08, 0x0400: 6.103515625e-05,
	}
	for h, want := range cases {
		if got := Float16ToFloat32(h); got != want {
			t.Errorf("%#04x: got %v want %v", h, got, want)
		}
	}
	if !math.IsInf(float64(Float16ToFloat32(0x7c00)), 1) {
		t.Error("0x7c00 should be +inf")
	}
}

func TestDecodeDense(t *testing.T) {
	in := []float32{1, -2.5, 3}
	out, err := Decode(pb.Encoding_DENSE_F32, params.EncodeF32(in), 3)
	if err != nil {
		t.Fatal(err)
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("got %v want %v", out, in)
		}
	}
	if _, err := Decode(pb.Encoding_DENSE_F32, params.EncodeF32(in), 4); err == nil {
		t.Fatal("expected length error")
	}
}

func TestDecodeTopK(t *testing.T) {
	b := make([]byte, 4+6*2)
	binary.LittleEndian.PutUint32(b, 2)
	binary.LittleEndian.PutUint32(b[4:], 1)
	binary.LittleEndian.PutUint32(b[8:], 4)
	binary.LittleEndian.PutUint16(b[12:], 0x3c00)
	binary.LittleEndian.PutUint16(b[14:], 0xc000)
	out, err := Decode(pb.Encoding_TOPK, b, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 1, 0, 0, -2}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("got %v want %v", out, want)
		}
	}
	binary.LittleEndian.PutUint32(b[8:], 9)
	if _, err := Decode(pb.Encoding_TOPK, b, 5); err == nil {
		t.Fatal("expected out-of-range error")
	}
}

func TestDecodeInt8(t *testing.T) {
	b := make([]byte, 4+3)
	binary.LittleEndian.PutUint32(b, math.Float32bits(0.5))
	b[4], b[5], b[6] = 2, 0xff, 127
	out, err := Decode(pb.Encoding_INT8, b, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, -0.5, 63.5}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("got %v want %v", out, want)
		}
	}
}
