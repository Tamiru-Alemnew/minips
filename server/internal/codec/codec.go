package codec

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/teim/minips/server/internal/params"
	"github.com/teim/minips/server/internal/pb"
)

// Decode turns one tensor's wire payload into a dense gradient of length n.
//
//	DENSE_F32: n float32
//	TOPK:      uint32 k | k uint32 indices | k float16 values
//	INT8:      float32 scale | n int8
func Decode(enc pb.Encoding, data []byte, n int) ([]float32, error) {
	switch enc {
	case pb.Encoding_DENSE_F32:
		if len(data) != 4*n {
			return nil, fmt.Errorf("dense: got %d bytes, want %d", len(data), 4*n)
		}
		return params.DecodeF32(data), nil
	case pb.Encoding_TOPK:
		return decodeTopK(data, n)
	case pb.Encoding_INT8:
		if len(data) != 4+n {
			return nil, fmt.Errorf("int8: got %d bytes, want %d", len(data), 4+n)
		}
		scale := math.Float32frombits(binary.LittleEndian.Uint32(data))
		out := make([]float32, n)
		for i, q := range data[4:] {
			out[i] = float32(int8(q)) * scale
		}
		return out, nil
	}
	return nil, fmt.Errorf("unknown encoding %v", enc)
}

func decodeTopK(data []byte, n int) ([]float32, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("topk: short payload")
	}
	k := int(binary.LittleEndian.Uint32(data))
	if len(data) != 4+6*k {
		return nil, fmt.Errorf("topk: got %d bytes for k=%d", len(data), k)
	}
	idx := data[4 : 4+4*k]
	val := data[4+4*k:]
	out := make([]float32, n)
	for j := 0; j < k; j++ {
		i := int(binary.LittleEndian.Uint32(idx[4*j:]))
		if i >= n {
			return nil, fmt.Errorf("topk: index %d out of range %d", i, n)
		}
		out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(val[2*j:]))
	}
	return out, nil
}

func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	case exp != 0:
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	case frac == 0:
		return math.Float32frombits(sign)
	}
	e := uint32(127 - 15 + 1)
	for frac&0x400 == 0 {
		frac <<= 1
		e--
	}
	return math.Float32frombits(sign | e<<23 | (frac&0x3ff)<<13)
}
