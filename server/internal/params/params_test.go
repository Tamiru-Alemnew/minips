package params

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestRead(t *testing.T) {
	hdr := []byte(`[{"name":"a","shape":[2],"dtype":"float32"},{"name":"b","shape":[1,1],"dtype":"float32"}]`)
	var buf bytes.Buffer
	buf.WriteString(magic)
	binary.Write(&buf, binary.LittleEndian, uint32(len(hdr)))
	buf.Write(hdr)
	buf.Write(EncodeF32([]float32{1, 2, 3}))
	ts, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(ts) != 2 || ts[0].Data[1] != 2 || ts[1].Data[0] != 3 || ts[1].Numel() != 1 {
		t.Fatalf("got %+v", ts)
	}
}
