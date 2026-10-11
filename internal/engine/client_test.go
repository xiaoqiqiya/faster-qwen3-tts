package engine

import (
	"bytes"
	"encoding/base64"
	"io"
	"strings"
	"testing"
)

type fragmented struct{ r io.Reader }

func (f fragmented) Read(p []byte) (int, error) { return f.r.Read(p[:min(len(p), 1)]) }
func delta(audio []byte) string {
	return "event: speech.audio.delta\r\ndata: {\"type\":\"speech.audio.delta\",\"response_format\":\"pcm\",\"audio\":\"" + base64.StdEncoding.EncodeToString(audio) + "\"}\r\n\r\n"
}

const done = "event: speech.audio.done\ndata: {\"type\":\"speech.audio.done\"}\n\n"

func TestSSEFragmentedAndTerminalValidation(t *testing.T) {
	valid := delta([]byte{1, 2, 3, 4}) + done
	var out bytes.Buffer
	if e := decodeSSE(fragmented{strings.NewReader(valid)}, func(b []byte) error { _, e := out.Write(b); return e }); e != nil {
		t.Fatal(e)
	}
	if !bytes.Equal(out.Bytes(), []byte{1, 2, 3, 4}) {
		t.Fatal(out.Bytes())
	}
	for name, s := range map[string]string{"missing_done": delta([]byte{1, 2}), "empty": done, "odd": delta([]byte{1}) + done, "late": valid + delta([]byte{1, 2}), "duplicate_done": valid + done, "error": delta([]byte{1, 2}) + "data: {\"type\":\"speech.audio.error\"}\n\n", "truncated": strings.TrimRight(valid, "\n"), "large": "data: " + strings.Repeat("a", 2<<20) + "\n\n"} {
		t.Run(name, func(t *testing.T) {
			if e := decodeSSE(strings.NewReader(s), func([]byte) error { return nil }); e == nil {
				t.Fatal("accepted corrupt stream")
			}
		})
	}
}

func TestKeyIgnoresTransportButIncludesGenerationIdentity(t *testing.T) {
	r := Request{Input: "你好", Voice: "vivian", Language: "Chinese", Speed: 1, Format: "pcm"}
	i := &Info{Identity: "a"}
	k := Key(r, i)
	r.Format = "pcm_framed"
	if Key(r, i) != k {
		t.Fatal("format changes cache key")
	}
	i.Identity = "b"
	if Key(r, i) == k {
		t.Fatal("model identity omitted")
	}
	i.Identity = "a"
	r.Voice = "ryan"
	if Key(r, i) == k {
		t.Fatal("voice omitted")
	}
}
