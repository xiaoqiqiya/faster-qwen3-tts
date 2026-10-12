package httpapi

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"tts/internal/cache"
	"tts/internal/config"
	"tts/internal/engine"
)

func TestRequestLogging(t *testing.T) {
	var output bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&output)
	defer log.SetOutput(previous)
	g := New(config.Config{}, nil, nil)
	seen := map[string]bool{}
	for _, tc := range []struct {
		method string
		path   string
		status int
	}{
		{"GET", "/", 200},
		{"GET", "/missing", 404},
		{"GET", "/v1/audio/speech", 405},
		{"POST", "/v1/audio/speech", 400},
	} {
		output.Reset()
		w := httptest.NewRecorder()
		r := httptest.NewRequest(tc.method, tc.path+"?token=secret", strings.NewReader(`{}`))
		g.ServeHTTP(w, r)
		for _, want := range []string{"请求开始", "请求结束", "method=" + tc.method, fmt.Sprintf("path=%q", tc.path), fmt.Sprintf("status=%d", tc.status), "duration="} {
			if !strings.Contains(output.String(), want) {
				t.Fatalf("missing %q in log: %s", want, output.String())
			}
		}
		ids := regexp.MustCompile(`trackid=([A-Z2-7]+)`).FindAllStringSubmatch(output.String(), -1)
		if len(ids) < 2 || seen[ids[0][1]] {
			t.Fatalf("missing or reused trackid: %s", output.String())
		}
		for _, id := range ids {
			if id[1] != ids[0][1] {
				t.Fatalf("trackid changed during request: %s", output.String())
			}
		}
		seen[ids[0][1]] = true
		if w.Code != tc.status || strings.Contains(output.String(), "secret") || strings.Contains(output.String(), "bytes=") {
			t.Fatalf("response status or log privacy: %d %s", w.Code, output.String())
		}
	}
}

func TestHTTPStreamsShareDiskAndWrapErrors(t *testing.T) {
	var logs bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&logs)
	defer log.SetOutput(previous)
	var calls atomic.Int32
	info := engine.Info{Identity: strings.Repeat("a", 64), TaskType: "CustomVoice", Voices: []string{"vivian"}, DefaultVoice: "vivian", Languages: []string{"Chinese"}, SampleRate: 24000, MaxTokens: 1024, Concurrency: 2}
	info.ServedModel = "tts-" + info.Identity
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(200)
		case "/v1/models":
			fmt.Fprintf(w, `{"data":[{"id":%q}]}`, info.ServedModel)
		case "/v1/audio/speech":
			calls.Add(1)
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			if req["model"] != info.ServedModel || req["stream"] != true || req["voice"] != "vivian" {
				t.Error("invalid upstream request", req)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "event: speech.audio.delta\ndata: {\"type\":\"speech.audio.delta\",\"audio\":\"AQIDBA==\",\"response_format\":\"pcm\"}\n\nevent: speech.audio.done\ndata: {\"type\":\"speech.audio.done\"}\n\n")
		default:
			w.WriteHeader(404)
		}
	}))
	defer upstream.Close()
	cfg := config.Config{Backend: upstream.URL, State: filepath.Join(t.TempDir(), "backend.json"), Directory: t.TempDir(), Concurrency: 2, Pending: 2, Readers: 10, Capacity: 32768, Output: 4096, TTL: time.Hour, Task: 5 * time.Second, Send: time.Second, Sweep: time.Hour}
	data, _ := json.Marshal(info)
	if err := os.WriteFile(cfg.State, data, 0600); err != nil {
		t.Fatal(err)
	}
	b := engine.New(cfg)
	store, err := cache.New(cfg, b.Generate)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	server := httptest.NewServer(New(cfg, b, store))
	defer server.Close()
	client := http.Client{Timeout: 5 * time.Second}
	post := func(path, body string) (int, []byte) {
		t.Helper()
		resp, e := client.Post(server.URL+path, "application/json", strings.NewReader(body))
		if e != nil {
			t.Fatal(e)
		}
		defer resp.Body.Close()
		data, e := io.ReadAll(resp.Body)
		if e != nil {
			t.Fatal(e)
		}
		return resp.StatusCode, data
	}
	code, body := post("/v1/audio/speech", `{"input":"你好"}`)
	if code != 200 || len(body) != 14 || body[0] != 1 || binary.BigEndian.Uint32(body[1:5]) != 4 || body[9] != 2 {
		t.Fatalf("framed: %d %v", code, body)
	}
	code, body = post("/v1/audio/speech", `{"input":"你好","response_format":"pcm"}`)
	if code != 200 || !bytes.Equal(body, []byte{1, 2, 3, 4}) {
		t.Fatalf("PCM: %d %v", code, body)
	}
	code, body = post("/v1/audio/speech/sse", `{"input":"你好"}`)
	if code != 200 || !bytes.Contains(body, []byte(`"type":"done"`)) {
		t.Fatalf("SSE: %d %s", code, body)
	}
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var e struct {
			Data struct {
				Type  string `json:"type"`
				Audio string `json:"audio_b64"`
			} `json:"data"`
		}
		if err = json.Unmarshal([]byte(line[6:]), &e); err != nil {
			t.Fatal(err)
		}
		if e.Data.Type == "chunk" {
			audio, e := base64.StdEncoding.DecodeString(e.Data.Audio)
			if e != nil || len(audio) != 48 || string(audio[:4]) != "RIFF" {
				t.Fatal("invalid independent WAV")
			}
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("cache missed across transports: %d", calls.Load())
	}
	for _, bad := range []string{`{}`, `{"input":" "}`, `{"input":"x","speed":2}`, `{"input":"x","extra":1}`, `{"input":"x","input":"y"}`, `{"input":"x","language":null}`, `{"input":"x"} {}`} {
		code, body = post("/v1/audio/speech", bad)
		var e envelope
		if err = json.Unmarshal(body, &e); err != nil || code != 400 || e.Code != code || e.Message == "" {
			t.Fatalf("validation: %s %s", bad, body)
		}
	}
	code, _ = post("/v1/audio/speech", `{"input":"x","voice":"missing"}`)
	if code != 404 {
		t.Fatal(code)
	}
	resp, err := client.Get(server.URL + "/v1/audio/voices")
	if err != nil {
		t.Fatal(err)
	}
	var voices map[string]any
	err = json.NewDecoder(resp.Body).Decode(&voices)
	resp.Body.Close()
	if err != nil || voices["rows"] == nil || voices["data"] != nil {
		t.Fatal(voices, err)
	}
	// A new process must verify the engine before using old disk audio.
	unverified := engine.New(cfg)
	upstream.Close()
	if _, err = unverified.Info(context.Background()); err == nil || unverified.CachedInfo() != nil {
		t.Fatal("unverified identity trusted")
	}
	code, body = post("/v1/audio/speech", `{"input":"你好","response_format":"pcm"}`)
	if code != 200 || len(body) != 4 {
		t.Fatalf("cached fallback: %d %s", code, body)
	}
	code, _ = post("/v1/audio/speech", `{"input":"不同内容"}`)
	if code != 503 {
		t.Fatal("offline cache miss admitted", code)
	}
	info.Identity = strings.Repeat("b", 64)
	info.ServedModel = "tts-" + info.Identity
	data, _ = json.Marshal(info)
	if err = os.WriteFile(cfg.State, data, 0600); err != nil {
		t.Fatal(err)
	}
	code, _ = post("/v1/audio/speech", `{"input":"你好"}`)
	if code != 503 {
		t.Fatal("served stale audio while replacement engine was not ready", code)
	}
	server.Close()
	if !strings.Contains(logs.String(), `params={"input":"你好","voice":"","language":"Chinese","instruct":"","speed":1,"response_format":"pcm_framed"}`) {
		t.Fatalf("missing request parameters: %s", logs.String())
	}
}
