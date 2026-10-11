package httpapi

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"github.com/xiaoqiqiya/faster-qwen3-tts/internal/cache"
	"github.com/xiaoqiqiya/faster-qwen3-tts/internal/config"
	"github.com/xiaoqiqiya/faster-qwen3-tts/internal/engine"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

type envelope struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data"`
}

func reply(w http.ResponseWriter, code int, message string, data any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(envelope{code, message, data})
}

func decodeRequest(w http.ResponseWriter, r *http.Request) (engine.Request, error) {
	req := engine.Request{Language: "Chinese", Speed: 1, Format: "pcm_framed"}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32768))
	if err != nil || !utf8.Valid(body) {
		return req, errors.New("请求体过大或编码无效")
	}
	d := json.NewDecoder(bytes.NewReader(body))
	tok, err := d.Token()
	if err != nil || tok != json.Delim('{') {
		return req, errors.New("请求体必须是 JSON 对象")
	}
	seen := map[string]bool{}
	for d.More() {
		tok, e := d.Token()
		if e != nil {
			return req, e
		}
		name, ok := tok.(string)
		if !ok || seen[name] {
			return req, errors.New("请求字段重复或无效")
		}
		seen[name] = true
		var value json.RawMessage
		if e = d.Decode(&value); e != nil {
			return req, e
		}
		if string(value) == "null" {
			if name == "voice" {
				continue
			}
			return req, errors.New("请求字段不能为 null")
		}
		switch name {
		case "input":
			e = json.Unmarshal(value, &req.Input)
		case "voice":
			e = json.Unmarshal(value, &req.Voice)
		case "language":
			e = json.Unmarshal(value, &req.Language)
		case "instruct":
			e = json.Unmarshal(value, &req.Instruct)
		case "speed":
			e = json.Unmarshal(value, &req.Speed)
		case "response_format":
			e = json.Unmarshal(value, &req.Format)
		default:
			return req, errors.New("请求包含未知字段")
		}
		if e != nil {
			return req, errors.New("请求字段类型无效")
		}
		if name == "voice" && req.Voice == "" {
			return req, errors.New("音色名称不能为空")
		}
	}
	if _, err = d.Token(); err != nil {
		return req, err
	}
	if _, err = d.Token(); err != io.EOF {
		return req, errors.New("请求体包含多余内容")
	}
	if strings.TrimSpace(req.Input) == "" || utf8.RuneCountInString(req.Input) > 2048 || utf8.RuneCountInString(req.Voice) > 128 || len(req.Language) == 0 || utf8.RuneCountInString(req.Language) > 32 || utf8.RuneCountInString(req.Instruct) > 512 || req.Speed != 1 || (req.Format != "pcm" && req.Format != "pcm_framed") {
		return req, errors.New("文本、音色、语言、指令、语速或输出格式无效")
	}
	return req, nil
}

func resolve(req *engine.Request, info *engine.Info) (int, string) {
	if req.Voice == "" {
		req.Voice = info.DefaultVoice
	}
	found := false
	for _, v := range info.Voices {
		if req.Voice == v {
			found = true
		}
	}
	if !found {
		return 404, "指定音色不存在"
	}
	found = false
	for _, v := range info.Languages {
		if strings.EqualFold(req.Language, v) {
			req.Language = v
			found = true
			break
		}
	}
	if !found {
		return 400, "指定语言不受支持"
	}
	if req.Instruct != "" && !info.SupportsInstruct {
		return 400, "当前模型不支持语音指令"
	}
	return 0, ""
}

type Handler struct {
	c     config.Config
	b     *engine.Client
	cache *cache.Store
}

func (g *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	method := "GET"
	switch path {
	case "/", "/health", "/v1/audio/voices":
	case "/v1/audio/speech", "/generate/stream":
		method = "POST"
	default:
		reply(w, 404, "资源不存在", nil)
		return
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		reply(w, 405, "请求方法不允许", nil)
		return
	}
	if path == "/" {
		reply(w, 200, "Qwen3-TTS 流式网关", map[string]string{"health": "/health", "voices": "/v1/audio/voices", "speech": "/v1/audio/speech", "generate_stream": "/generate/stream"})
		return
	}
	if method == "POST" {
		g.speech(w, r, path == "/generate/stream")
		return
	}
	info, err := g.b.Info(r.Context())
	if err != nil {
		reply(w, 503, "推理服务暂不可用或配置不一致", nil)
		return
	}
	if path == "/health" {
		data := g.cache.Stats()
		data["backend"] = info
		reply(w, 200, "流式网关已就绪", data)
		return
	}
	rows := []map[string]any{}
	for _, v := range sortedVoices(info) {
		rows = append(rows, map[string]any{"voice": v, "type": info.TaskType, "default": v == info.DefaultVoice})
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(map[string]any{"code": 200, "message": "音色列表查询成功", "rows": rows})
}

func frame(kind byte, data []byte) []byte {
	buf := make([]byte, 5+len(data))
	buf[0] = kind
	binary.BigEndian.PutUint32(buf[1:5], uint32(len(data)))
	copy(buf[5:], data)
	return buf
}
func wav(pcm []byte) []byte {
	w := make([]byte, 44+len(pcm))
	copy(w, "RIFF")
	binary.LittleEndian.PutUint32(w[4:], uint32(36+len(pcm)))
	copy(w[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(w[16:], 16)
	binary.LittleEndian.PutUint16(w[20:], 1)
	binary.LittleEndian.PutUint16(w[22:], 1)
	binary.LittleEndian.PutUint32(w[24:], 24000)
	binary.LittleEndian.PutUint32(w[28:], 48000)
	binary.LittleEndian.PutUint16(w[32:], 2)
	binary.LittleEndian.PutUint16(w[34:], 16)
	copy(w[36:], "data")
	binary.LittleEndian.PutUint32(w[40:], uint32(len(pcm)))
	copy(w[44:], pcm)
	return w
}
func sse(kind, message string, pcm []byte) []byte {
	data := map[string]any{"type": kind}
	if pcm != nil {
		data["sample_rate"] = 24000
		data["audio_b64"] = base64.StdEncoding.EncodeToString(wav(pcm))
	}
	body, _ := json.Marshal(envelope{200, message, data})
	return append(append([]byte("data: "), body...), '\n', '\n')
}

func (g *Handler) speech(w http.ResponseWriter, r *http.Request, asSSE bool) {
	req, err := decodeRequest(w, r)
	if err != nil {
		reply(w, 400, "请求参数校验失败", map[string]string{"detail": err.Error()})
		return
	}
	info, err := g.b.Info(r.Context())
	allowNew := err == nil
	if err != nil {
		info = g.b.CachedInfo()
		if info == nil {
			reply(w, 503, "推理服务暂不可用", nil)
			return
		}
	}
	if code, msg := resolve(&req, info); code != 0 {
		reply(w, code, msg, nil)
		return
	}
	rec, err := g.cache.Acquire(engine.Key(req, info), req, info, allowNew)
	if err != nil {
		reply(w, 503, err.Error(), nil)
		return
	}
	defer g.cache.Release(rec)
	buffer := make([]byte, 262144)
	n, err := g.cache.Read(r.Context(), rec, 0, buffer)
	if err != nil {
		reply(w, 503, "语音生成失败或请求已取消", nil)
		return
	}
	h := w.Header()
	h.Set("Cache-Control", "no-store")
	h.Set("X-Accel-Buffering", "no")
	h.Set("X-Audio-Sample-Rate", "24000")
	h.Set("X-Audio-Channels", "1")
	h.Set("X-Audio-Sample-Format", "s16le")
	if asSSE {
		h.Set("Content-Type", "text/event-stream")
		h.Set("X-Audio-Protocol", "sse-wav-v1")
	} else if req.Format == "pcm_framed" {
		h.Set("Content-Type", "application/vnd.qwen.pcm-stream")
		h.Set("X-Audio-Protocol", "qwen-pcm-v1")
	} else {
		h.Set("Content-Type", "audio/pcm")
		h.Set("X-Audio-Protocol", "raw")
	}
	rc := http.NewResponseController(w)
	send := func(body []byte) error {
		if e := rc.SetWriteDeadline(time.Now().Add(g.c.Send)); e != nil {
			return e
		}
		if _, e := w.Write(body); e != nil {
			return e
		}
		return rc.Flush()
	}
	defer rc.SetWriteDeadline(time.Time{})
	offset := int64(0)
	for {
		if n > 0 {
			chunk := buffer[:n]
			if asSSE {
				chunk = sse("chunk", "音频片段已生成", chunk)
			} else if req.Format == "pcm_framed" {
				chunk = frame(1, chunk)
			}
			if send(chunk) != nil {
				return
			}
			offset += int64(n)
		}
		n, err = g.cache.Read(r.Context(), rec, offset, buffer)
		if err != nil {
			break
		}
	}
	if errors.Is(err, io.EOF) {
		if asSSE {
			_ = send(sse("done", "语音生成完成", nil))
		} else if req.Format == "pcm_framed" {
			_ = send(frame(2, nil))
		}
		return
	}
	if r.Context().Err() != nil {
		return
	}
	if asSSE {
		_ = send(sse("error", "语音生成失败，请丢弃已接收音频", nil))
	} else if req.Format == "pcm_framed" {
		data, _ := json.Marshal(envelope{500, "语音生成失败，请丢弃已接收音频", nil})
		_ = send(frame(3, data))
	} else {
		panic(http.ErrAbortHandler)
	}
}

func New(c config.Config, b *engine.Client, store *cache.Store) *Handler {
	return &Handler{c, b, store}
}
func sortedVoices(info *engine.Info) []string {
	v := append([]string(nil), info.Voices...)
	sort.Strings(v)
	return v
}
