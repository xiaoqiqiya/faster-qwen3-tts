package engine

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/xiaoqiqiya/faster-qwen3-tts/internal/config"
	"io"
	"mime"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type Info struct {
	Identity         string   `json:"identity"`
	Model            string   `json:"model"`
	ServedModel      string   `json:"served_model"`
	TaskType         string   `json:"task_type"`
	Voices           []string `json:"voices"`
	DefaultVoice     string   `json:"default_voice"`
	Languages        []string `json:"languages"`
	SampleRate       int      `json:"sample_rate"`
	SupportsInstruct bool     `json:"supports_instruct"`
	MaxTokens        int      `json:"max_new_tokens"`
	Concurrency      int      `json:"concurrency"`
}

type Client struct {
	c      config.Config
	client *http.Client
	mu     sync.Mutex
	last   *Info
}

func New(c config.Config) *Client {
	return &Client{c: c, client: &http.Client{Transport: &http.Transport{Proxy: nil, DialContext: (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext, MaxIdleConns: 256, MaxIdleConnsPerHost: 256, ResponseHeaderTimeout: c.Task}, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
}

func (b *Client) Info(ctx context.Context) (*Info, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	f, err := os.Open(b.c.State)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var info Info
	if err = json.NewDecoder(io.LimitReader(f, 1<<20)).Decode(&info); err != nil {
		return nil, err
	}
	if len(info.Identity) != 64 || info.ServedModel != "tts-"+info.Identity || info.SampleRate != 24000 || len(info.Voices) == 0 || info.Concurrency != b.c.Concurrency || (info.TaskType != "Base" && info.TaskType != "CustomVoice") {
		return nil, errors.New("推理服务元数据或并发配置不一致")
	}
	for _, path := range []string{"/health", "/v1/models"} {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, b.c.Backend+path, nil)
		resp, e := b.client.Do(req)
		if e != nil {
			return nil, e
		}
		if resp.StatusCode != 200 {
			resp.Body.Close()
			return nil, fmt.Errorf("推理服务未就绪: %d", resp.StatusCode)
		}
		if path == "/v1/models" {
			var models struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			e = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&models)
			found := false
			for _, m := range models.Data {
				if m.ID == info.ServedModel {
					found = true
				}
			}
			if e != nil || !found {
				resp.Body.Close()
				return nil, errors.New("运行中的模型与元数据不一致")
			}
		}
		resp.Body.Close()
	}
	b.mu.Lock()
	b.last = &info
	b.mu.Unlock()
	return &info, nil
}

func (b *Client) CachedInfo() *Info { b.mu.Lock(); defer b.mu.Unlock(); return b.last }

func Key(req Request, info *Info) string {
	req.Format = ""
	data, _ := json.Marshal(struct {
		Request  Request
		Identity string
	}{req, info.Identity})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func (b *Client) Generate(ctx context.Context, req Request, info *Info, emit func([]byte) error) error {
	live, err := b.Info(ctx)
	if err != nil || live.Identity != info.Identity {
		return errors.New("推理服务不可用或排队期间模型已变更")
	}
	body, _ := json.Marshal(map[string]any{"model": info.ServedModel, "input": req.Input, "voice": req.Voice, "language": req.Language, "instructions": req.Instruct, "task_type": info.TaskType, "stream": true, "stream_format": "sse", "response_format": "pcm", "max_new_tokens": info.MaxTokens})
	r, _ := http.NewRequestWithContext(ctx, http.MethodPost, b.c.Backend+"/v1/audio/speech", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	resp, err := b.client.Do(r)
	if err != nil {
		return errors.New("无法连接推理服务")
	}
	defer resp.Body.Close()
	media, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if resp.StatusCode != 200 || media != "text/event-stream" {
		return fmt.Errorf("推理服务拒绝请求，状态码 %d", resp.StatusCode)
	}
	if err = decodeSSE(resp.Body, emit); err != nil {
		return err
	}
	live, err = b.Info(ctx)
	if err != nil || live.Identity != info.Identity {
		return errors.New("生成期间推理服务身份失效")
	}
	return nil
}

// A terminal event is required: EOF alone must never commit a reusable cache entry.
func decodeSSE(reader io.Reader, emit func([]byte) error) error {
	s := bufio.NewScanner(reader)
	s.Buffer(make([]byte, 4096), 2<<20)
	var data strings.Builder
	event := ""
	done := false
	audio := false
	dispatch := func() error {
		if data.Len() == 0 {
			event = ""
			return nil
		}
		if done {
			return errors.New("推理服务在完成后继续发送数据")
		}
		var msg struct {
			Type   string `json:"type"`
			Audio  string `json:"audio"`
			Format string `json:"response_format"`
		}
		if err := json.Unmarshal([]byte(data.String()), &msg); err != nil {
			return errors.New("推理服务返回无效事件")
		}
		if event != "" && event != msg.Type {
			return errors.New("推理事件类型不一致")
		}
		switch msg.Type {
		case "speech.audio.delta":
			pcm, err := base64.StdEncoding.DecodeString(msg.Audio)
			if err != nil || len(pcm) == 0 || len(pcm)%2 != 0 || msg.Format != "pcm" {
				return errors.New("推理服务返回无效 PCM 音频")
			}
			for len(pcm) > 0 {
				n := min(len(pcm), 262144)
				if err = emit(pcm[:n]); err != nil {
					return err
				}
				pcm = pcm[n:]
			}
			audio = true
		case "speech.audio.done":
			if !audio {
				return errors.New("推理服务未生成音频")
			}
			done = true
		case "speech.audio.error":
			return errors.New("推理服务报告生成失败，请丢弃已接收音频")
		default:
			return errors.New("推理服务返回未知事件")
		}
		data.Reset()
		event = ""
		return nil
	}
	for s.Scan() {
		line := s.Text()
		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		key, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch key {
		case "event":
			event = value
		case "data":
			if data.Len()+len(value)+1 > 2<<20 {
				return errors.New("推理事件超过大小限制")
			}
			data.WriteString(value)
			data.WriteByte('\n')
		}
	}
	if s.Err() != nil {
		return errors.New("推理音频流读取失败")
	}
	if data.Len() != 0 || !done {
		return errors.New("推理音频流中断或缺少完成事件")
	}
	return nil
}

type Request struct {
	Input    string  `json:"input"`
	Voice    string  `json:"voice"`
	Language string  `json:"language"`
	Instruct string  `json:"instruct"`
	Speed    float64 `json:"speed"`
	Format   string  `json:"response_format"`
}
