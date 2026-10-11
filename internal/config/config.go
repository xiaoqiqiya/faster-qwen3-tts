package config

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Addr, Backend, State, Directory string
	Concurrency, Pending, Readers   int
	Capacity, Output                int64
	TTL, Task, Send, Sweep          time.Duration
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func Load() (Config, error) {
	c := Config{Addr: env("TTS_LISTEN", ":8000"), Backend: env("TTS_BACKEND_URL", "http://backend:8001"), State: env("TTS_BACKEND_STATE", "/state/backend.json"), Directory: env("TTS_CACHE_DIR", "/cache/audio")}
	values := []struct {
		name               string
		fallback, min, max int64
		set                func(int64)
	}{
		{"TTS_MAX_CONCURRENT_TASKS", 4, 1, 256, func(n int64) { c.Concurrency = int(n) }},
		{"TTS_MAX_PENDING_TASKS", 8, 0, 10000, func(n int64) { c.Pending = int(n) }},
		{"TTS_MAX_READERS", 256, 1, 100000, func(n int64) { c.Readers = int(n) }},
		{"TTS_CACHE_MAX_MB", 256, 1, 1048576, func(n int64) { c.Capacity = n * 1024 * 1024 }},
		{"TTS_CACHE_TTL_SECONDS", 300, 1, 31536000, func(n int64) { c.TTL = time.Duration(n) * time.Second }},
		{"TTS_TASK_TIMEOUT_SECONDS", 120, 1, 86400, func(n int64) { c.Task = time.Duration(n) * time.Second }},
		{"TTS_SEND_TIMEOUT_SECONDS", 15, 1, 3600, func(n int64) { c.Send = time.Duration(n) * time.Second }},
		{"TTS_CACHE_CLEANUP_SECONDS", 5, 1, 3600, func(n int64) { c.Sweep = time.Duration(n) * time.Second }},
	}
	for _, v := range values {
		n, err := strconv.ParseInt(env(v.name, strconv.FormatInt(v.fallback, 10)), 10, 64)
		if err != nil || n < v.min || n > v.max {
			return c, fmt.Errorf("%s 必须在 %d 到 %d 之间", v.name, v.min, v.max)
		}
		v.set(n)
	}
	c.Output = min(int64(8*1024*1024), c.Capacity)
	c.Backend = strings.TrimRight(c.Backend, "/")
	u, err := url.Parse(c.Backend)
	if err != nil || u.Scheme != "http" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return c, fmt.Errorf("TTS_BACKEND_URL 必须是有效的内部 HTTP 地址")
	}
	return c, nil
}
