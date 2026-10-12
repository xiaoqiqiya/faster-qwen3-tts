package cache

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"tts/internal/config"
	"tts/internal/engine"
)

func settings(t *testing.T) config.Config {
	return config.Config{Directory: t.TempDir(), Concurrency: 2, Pending: 1, Readers: 16, Capacity: 32768, Output: 4096, TTL: time.Hour, Task: 5 * time.Second, Sweep: time.Hour}
}
func key(n string) string { return strings.Repeat(n, 64) }
func acquire(t *testing.T, c *Store, k string) *Record {
	t.Helper()
	r, e := c.Acquire(k, engine.Request{Input: k}, &engine.Info{}, true)
	if e != nil {
		t.Fatal(e)
	}
	return r
}
func drain(t *testing.T, c *Store, r *Record) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	buf := make([]byte, 64)
	out := []byte{}
	for {
		n, e := c.Read(ctx, r, int64(len(out)), buf)
		out = append(out, buf[:n]...)
		if e == io.EOF {
			return out
		}
		if e != nil {
			t.Fatal(e)
		}
	}
}
func await(t *testing.T, ch <-chan string) string {
	t.Helper()
	select {
	case s := <-ch:
		return s
	case <-time.After(3 * time.Second):
		t.Fatal("producer did not start")
		return ""
	}
}

func TestParallelSingleFlightReplayAndQueue(t *testing.T) {
	started := make(chan string, 4)
	finish := make(chan struct{})
	var calls atomic.Int32
	c, e := New(settings(t), func(ctx context.Context, r engine.Request, _ *engine.Info, emit func([]byte) error) error {
		calls.Add(1)
		if e := emit([]byte{1, 2}); e != nil {
			return e
		}
		started <- r.Input
		select {
		case <-finish:
			return emit([]byte{3, 4})
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	a := acquire(t, c, key("a"))
	b := acquire(t, c, key("b"))
	await(t, started)
	await(t, started)
	dup := acquire(t, c, key("a"))
	if dup != a {
		t.Fatal("identical request created another producer")
	}
	late := make([]byte, 2)
	n, e := c.Read(context.Background(), dup, 0, late)
	if e != nil || n != 2 || !bytes.Equal(late, []byte{1, 2}) {
		t.Fatal("late subscriber cannot replay first chunk")
	}
	queued := acquire(t, c, key("c"))
	if _, e = c.Acquire(key("d"), engine.Request{}, &engine.Info{}, true); e == nil {
		t.Fatal("queue was not bounded")
	}
	c.Release(a)
	c.Release(b)
	close(finish)
	if got := drain(t, c, dup); !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Fatalf("audio %v", got)
	}
	c.Release(dup)
	_ = drain(t, c, queued)
	c.Release(queued)
	if calls.Load() != 3 {
		t.Fatalf("generations %d", calls.Load())
	}
}

func TestRestartTTLAndCapacity(t *testing.T) {
	cfg := settings(t)
	gen := func(context.Context, engine.Request, *engine.Info, func([]byte) error) error {
		return errors.New("must hit disk")
	}
	c, e := New(cfg, func(_ context.Context, _ engine.Request, _ *engine.Info, emit func([]byte) error) error {
		return emit([]byte{1, 2, 3, 4})
	})
	if e != nil {
		t.Fatal(e)
	}
	r := acquire(t, c, key("a"))
	_ = drain(t, c, r)
	c.Release(r)
	c.mu.Lock()
	r.Completed = epoch() - 600
	r.Access = epoch() - 600
	if e = persist(r); e != nil {
		t.Fatal(e)
	}
	c.mu.Unlock()
	c.Close()
	// A ten-minute-old entry survives restart with a one-hour configured TTL.
	c, e = New(cfg, gen)
	if e != nil {
		t.Fatal(e)
	}
	r, e = c.Acquire(key("a"), engine.Request{}, &engine.Info{}, false)
	if e != nil {
		t.Fatal(e)
	}
	if got := drain(t, c, r); len(got) != 4 {
		t.Fatal(got)
	}
	c.Release(r)
	c.Close()
	cfg.TTL = 300 * time.Second
	// A hit must persist its sliding expiration across another restart.
	c, e = New(cfg, gen)
	if e != nil {
		t.Fatal(e)
	}
	r, e = c.Acquire(key("a"), engine.Request{}, &engine.Info{}, false)
	if e != nil {
		t.Fatal(e)
	}
	c.Release(r)
	c.mu.Lock()
	r.Access = epoch() - 600
	r.Completed = r.Access
	if e = persist(r); e != nil {
		t.Fatal(e)
	}
	c.mu.Unlock()
	c.Close()
	c, e = New(cfg, gen)
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	if _, e = c.Acquire(key("a"), engine.Request{}, &engine.Info{}, false); e == nil {
		t.Fatal("expired entry restored")
	}
}

func TestFailedGenerationCanRetryAndCannotRestore(t *testing.T) {
	cfg := settings(t)
	var attempts atomic.Int32
	c, e := New(cfg, func(_ context.Context, _ engine.Request, _ *engine.Info, emit func([]byte) error) error {
		if e := emit([]byte{1, 2}); e != nil {
			return e
		}
		if attempts.Add(1) == 1 {
			return errors.New("upstream EOF")
		}
		return nil
	})
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	r := acquire(t, c, key("a"))
	buf := make([]byte, 64)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	for offset := int64(0); ; {
		n, e := c.Read(ctx, r, offset, buf)
		offset += int64(n)
		if e != nil {
			if e == io.EOF {
				t.Fatal("failed generation was successful")
			}
			break
		}
	}
	c.Release(r)
	retry := acquire(t, c, key("a"))
	if retry == r {
		t.Fatal("failed record reused")
	}
	_ = drain(t, c, retry)
	c.Release(retry)
	if _, e = os.Stat(metadataPath(r)); !os.IsNotExist(e) {
		t.Fatal("failed entry has metadata")
	}
}

func TestOrphansOutputLimitAndShutdown(t *testing.T) {
	cfg := settings(t)
	cfg.Concurrency = 1
	cfg.Pending = 0
	orphan := filepath.Join(cfg.Directory, "tts-orphan.pcm")
	if e := os.WriteFile(orphan, []byte{0, 0}, 0600); e != nil {
		t.Fatal(e)
	}
	c, e := New(cfg, func(_ context.Context, _ engine.Request, _ *engine.Info, emit func([]byte) error) error {
		return emit(make([]byte, 4098))
	})
	if e != nil {
		t.Fatal(e)
	}
	if _, e = os.Stat(orphan); !os.IsNotExist(e) {
		t.Fatal("orphan not cleaned")
	}
	r := acquire(t, c, key("a"))
	_, e = c.Read(context.Background(), r, 0, make([]byte, 2))
	if e == nil || e == io.EOF {
		t.Fatal("output overflow accepted")
	}
	c.Release(r)
	c.Close()
	c, e = New(cfg, func(ctx context.Context, _ engine.Request, _ *engine.Info, _ func([]byte) error) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if e != nil {
		t.Fatal(e)
	}
	r = acquire(t, c, key("b"))
	c.Cancel()
	c.Close()
	c.Release(r)
	if _, e = os.Stat(metadataPath(r)); !os.IsNotExist(e) {
		t.Fatal("canceled task persisted")
	}
}

func TestCapacityProtectsReadersAndEvictsIdle(t *testing.T) {
	cfg := settings(t)
	cfg.Output = 4
	cfg.Capacity = 4
	c, e := New(cfg, func(_ context.Context, _ engine.Request, _ *engine.Info, emit func([]byte) error) error {
		return emit([]byte{1, 2, 3, 4})
	})
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	a := acquire(t, c, key("a"))
	_ = drain(t, c, a)
	if _, e = c.Acquire(key("b"), engine.Request{}, &engine.Info{}, true); e == nil {
		t.Fatal("active reader evicted")
	}
	c.Release(a)
	b := acquire(t, c, key("b"))
	_ = drain(t, c, b)
	c.Release(b)
	if _, e = os.Stat(a.path); !os.IsNotExist(e) {
		t.Fatal("idle LRU was not removed")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.accounted != 4 {
		t.Fatal(c.accounted)
	}
}
