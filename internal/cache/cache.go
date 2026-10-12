package cache

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"tts/internal/config"
	"tts/internal/engine"
)

type diskMeta struct {
	Version   int     `json:"version"`
	Key       string  `json:"key"`
	Size      int64   `json:"size"`
	Completed float64 `json:"completed_at"`
	Access    float64 `json:"last_access_at"`
}

type Record struct {
	diskMeta
	path      string
	done      bool
	err       error
	readers   int
	accounted int64
	changed   chan struct{}
	ctx       context.Context
	cancel    context.CancelFunc
	req       engine.Request
	info      *engine.Info
}

type generator func(context.Context, engine.Request, *engine.Info, func([]byte) error) error
type Store struct {
	mu                      sync.Mutex
	c                       config.Config
	records                 map[string]*Record
	accounted               int64
	readers, tasks, running int
	closed                  bool
	queue                   chan *Record
	ctx                     context.Context
	cancel                  context.CancelFunc
	wg                      sync.WaitGroup
	generate                generator
}

func epoch() float64 { return float64(time.Now().UnixNano()) / 1e9 }
func expired(r *Record, ttl time.Duration) bool {
	return epoch()-max(r.Access, r.Completed) >= ttl.Seconds()
}
func notify(r *Record) { close(r.changed); r.changed = make(chan struct{}) }

func New(c config.Config, generate generator) (*Store, error) {
	if err := os.MkdirAll(c.Directory, 0700); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	x := &Store{c: c, records: make(map[string]*Record), queue: make(chan *Record, c.Concurrency+c.Pending), ctx: ctx, cancel: cancel, generate: generate}
	if err := x.restore(); err != nil {
		cancel()
		return nil, err
	}
	for i := 0; i < c.Concurrency; i++ {
		x.wg.Add(1)
		go func() {
			defer x.wg.Done()
			for r := range x.queue {
				x.produce(r)
			}
		}()
	}
	x.wg.Add(1)
	go func() {
		defer x.wg.Done()
		ticker := time.NewTicker(c.Sweep)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				x.mu.Lock()
				x.sweepLocked()
				x.mu.Unlock()
			}
		}
	}()
	return x, nil
}

func (c *Store) restore() error {
	files, err := filepath.Glob(filepath.Join(c.c.Directory, "tts-*"))
	if err != nil {
		return err
	}
	for _, path := range files {
		if filepath.Ext(path) != ".json" {
			continue
		}
		var m diskMeta
		var data []byte
		f, e := os.Open(path)
		if e == nil {
			data, e = io.ReadAll(io.LimitReader(f, 4097))
			f.Close()
		}
		pcm := strings.TrimSuffix(path, ".json") + ".pcm"
		st, se := os.Lstat(pcm)
		valid := e == nil && len(data) <= 4096 && json.Unmarshal(data, &m) == nil
		_, ke := hex.DecodeString(m.Key)
		valid = valid && m.Version == 1 && len(m.Key) == 64 && ke == nil && m.Size > 0 && m.Size%2 == 0 && m.Size <= c.c.Output && se == nil && st.Mode().IsRegular() && st.Size() == m.Size && m.Completed > 0 && m.Access > 0 && max(m.Access, m.Completed) <= epoch()+1
		r := &Record{diskMeta: m, path: pcm, done: true, accounted: m.Size, changed: make(chan struct{})}
		if !valid || expired(r, c.c.TTL) {
			_ = os.Remove(path)
			continue
		}
		if old := c.records[m.Key]; old != nil {
			if old.Access >= r.Access {
				_ = os.Remove(path)
				continue
			}
			c.removeLocked(old)
		}
		c.records[m.Key] = r
		c.accounted += r.accounted
	}
	keep := map[string]bool{}
	for _, r := range c.records {
		keep[r.path] = true
		keep[metadataPath(r)] = true
	}
	for _, path := range files {
		if !keep[path] {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return err
			}
		}
	}
	for c.accounted > c.c.Capacity {
		if !c.evictLocked() {
			return errors.New("缓存无法收缩到配置容量")
		}
	}
	return nil
}

func metadataPath(r *Record) string { return strings.TrimSuffix(r.path, ".pcm") + ".json" }
func persist(r *Record) error {
	data, err := json.Marshal(r.diskMeta)
	if err != nil {
		return err
	}
	path := metadataPath(r)
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	ce := f.Close()
	if err == nil {
		err = ce
	}
	if err == nil {
		err = os.Rename(tmp, path)
	}
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	d, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

func (c *Store) removeLocked(r *Record) {
	if c.records[r.Key] == r {
		delete(c.records, r.Key)
	}
	// Remove the commit marker first; an orphan PCM is safe to clean on restart.
	for _, path := range []string{metadataPath(r), r.path} {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			log.Printf("缓存文件清理失败: %v", err)
		}
	}
	c.accounted -= r.accounted
	r.accounted = 0
}
func (c *Store) sweepLocked() {
	for _, r := range c.records {
		if r.done && r.readers == 0 && expired(r, c.c.TTL) {
			c.removeLocked(r)
		}
	}
}
func (c *Store) evictLocked() bool {
	var oldest *Record
	for _, r := range c.records {
		if r.done && r.readers == 0 && (oldest == nil || r.Access < oldest.Access) {
			oldest = r
		}
	}
	if oldest == nil {
		return false
	}
	c.removeLocked(oldest)
	return true
}

func (c *Store) Acquire(key string, req engine.Request, info *engine.Info, allowNew bool) (*Record, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.ctx.Err() != nil {
		return nil, errors.New("网关正在关闭")
	}
	if c.readers >= c.c.Readers {
		return nil, errors.New("音频连接数已达上限")
	}
	c.sweepLocked()
	if r := c.records[key]; r != nil {
		r.Access = epoch()
		if r.done {
			if err := persist(r); err != nil {
				return nil, errors.New("缓存访问时间写入失败")
			}
		}
		r.readers++
		c.readers++
		return r, nil
	}
	if !allowNew {
		return nil, errors.New("推理服务暂不可用且没有匹配缓存")
	}
	if c.tasks >= c.c.Concurrency+c.c.Pending {
		return nil, errors.New("推理任务和等待队列已满")
	}
	for c.accounted+c.c.Output > c.c.Capacity {
		if !c.evictLocked() {
			return nil, errors.New("磁盘缓存容量已达上限")
		}
	}
	reserved := int64(0)
	for _, r := range c.records {
		if !r.done {
			reserved += r.accounted - r.Size
		}
	}
	if err := diskAvailable(c.c.Directory, reserved+c.c.Output); err != nil {
		return nil, errors.New("缓存磁盘可用空间不足")
	}
	f, err := os.CreateTemp(c.c.Directory, "tts-*.pcm")
	if err != nil {
		return nil, errors.New("无法创建磁盘缓存")
	}
	path := f.Name()
	if err = f.Close(); err != nil {
		_ = os.Remove(path)
		return nil, errors.New("磁盘缓存初始化失败")
	}
	ctx, cancel := context.WithTimeout(c.ctx, c.c.Task)
	r := &Record{diskMeta: diskMeta{Version: 1, Key: key, Access: epoch()}, path: path, readers: 1, accounted: c.c.Output, changed: make(chan struct{}), ctx: ctx, cancel: cancel, req: req, info: info}
	c.records[key] = r
	c.accounted += r.accounted
	c.readers++
	c.tasks++
	c.queue <- r
	return r, nil
}

func (c *Store) produce(r *Record) {
	c.mu.Lock()
	c.running++
	c.mu.Unlock()
	defer r.cancel()
	f, err := os.OpenFile(r.path, os.O_WRONLY|os.O_APPEND, 0600)
	if err == nil {
		if err = r.ctx.Err(); err == nil {
			err = c.generate(r.ctx, r.req, r.info, func(chunk []byte) error {
				c.mu.Lock()
				defer c.mu.Unlock()
				if err := r.ctx.Err(); err != nil {
					return errors.New("生成任务超时或已取消")
				}
				if len(chunk) == 0 || len(chunk)%2 != 0 || int64(len(chunk))+r.Size > c.c.Output {
					return errors.New("音频无效或超过单任务输出容量")
				}
				n, e := f.Write(chunk)
				if e != nil {
					return errors.New("音频缓存写入失败")
				}
				if n != len(chunk) {
					return io.ErrShortWrite
				}
				r.Size += int64(n)
				notify(r)
				return nil
			})
		}
		if err == nil {
			err = f.Sync()
		}
		ce := f.Close()
		if err == nil {
			err = ce
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if r.ctx.Err() != nil {
		err = errors.New("生成任务超时或已取消")
	}
	if err == nil && r.Size == 0 {
		err = errors.New("推理服务未生成音频")
	}
	if err == nil {
		r.Completed = epoch()
		r.Access = max(r.Access, r.Completed)
		err = persist(r)
	}
	if r.ctx.Err() != nil {
		err = errors.New("生成任务超时或已取消")
	}
	r.done = true
	c.tasks--
	c.running--
	if err != nil {
		log.Printf("生成任务失败 key=%s: %v", r.Key, err)
		r.err = errors.New("语音生成失败，请重试；已接收音频应丢弃")
		if c.records[r.Key] == r {
			delete(c.records, r.Key)
		}
		_ = os.Remove(metadataPath(r))
		if r.readers == 0 {
			c.removeLocked(r)
		}
	} else {
		c.accounted -= r.accounted - r.Size
		r.accounted = r.Size
	}
	notify(r)
}

func (c *Store) Release(r *Record) {
	c.mu.Lock()
	defer c.mu.Unlock()
	r.readers--
	c.readers--
	if r.done && r.err != nil && r.readers == 0 {
		c.removeLocked(r)
	}
}

func (c *Store) Read(ctx context.Context, r *Record, offset int64, buffer []byte) (int, error) {
	for {
		c.mu.Lock()
		size, done, err, ch := r.Size, r.done, r.err, r.changed
		c.mu.Unlock()
		if err != nil {
			return 0, err
		}
		if offset < size {
			f, e := os.Open(r.path)
			if e != nil {
				return 0, errors.New("无法读取磁盘音频缓存")
			}
			n, e := f.ReadAt(buffer[:min(int64(len(buffer)), size-offset)], offset)
			f.Close()
			if e != nil {
				return 0, errors.New("磁盘音频缓存不完整")
			}
			return n, nil
		}
		if done {
			return 0, io.EOF
		}
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-r.ctx.Done():
			c.mu.Lock()
			finished := r.done
			c.mu.Unlock()
			if !finished {
				return 0, errors.New("生成任务超时或已取消")
			}
		case <-ch:
		}
	}
}

func (c *Store) Stats() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	return map[string]any{"cache_entries": len(c.records), "active_readers": c.readers, "active_tasks": c.tasks, "running_tasks": c.running, "pending_tasks": c.tasks - c.running, "cache_accounted_bytes": c.accounted, "cache_storage": "disk", "cache_directory": c.c.Directory, "max_concurrent_tasks": c.c.Concurrency, "max_pending_tasks": c.c.Pending}
}
func (c *Store) Close() {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return
	}
	c.closed = true
	c.cancel()
	close(c.queue)
	c.mu.Unlock()
	c.wg.Wait()
}

func (c *Store) Cancel() { c.cancel() }
