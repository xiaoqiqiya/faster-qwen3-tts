package main

import (
	"tts/internal/cache"
	"tts/internal/config"
	"tts/internal/engine"
	"tts/internal/httpapi"

	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	if len(os.Args) == 2 && os.Args[1] == "healthcheck" {
		client := http.Client{Timeout: 8 * time.Second}
		resp, err := client.Get("http://127.0.0.1:8000/health")
		if err != nil {
			os.Exit(1)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			os.Exit(1)
		}
		return
	}
	c, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	b := engine.New(c)
	store, err := cache.New(c, b.Generate)
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()
	srv := &http.Server{Addr: c.Addr, Handler: httpapi.New(c, b, store), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16384}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		<-ctx.Done()
		store.Cancel()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdown); err != nil {
			_ = srv.Close()
		}
	}()
	log.Printf("Go 网关监听 %s，推理并发 %d，等待上限 %d", c.Addr, c.Concurrency, c.Pending)
	err = srv.ListenAndServe()
	stop()
	<-finished
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("HTTP 服务退出: %v", err)
	}
}
