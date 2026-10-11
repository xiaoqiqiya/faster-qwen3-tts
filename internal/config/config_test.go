package config

import "testing"

func TestQueueZeroUnitsAndInvalidConfig(t *testing.T) {
	t.Setenv("TTS_MAX_PENDING_TASKS", "0")
	t.Setenv("TTS_CACHE_MAX_MB", "3")
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if c.Pending != 0 || c.Capacity != 3*1024*1024 || c.Output != c.Capacity {
		t.Fatal(c)
	}
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "0")
	if _, err = Load(); err == nil {
		t.Fatal("accepted zero concurrency")
	}
	t.Setenv("TTS_MAX_CONCURRENT_TASKS", "4")
	t.Setenv("TTS_BACKEND_URL", ":invalid")
	if _, err = Load(); err == nil {
		t.Fatal("accepted invalid backend address")
	}
}
