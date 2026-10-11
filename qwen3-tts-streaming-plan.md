# Qwen3-TTS 实现方案

本分支采用 Go 网关 + vLLM-Omni v0.28.0 官方推理引擎，替代旧 faster 实现。

- 网关、推理端两个容器；使用原生 HTTP/SSE 对接。
- 不同请求通过多个 Go worker 并行提交，实际连续批处理由 vLLM-Omni 两阶段调度器负责。
- 相同请求共享生成，音频落盘并增量分发，晚加入客户端从头补读。
- 磁盘缓存按配置的 TTL 和总容量管理，完整生成后原子提交元数据，重启恢复。
- 使用 `voice` 区分内置或多个预计算参考音色。
- 统一中文 JSON 响应，保留 PCM、带完成帧的 PCM 和 WAV 分块 SSE。
- 本机只进行语法检查，云端 CI 执行测试与构建，远端 GPU 验收。

具体设计见 [架构文档](docs/architecture.md)，使用方式见 [README](README.md)。
