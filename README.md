# Qwen3-TTS 流式服务

Go 网关 + vLLM-Omni 推理后端，面向 Linux / NVIDIA GPU 部署。支持不同文本并行推理、命名音色、真正增量音频、相同请求共享生成，以及重启后可恢复的磁盘缓存。

推理镜像固定为 vLLM-Omni v0.28.0 官方发布版及其摘要。旧 faster 推理框架、Python 网关、GGML、Gradio 和 Python 包发布流程已移除。

## 工程目录

```text
cmd/gateway/            Go 进程入口、启动与退出
internal/config/       配置解析与校验
internal/engine/       vLLM-Omni HTTP/SSE 客户端、模型身份、请求定义
internal/cache/        磁盘缓存、任务队列、共享生成、补读和过期清理
internal/httpapi/      对外 HTTP 接口、参数校验、PCM/SSE 协议
backend/               官方推理服务启动适配、命名音色预计算
deploy/                两端 Dockerfile、Nginx 流式代理示例
docs/                  架构、接口、部署和音色使用文档
examples/              独立客户端示例
references/            参考录音及配置模板
.github/workflows/     云端测试、镜像构建与 GHCR 发布
compose.yaml           从源码构建并部署
compose.images.yaml    拉取已发布镜像部署
.env.example           部署配置模板
```

Go 测试与实现放在同一包中；Go 部分只使用标准库。backend 中的 Python 只负责加载配置、预计算音色并启动官方引擎，没有另写一套推理服务。

## 快速部署

准备 NVIDIA 驱动、NVIDIA Container Toolkit、Docker Engine 和支持 `gpus: all` 的 Docker Compose v2.30+。GPU/驱动需要兼容固定的官方 vLLM-Omni 镜像；请按 [部署文档](docs/deployment.md) 做启动检查。

```bash
git clone -b feat/vllm-omni-backend https://github.com/xiaoqiqiya/faster-qwen3-tts.git
cd faster-qwen3-tts
cp .env.example .env
mkdir -p cache references /data/qwen3-tts/models
# 把完整模型下载到 /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
docker compose -f compose.images.yaml pull
docker compose -f compose.images.yaml up -d
docker compose -f compose.images.yaml logs -f backend
```

`.env` 默认使用本地 0.6B CustomVoice 模型，音色 `vivian`，最多 4 个不同任务并行推理、8 个等待任务。相同文本、音色、语言和指令的请求共享一次推理，输出协议不同也能共享。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/audio/voices
python examples/http_streaming_client.py --url http://127.0.0.1:8000 \
  --text '前方船舶请注意安全。' --voice vivian --output speech.wav
```

## 文档

- [Linux 部署、hfd 下载、升级与配置](docs/deployment.md)
- [内置音色与多个自定义音色](docs/voices.md)
- [HTTP、PCM 帧和 SSE 接口](docs/api.md)
- [模块职责、并发与缓存语义](docs/architecture.md)

## 验证与发布

本机仅做 Go 格式/语法解析和 Python AST 语法检查。GitHub Actions 在 Linux 上执行 Go race tests、vet、后端准备逻辑测试、Compose 一致性检查，再构建并发布两个镜像。

迁移分支发布 `vllm-omni` 标签；`main` 发布 `latest`。两个镜像都通过构建后才移动标签。部署固定版本时把 `TTS_IMAGE_TAG` 改为 `sha-完整提交SHA`。

CPU CI 不验证 GPU 显存、模型启动、音色效果或实时性。这些需要在远端实际 GPU 上验收。
