# Linux 部署

## 需要上传的文件

直接使用镜像时，只需要：

```text
/data/qwen3-tts/service/
  compose.images.yaml
  .env
  references/             # CustomVoice 可为空，Base 放录音、转写和 voices.json
  cache/                  # 自动写入的磁盘音频缓存
```

模型提前放在 `/data/qwen3-tts/models`。Go 程序、Python 启动适配和官方推理引擎都已在镜像中，不需要上传源码、Go 或 Python 依赖。backend-state 为 Docker 持久卷，保存元数据和预计算音色。

## 主机准备与模型下载

使用 Linux x86_64、NVIDIA GPU、兼容官方镜像 CUDA 的主机驱动、NVIDIA Container Toolkit、Docker Engine 和 Compose v2.30+。官方 vLLM-Omni 镜像较大，拉取和解压需要为 Docker 数据目录准备足够空间；模型、缓存另计。

不要继续按旧 faster/cu128 镜像的驱动要求部署。可在远端查看固定镜像的 CUDA / PyTorch 信息：

```bash
docker run --rm --gpus all --entrypoint python \
  vllm/vllm-omni:v0.28.0@sha256:6f8be103eaf0055448cf7578cfd621405fd669079d4361bd58896326b2bf722a \
  -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

使用已有 `hfd.sh` 下载内置音色模型：

```bash
mkdir -p /data/qwen3-tts/models
bash hfd.sh Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

需要镜像站时先设置 `HF_ENDPOINT` 为你可访问的地址，再运行 hfd。完整下载仓库，保留 `speech_tokenizer/`、所有权重、JSON 配置和 tokenizer 文件。不能只下载主模型的 safetensors。

配置：

```dotenv
TTS_MODEL_HOST_DIR=/data/qwen3-tts/models
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_DEFAULT_VOICE=vivian
TTS_IMAGE_PREFIX=ghcr.io/owner/repo
TTS_IMAGE_TAG=vllm-omni
```

本地模型加载按实际文件内容生成身份；`TTS_MODEL_REVISION` 只用于填写 Hugging Face 仓库 ID 时的下载。模型每次启动会被流式计算内容哈希，大模型会增加启动时间。请在停服务时更新模型，避免运行中覆盖权重。

`TTS_IMAGE_PREFIX` 必须替换为实际镜像前缀，不含 `-backend`、`-gateway` 或标签。例如仓库为 `myteam/voice-service`，填写 `ghcr.io/myteam/voice-service`。CI 从当前 GitHub 仓库自动生成这个前缀并转为小写；仓库改名后，等待新仓库镜像发布完成，再修改部署配置即可。镜像部署未填写此前缀时会直接报错；源码构建无需填写。

## 启动、更新和日志

```bash
cp .env.example .env                  # 仅首次，从源码目录准备配置时执行
# 编辑 .env，按上文填写 TTS_IMAGE_PREFIX 和模型配置。
mkdir -p references cache
docker compose -f compose.images.yaml pull
docker compose -f compose.images.yaml up -d
docker compose -f compose.images.yaml logs -f backend
curl http://127.0.0.1:8000/health
```

后端预处理、加载两阶段模型并完成预热后才启动 gateway，首次可能较慢。默认仅绑定主机 127.0.0.1；需要远程直连时设置 `TTS_BIND_ADDRESS=0.0.0.0` 并用部署环境的鉴权/访问控制保护服务。

修改配置、模型或参考文件后：

```bash
docker compose -f compose.images.yaml up -d --force-recreate
```

镜像版本默认跟随迁移分支的 `vllm-omni`；生产环境建议两端同时固定到 `sha-完整提交SHA`。源代码构建使用 `docker compose -f compose.yaml up -d --build`，在远端或 CI 中执行即可。

普通容器重建/重启会保留宿主机缓存和命名卷；`docker compose down -v` 会删除命名卷，因此需要保留音色预计算结果时不要使用 `-v`。

从旧版本升级时，下载本分支的新 Compose 并以新的 `.env.example` 重新核对参数。旧缓存因模型/引擎身份改变不会命中，之后按 TTL/容量清理。旧版本的 `TTS_CHUNK_SIZE`、`TTS_MAX_SEQ_LEN`、单参考 `TTS_REF_AUDIO` / `TTS_REF_TEXT_FILE` 已移除；Base 改用命名音色配置。

## 运行配置

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TTS_MAX_CONCURRENT_TASKS` | 4 | 两端统一的不同任务并发上限，范围 1–256 |
| `TTS_MAX_PENDING_TASKS` | 8 | 额外等待任务数，0 表示不排队，最大 10000 |
| `TTS_GPU_MEMORY_UTILIZATION` | 0.3 | 每个阶段的显存比例，两个阶段合计约 0.6；必须大于 0 且小于 0.5 |
| `TTS_MAX_NEW_TOKENS` | 1024 | 音频 codec token 生成上限，2–4096 |
| `TTS_CACHE_HOST_DIR` | ./cache | 宿主机音频缓存路径 |
| `TTS_CACHE_MAX_MB` | 256 | PCM 磁盘缓存与活跃任务预留预算，MB 为 1024² 字节 |
| `TTS_CACHE_TTL_SECONDS` | 300 | 最后访问/完成后的滑动过期时间 |
| `TTS_CACHE_CLEANUP_SECONDS` | 5 | 清理扫描间隔，不是第二个 TTL |
| `TTS_MAX_READERS` | 256 | 同时读取音频的客户端数量 |
| `TTS_TASK_TIMEOUT_SECONDS` | 120 | 入队到完成的总时限 |
| `TTS_SEND_TIMEOUT_SECONDS` | 15 | 每次向客户端写入/刷新数据的时限 |

不同请求最多受并发数、等待数和磁盘预算共同限制。例如 4 并发 + 8 等待时，最多 12 个不同任务；相同请求共享任务，但各自占用一个 reader。每个未完成任务按最多 8 MB 预留，因此把并发/等待配置调大时也需考虑缓存预算。

并发不是越大越快：先用 2–4 路不同文本验收，再根据首包延迟、实时率和显存余量增加。官方默认部署配置以 H100 为验证场景，本项目没有承诺任意消费级 GPU 都能运行默认值。显存不足时先降低并发，并结合引擎日志调整每阶段显存比例；过低比例也可能无法容纳模型。

## 验收

```bash
curl http://127.0.0.1:8000/v1/audio/voices
python examples/http_streaming_client.py --text '第一条播报。' --voice vivian --output first.wav &
python examples/http_streaming_client.py --text '这是另一条不同的播报。' --voice vivian --output second.wav &
wait
```

在健康接口查看 `running_tasks`、`pending_tasks`。验证不同文本并行、相同请求共享、首块到达早于完成，以及生成后重启 gateway 仍可命中磁盘缓存。停机超过配置 TTL 的音频应失效。CI 通过只说明 CPU 逻辑和镜像构建通过，GPU 启动、音频质量和并行吞吐需在此环境确认。
