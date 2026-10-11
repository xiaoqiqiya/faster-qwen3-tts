# Linux 部署：直接拉取 GitHub 镜像

GitHub Actions 构建两端镜像并发布到 GitHub Container Registry（GHCR），服务器直接拉取即可，无需安装项目 Python 依赖或执行 `docker compose build`。

| 服务 | 镜像 |
|---|---|
| GPU 推理后端 | `ghcr.io/xiaoqiqiya/faster-qwen3-tts-backend:latest` |
| HTTP 网关 | `ghcr.io/xiaoqiqiya/faster-qwen3-tts-gateway:latest` |

镜像面向 `linux/amd64`，不包含模型权重、参考录音和音频缓存。内置音色和克隆音色使用同一套镜像，通过 `.env` 选择模型和音色。

## 1. GitHub 构建与发布

修改服务源码、Dockerfile、依赖或 Compose 并推送到 `main` 后，仓库的 **Actions → Docker images** 自动构建。也可在该工作流页面选择 `main`，点击 **Run workflow** 手动构建。

工作流使用 GitHub 自带的 `GITHUB_TOKEN` 发布 GHCR，无需额外配置 Docker Hub 密码。每次构建先发布两端的 `sha-完整提交SHA` 标签，两端都成功后才更新 `latest`。构建不会启动 GPU 模型或部署服务器。

首次发布后检查两个 package 的可见性：公开仓库不代表镜像自动公开。若需要免登录下载，在 GitHub 个人主页 **Packages** 中分别打开两个镜像的 **Package settings → Change visibility → Public**。私有镜像需要有权限的账号和具有 `read:packages` 权限的 PAT，在服务器执行：

```bash
docker login ghcr.io -u xiaoqiqiya
```

在密码提示中输入 PAT，不将 token 写入 `.env` 或 Compose。

## 2. 服务器需要哪些文件

需要 NVIDIA 驱动、NVIDIA Container Toolkit、Docker 和 Docker Compose 2.30 或更新版本。GPU 和驱动需兼容 CUDA 12.8 / PyTorch 2.8.0 cu128。

部署目录只需要：

```text
/data/qwen3-tts/service/
├── compose.images.yaml
├── .env                   # 从仓库 .env.example 复制并修改
└── references/            # 内置音色可为空；克隆模式放录音、转写和 voices.json
```

不需要上传源码、Dockerfile 或 `pyproject.toml`。如需用交付客户端保存 WAV，可另外下载 `examples/http_streaming_client.py` 到调用机器。

首次准备目录并下载部署文件：

```bash
mkdir -p /data/qwen3-tts/service/references /data/qwen3-tts/cache /data/qwen3-tts/models
cd /data/qwen3-tts/service
curl -fL https://raw.githubusercontent.com/xiaoqiqiya/faster-qwen3-tts/main/compose.images.yaml -o compose.images.yaml
curl -fL https://raw.githubusercontent.com/xiaoqiqiya/faster-qwen3-tts/main/.env.example -o .env
```

以上下载 `.env` 的命令只用于首次部署；升级时保留自己的配置。固定镜像版本时，部署文件也应从同一提交下载，将 URL 中的 `main` 改成该完整提交 SHA。

## 3. 内置音色配置

用已有的 hfd 脚本提前下载完整模型：

```bash
./hfd.sh Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

保留全部权重、配置、tokenizer 和 `speech_tokenizer/` 子目录。下载完成后再启动服务。

编辑部署目录中的 `.env`：

```dotenv
TTS_IMAGE_PREFIX=ghcr.io/xiaoqiqiya/faster-qwen3-tts
TTS_IMAGE_TAG=latest
TTS_MODEL_HOST_DIR=/data/qwen3-tts/models
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_DEFAULT_VOICE=vivian
TTS_VOICES_FILE=
TTS_BIND_ADDRESS=0.0.0.0
TTS_PORT=8000
TTS_CACHE_HOST_DIR=/data/qwen3-tts/cache
TTS_CACHE_MAX_MB=256
TTS_CACHE_TTL_SECONDS=300
TTS_MAX_PENDING_TASKS=8
```

`TTS_MODEL_HOST_DIR` 是宿主机目录，Compose 自动挂载到 `/models`；`TTS_MODEL` 是容器内模型目录。模型目录保持可写，框架缓存也使用 `/models`。无需手动修改 Compose 的挂载。

音频缓存落在宿主机 `/data/qwen3-tts/cache/audio/`，容器重启后恢复未过期的完整结果。容量单位为 MB，`1 MB = 1024 * 1024` 字节；TTL 按配置计算，停机时间计入过期。达到容量时可以先淘汰无人读取的 LRU 结果，详见 [缓存规则](README.md#缓存和资源限制)。

`TTS_MAX_PENDING_TASKS` 是一个执行任务之外允许排队的生成任务数，填写正整数，默认 8。调大只增加等待容量，GPU 仍同时生成一个任务；排队受缓存容量和任务总超时约束。升级后使用新 `compose.images.yaml`，修改 `.env` 并重新创建网关才会生效。

如只通过服务器本机或反向代理访问，使用 `TTS_BIND_ADDRESS=127.0.0.1`。当前接口没有账户认证，对外访问由部署环境控制。

## 4. 自定义克隆音色配置

下载 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 到 `/data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base`，在部署目录的 `references/` 放入录音、UTF-8 转写和配置：

```json
{
  "zhangsan": {"ref_audio": "zhangsan.wav", "ref_text_file": "zhangsan.txt"},
  "lisi": {"ref_audio": "lisi.wav", "ref_text_file": "lisi.txt"}
}
```

将上述内置音色配置中的三项改为：

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-Base
TTS_DEFAULT_VOICE=zhangsan
TTS_VOICES_FILE=/references/voices.json
```

请求通过 `voice=zhangsan` 或 `voice=lisi` 选择。录音要求、单音色简化方式和请求示例见 [自定义音色文档](voice-clone-linux.md)。使用该文档中的音色步骤，启动、日志和重新创建命令统一带上 `-f compose.images.yaml`。

## 5. 拉取和启动

在部署目录执行：

```bash
docker compose -f compose.images.yaml pull
docker compose -f compose.images.yaml up -d
docker compose -f compose.images.yaml ps
docker compose -f compose.images.yaml logs -f backend gateway
```

首次加载会计算本地模型哈希、准备参考特征并预热 CUDA Graph。后端就绪后网关才启动，后端健康检查预留 30 分钟。检查状态和音色：

```bash
curl -s http://127.0.0.1:8000/health
# 已安装 jq 时：
curl -s http://127.0.0.1:8000/health \
  | jq '.data.backend | {model, model_type, voices, default_voice}'
```

成功时 HTTP 状态和 JSON `code` 均为 200，音色位于 `data.backend.voices`。流式调用和客户端保存 WAV 见 [接口文档](README.md#请求和持续播放)。

## 6. 更新和固定版本

跟随 `latest` 更新：

```bash
docker compose -f compose.images.yaml pull
docker compose -f compose.images.yaml up -d
```

固定版本时，将 `.env` 的 `TTS_IMAGE_TAG` 设置为 Actions 构建摘要提供的 `sha-完整提交SHA`，两端会使用同一提交的镜像，再执行以上命令。回滚时改为之前成功发布的标签并重新拉取启动。正式部署建议固定提交标签，避免 `latest` 在两次拉取之间变化。

修改参考文件、音色配置或默认音色后：

```bash
docker compose -f compose.images.yaml up -d --force-recreate backend gateway
```

宿主机的模型、参考文件及缓存目录不会随容器重新创建而消失。不要将同一个缓存目录交给多个网关实例使用。
