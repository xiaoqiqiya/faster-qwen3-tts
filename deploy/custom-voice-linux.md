# Linux 部署：使用内置音色，不克隆

本方式使用 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`，直接通过请求的 `voice` 选择模型内置音色，例如 `vivian`。不需要参考录音、转写文本或 `voices.json`。

部署仍使用两个容器：`backend` 加载一份模型并执行 GPU 推理，`gateway` 提供缓存和流式 HTTP 接口。客户端访问网关的 `8000` 端口，后端 `8001` 只在 Docker 内部网络使用。切换内置音色不需要为每个音色加载一份模型。

以下命令在目标 Linux 服务器执行。先将当前项目源码放到服务器，所有 `docker compose` 命令都在项目根目录执行。

## 1. 服务器准备

需要安装：

- NVIDIA 驱动，以及与 CUDA 12.8 容器兼容的 GPU 和驱动版本。
- Docker、NVIDIA Container Toolkit。
- Docker Compose 2.30 或更新版本，支持当前配置的 `gpus: all`。
- hfd，用于提前下载模型。

可以先检查：

```bash
nvidia-smi
docker compose version
```

镜像固定使用 CUDA 12.8 和 PyTorch 2.8.0/cu128。服务器无需另外安装项目的 Python 推理依赖，依赖随镜像构建安装。

## 2. 用 hfd 下载 CustomVoice 模型

示例将模型放在宿主机的 `/data/qwen3-tts/models`：

```bash
mkdir -p /data/qwen3-tts/models
hfd Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

如果你的工具是脚本，使用 `./hfd.sh` 代替 `hfd`；参数以该版本的 `--help` 为准。下载完整仓库，保留 `config.json`、全部权重、tokenizer 配置以及 `speech_tokenizer/` 等子目录。下载完成后再启动服务。

目录对应关系：

```text
宿主机：/data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
容器内：/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

不要下载 `Base` 模型用于本方式；`Base` 在当前服务中走参考音频克隆流程。

## 3. 配置模型挂载和默认音色

在项目根目录复制配置；已有 `.env` 时直接编辑已有文件：

```bash
cp .env.example .env
```

修改 `compose.yaml` 中 **`backend` 的 `volumes`**，将原来的 `model-cache:/models` 替换成宿主机模型目录：

```yaml
    volumes:
      - /data/qwen3-tts/models:/models
      - ./references:/references:ro
```

这里只替换模型挂载，其他 Compose 配置沿用仓库文件。`/models` 也保存框架缓存，因此保持可写；`references` 挂载可保留，内置音色模式不读取参考录音和转写。

修改 `.env` 中以下项目，其他参数先保留默认值：

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_DEFAULT_VOICE=vivian
TTS_VOICES_FILE=
TTS_BIND_ADDRESS=0.0.0.0
TTS_PORT=8000
```

`TTS_MODEL` 必须使用容器内路径。`TTS_VOICES_FILE` 必须留空，从克隆部署切换过来时也要清空它，否则 CustomVoice 会拒绝启动。

`TTS_BIND_ADDRESS=0.0.0.0` 允许远程客户端访问服务器的 `8000` 端口；仅通过本机或本机反向代理访问时使用 `127.0.0.1`。当前服务没有账户认证，对外开放时由部署环境控制访问。

本地目录模式直接加载下载好的文件，不调用 `snapshot_download`，`TTS_MODEL_REVISION` 不参与加载。启动时会读取模型文件计算内容哈希，健康接口返回的版本形如 `local-sha256:...`；运行期间不要修改模型目录。

如果需要明确禁止模型加载阶段联网，可在 `backend.environment` 中添加 `HF_HUB_OFFLINE: "1"`。Docker 镜像构建仍需要基础镜像和 Python 依赖。

## 4. 构建和启动

在服务器上的项目根目录执行：

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend gateway
```

第一次启动会加载模型、计算本地文件哈希并预热 CUDA Graph，需要等待。后端就绪后网关才会启动；模型不打进镜像，通过上述宿主机目录挂载读取。

退出日志查看后检查接口：

```bash
curl -s http://127.0.0.1:8000/health
```

就绪时返回 HTTP 200，统一 JSON 响应中的 `code` 为 `200`，模型信息在 `data.backend`。依赖未就绪时健康接口返回 503。

## 5. 查看可用内置音色

已经安装 `jq` 时，可以提取模型和音色信息：

```bash
curl -s http://127.0.0.1:8000/health \
  | jq '.data.backend | {model, model_type, voices, default_voice}'
```

应看到 `model_type` 为 `custom_voice`，`default_voice` 为 `vivian`。`voices` 是实际加载模型支持的音色名称列表，调用时从该列表选择并按原样传入。

`vivian` 是一个内置音色名称。请求省略 `voice` 时使用 `.env` 配置的默认音色；同一个服务可以通过不同的 `voice` 合成不同内置音色，无需重启。更改默认音色或模型路径等启动配置后执行：

```bash
docker compose up -d --force-recreate backend gateway
```

## 6. 调用合成接口并保存 WAV

接口为 `POST /v1/audio/speech`，请求示例：

```json
{
  "input": "浙海一二三号船，请减速慢行。",
  "voice": "vivian",
  "language": "Chinese",
  "response_format": "pcm_framed"
}
```

`0.6B-CustomVoice` 不支持非空 `instruct`，请省略或传空字符串；当前服务的 `speed` 只能省略或设为 `1.0`。

成功响应是带完成帧的二进制 PCM 流。使用仓库客户端接收并保存为 WAV，在拥有项目客户端文件和 Python 3.10 或更新版本的机器上执行：

```bash
python3 examples/http_streaming_client.py \
  --url http://127.0.0.1:8000 \
  --text '浙海一二三号船，请减速慢行。' \
  --voice vivian \
  --output announcement.wav
```

远程调用时将 URL 中的 `127.0.0.1` 替换为服务器 IP。保存 WAV 只需要 Python 标准库，客户端不需要安装模型依赖。输出文件已存在时请换一个文件名。

不要把 `pcm_framed` 的 HTTP 响应直接用 curl 保存为 `.wav`：它包含协议帧头，需要先解帧，再写入 WAV 文件。接入其他客户端时参考 [通用部署说明中的二进制帧协议](README.md#二进制帧协议)。

如果之后需要通过 `instruct` 控制语气，可改用 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`，下载对应完整模型并同步修改挂载路径及 `TTS_MODEL`；内置音色仍通过健康接口查询，显存和效果需要在服务器确认。

本文按当前配置和接口编写，未在本机构建镜像、启动 GPU 服务或执行行为测试。镜像构建、容器 GPU 可用性、实际音质和首包延迟需在目标 Linux 服务器验证。
