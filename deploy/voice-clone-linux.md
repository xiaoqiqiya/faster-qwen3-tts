# Linux 部署：自定义克隆音色

本方式使用 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，通过参考录音和对应转写生成克隆特征，不需要训练或微调模型。给每个自定义音色取一个不同名称，调用时通过 `voice` 选择，例如 `zhangsan`、`lisi`。

部署使用两个容器：`backend` 加载模型和参考音色，执行 GPU 推理；`gateway` 提供缓存和流式 HTTP 接口。客户端访问网关的 `8000` 端口，后端 `8001` 仅供 Docker 内部网络使用。

以下操作都在目标 Linux 服务器进行。需要 NVIDIA 驱动、NVIDIA Container Toolkit、Docker 和 Docker Compose 2.30 或更新版本。当前镜像使用 CUDA 12.8 和 PyTorch 2.8.0/cu128，GPU 和驱动需与之兼容。所有 `docker compose` 命令在服务器的项目根目录执行。

## 1. 用 hfd 下载 Base 模型

```bash
mkdir -p /data/qwen3-tts/models
hfd Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base
```

如果使用脚本，将 `hfd` 换成 `./hfd.sh`；参数以你安装版本的 `--help` 为准。下载完整仓库，保留 `config.json`、全部权重、tokenizer 配置和 `speech_tokenizer/` 等子目录，下载完成后再启动服务。

自定义参考音色使用 `Base` 模型。`CustomVoice` 使用模型内置音色，不能加载这里的多音色参考配置。

## 2. 准备参考录音和转写

将文件放在服务器的 **项目根目录下的 `references/`**：

```text
项目根目录/
├── compose.yaml
├── .env
└── references/
    ├── voices.json
    ├── zhangsan.wav
    ├── zhangsan.txt
    ├── lisi.wav
    └── lisi.txt
```

每个音色准备自己的录音和转写：

- 录音时长必须大于 0 且不超过 30 秒。建议使用清晰 WAV 录音，单人说话，减少背景噪声和音乐。
- `.txt` 使用 UTF-8 编码，内容是对应录音中实际说出的文字。转写不能为空，最多 2048 字符。
- 转写不是需要合成的新文本；新文本通过接口的 `input` 提交。

例如，`zhangsan.wav` 中实际说的是“各位船员请注意，前方进入限速航段。”，那么 `zhangsan.txt` 就填写这句话。

服务会在启动时给参考音频追加 0.5 秒静音并提取特征，无需手工追加。参考文件通过只读挂载提供给容器，不打进镜像。

## 3. 配置多个自定义音色

在项目根目录执行；已有 `voices.json` 时直接编辑，不要覆盖已有配置：

```bash
cp references/voices.example.json references/voices.json
```

编辑 `references/voices.json`：

```json
{
  "zhangsan": {
    "ref_audio": "zhangsan.wav",
    "ref_text_file": "zhangsan.txt"
  },
  "lisi": {
    "ref_audio": "lisi.wav",
    "ref_text_file": "lisi.txt"
  }
}
```

`zhangsan`、`lisi` 是你自己定义的音色名称，接口的 `voice` 传同一个名称。只配置一个音色时保留一个条目即可。名称不能重复或全为空白，最多 128 字符；建议使用简单的英文名称。配置文件必须包含 1 到 32 个音色，每个条目必须配置且只配置 `ref_audio` 和 `ref_text_file`。

相对路径以 `voices.json` 所在目录为准。因此上面的 `zhangsan.wav` 在容器中解析为 `/references/zhangsan.wav`。也可以使用子目录，如 `zhangsan/reference.wav`；使用绝对路径时必须是容器能访问的路径。

当前服务通过配置文件在启动时登记音色，没有上传录音或动态创建音色的 HTTP 接口。

## 4. 配置挂载和环境变量

修改 `compose.yaml` 中 **`backend` 的 `volumes`**，将原来的 `model-cache:/models` 替换成宿主机模型目录：

```yaml
    volumes:
      - /data/qwen3-tts/models:/models
      - ./references:/references:ro
```

模型目录挂载保持可写，框架缓存也写入 `/models`。`./references` 指的是项目根目录下的参考文件目录。

首次部署先复制环境配置；已有 `.env` 时直接编辑：

```bash
cp .env.example .env
```

修改这些项目，其余参数先保持默认：

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-Base
TTS_VOICES_FILE=/references/voices.json
TTS_DEFAULT_VOICE=zhangsan
TTS_BIND_ADDRESS=0.0.0.0
TTS_PORT=8000
```

`TTS_MODEL` 和 `TTS_VOICES_FILE` 都使用容器内路径。默认音色必须存在于配置文件；留空时使用配置文件中的第一个音色。从内置音色部署切换时，将原来的 `vivian` 改为你的音色名称。

`0.0.0.0` 允许远程客户端访问；仅供本机或本机反向代理访问时改成 `127.0.0.1`。当前接口没有账户认证，对外开放时由部署环境控制访问。

本地模型目录直接加载，不调用 `snapshot_download`，`TTS_MODEL_REVISION` 不参与加载。启动会计算模型文件内容哈希，健康接口中的版本形如 `local-sha256:...`。运行期间不要修改模型目录。

## 5. 构建和启动

在服务器项目根目录执行：

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend gateway
```

后端启动时加载一份模型，依次为所有已配置音色提取参考特征，再预热 CUDA Graph。音色越多，首次启动的耗时越长；网关会等后端就绪后启动。

每个音色的参考特征保存在 CPU 内存，合成时只将选中的音色特征放到 GPU。多个音色不会各加载一份模型，GPU 推理保持单并发；音色增多主要增加 CPU 内存和启动时间，参考提取和推理峰值显存仍需在服务器确认。

## 6. 查看音色和调用接口

检查健康状态：

```bash
curl -s http://127.0.0.1:8000/health
```

就绪时返回 HTTP 200，响应中的 `code` 为 `200`。`data.backend.model_type` 为 `base`，`data.backend.voices` 返回配置名称，`default_voice` 返回默认名称，`reference_versions` 返回每个参考音色的内容哈希。

已经安装 `jq` 时可以提取这些信息：

```bash
curl -s http://127.0.0.1:8000/health \
  | jq '.data.backend | {model_type, voices, default_voice, reference_versions}'
```

合成接口为 `POST /v1/audio/speech`：

```json
{
  "input": "浙海一二三号船，请减速慢行。",
  "voice": "zhangsan",
  "language": "Chinese",
  "response_format": "pcm_framed"
}
```

选择另一个自定义音色时，将 `voice` 改为 `lisi`。参数名称是 `voice`；配置多个名称时分别传对应名称，无需都写成 `clone`。省略 `voice` 时使用默认音色。相同文本使用不同音色会分别生成和缓存。

未知名称返回 HTTP 404 和统一 JSON：

```json
{
  "code": 404,
  "message": "指定音色不存在",
  "data": null
}
```

成功响应是带显式完成帧的二进制 PCM 流，不返回新音色 ID，也不直接返回 WAV。使用仓库客户端接收并保存 WAV，在有 Python 3.10 或更新版本的客户端机器上执行：

```bash
python3 examples/http_streaming_client.py \
  --url http://127.0.0.1:8000 \
  --text '浙海一二三号船，请减速慢行。' \
  --voice zhangsan \
  --output zhangsan-announcement.wav

python3 examples/http_streaming_client.py \
  --url http://127.0.0.1:8000 \
  --text '浙海一二三号船，请减速慢行。' \
  --voice lisi \
  --output lisi-announcement.wav
```

远程调用时将 URL 换成服务器 IP。保存 WAV 只需要 Python 标准库，客户端无需安装模型依赖。已有输出文件不会被覆盖，需要换一个文件名。

`pcm_framed` 包含协议帧头，不能用 curl 直接保存成可播放的 `.wav`。其他客户端接入时参考 [二进制帧协议](README.md#二进制帧协议)。当前服务的 `speed` 只能省略或设为 `1.0`。

## 7. 新增或更换音色

将新录音及转写放入 `references/`，新增或修改 `voices.json` 中的条目；如需更换默认音色，同时修改 `.env`。然后重新创建容器：

```bash
docker compose up -d --force-recreate backend gateway
docker compose logs -f backend gateway
```

音色配置和参考特征在启动时读取，修改文件不会自动热更新。无需重新构建镜像。重新创建网关会清空其内存缓存；后端重新计算参考内容哈希，后续合成使用新的参考版本。

## 8. 单音色简化方式

也可以不使用 `voices.json`，在 `references/` 放入 `voice.wav` 和 `voice.txt`，配置：

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-Base
TTS_VOICES_FILE=
TTS_DEFAULT_VOICE=clone
```

此时音色名称固定为 `clone`，请求传 `"voice":"clone"` 或省略 `voice`。需要多个音色，或希望自定义名称时，使用前面的 `voices.json` 方式。

本文按当前代码和配置编写，未在本机构建镜像、加载 GPU 模型或执行行为测试。镜像构建、参考音频兼容性、克隆音质、显存和首包延迟需在目标 Linux 服务器验证。
