# Qwen3-TTS Docker 流式服务

推理容器使用当前仓库的 `faster-qwen3-tts 0.5.4` 源码和 Torch CUDA Graph 后端；网关是独立的轻量 CPU 容器。默认只加载 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 一个模型。
两端通过 Compose 内部网络传输带显式终止帧的 HTTP 流式 PCM；客户端只访问网关。推理进程重启不会清空网关缓存；本版不引入 gRPC。
当前实现已提供语法检测，尚未构建 Docker 镜像、启动模型或执行行为测试。真实首包、音质、GPU 显存及依赖安装需要在目标服务器验证。

只使用内置音色、不克隆时，按 [Linux 内置音色部署步骤](custom-voice-linux.md) 操作，包含 hfd 下载、本地模型挂载、音色查询和调用示例。

使用自己的参考录音时，按 [Linux 自定义克隆音色部署步骤](voice-clone-linux.md) 操作，包含 Base 模型下载、参考文件路径、多个命名音色配置和接口选择方式。

## 构建和启动

目标服务器需要 NVIDIA 驱动、NVIDIA Container Toolkit，以及支持 `gpus` 字段的 Docker Compose 2.30 或更新版本。CUDA 12.8 镜像与 PyTorch 2.8.0/cu128 固定；镜像使用 `sdpa`，不需要额外编译 FlashAttention。

在项目根目录：

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend gateway
curl -s http://127.0.0.1:8000/health
```

Windows PowerShell 对应的复制命令：

```powershell
Copy-Item .env.example .env
docker compose build
docker compose up -d
docker compose logs -f backend gateway
Invoke-RestMethod http://127.0.0.1:8000/health
```

模型权重在启动时下载到 `model-cache` 卷，不打入镜像。首次下载可能较长，健康检查预留 30 分钟；网关等待后端健康后启动。后端只有在 CUDA 可用、模型加载、参考特征准备及 CUDA Graph 预热完成后才就绪。首次完整 codec 解码仍可能产生额外开销，性能验收要分别记录冷启动和稳定运行。

首次启动 `TTS_MODEL_REVISION=main` 会解析为具体 Hugging Face 提交；`GET /health` 的 `data.backend.model_revision` 返回该 SHA。将它写回 `.env`，再执行 `docker compose up -d`，之后固定模型版本。镜像始终安装本 checkout 的源码，不从 PyPI 下载另一个 faster 版本；健康信息也包含推理源码内容哈希。正式交付还应保存本 checkout 的 Git 提交及镜像 digest。直接依赖在 `deploy/requirements-*.txt` 固定，传递依赖仍由 pip 解析，构建步骤会执行 `pip check`。

默认端口仅绑定服务器 `127.0.0.1:8000`。远程客户端直连时将 `.env` 的 `TTS_BIND_ADDRESS=0.0.0.0`，然后重新创建网关；也可以通过现有反向代理访问。当前接口没有账户认证，开放端口时使用部署环境的访问控制。

## 音色和模型

### 使用 hfd 提前下载

服务支持直接加载完整的本地模型目录，无需将 hfd 下载结果转换成 Hugging Face 缓存结构。以下示例假定已安装的 hfd 使用常见 `--local-dir` 参数；如果你的版本参数不同，以 `hfd --help` 为准。

```bash
mkdir -p /data/qwen3-tts/models
hfd Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

如果工具以 `hfd.sh` 提供，使用 `./hfd.sh` 代替 `hfd`。请下载完整仓库，保留 `config.json`、全部权重、tokenizer 配置与 `speech_tokenizer/` 等子目录。

将 `compose.yaml` 中 `backend.volumes` 的模型挂载改为：

```yaml
volumes:
  - /data/qwen3-tts/models:/models
  - ./references:/references:ro
```

`.env` 使用容器内路径：

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_DEFAULT_VOICE=vivian
```

此时启动直接加载本地文件，不执行 `snapshot_download`；`TTS_MODEL_REVISION` 不参与本地目录加载。启动会读取模型文件计算内容哈希，健康接口中的 `model_revision` 为 `local-sha256:...`，确保替换模型后不混用网关旧缓存。文件较大时，哈希计算会增加启动的磁盘读取时间。下载完成后再启动容器，运行期间不修改模型目录。

完全离线运行时还可在 `backend.environment` 增加 `HF_HUB_OFFLINE: "1"`。然后正常执行 `docker compose build` 和 `docker compose up -d`；镜像构建仍需获取基础镜像和 Python 依赖。

### 模型与音色选择

默认 `voice=vivian`，支持的音色、语言从健康接口查询；请求不传 `voice` 时使用默认音色。第一版只支持单段文本、PCM 输出及 `speed=1.0`；不支持的语速、格式和未知参数会返回中文 JSON 错误，避免接受参数却不生效。0.6B CustomVoice 的上游实现忽略风格指令，因此本服务明确拒绝非空 `instruct`；需要指令可选择 1.7B CustomVoice，最终效果仍需试听。

固定音色模型切换例：

```dotenv
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
TTS_MODEL_REVISION=main
TTS_DEFAULT_VOICE=vivian
```

克隆模式先在 `references/` 放入 `voice.wav` 和对应 UTF-8 `voice.txt`，然后：

```dotenv
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
TTS_MODEL_REVISION=main
TTS_DEFAULT_VOICE=clone
```

```bash
docker compose up -d --force-recreate backend gateway
```

未设置 `TTS_VOICES_FILE` 时，Base 使用单个 ICL 克隆音色 `clone`。多个自定义音色通过 `references/voices.json` 配置不同名称：

```json
{
  "zhangsan": {"ref_audio": "zhangsan.wav", "ref_text_file": "zhangsan.txt"},
  "lisi": {"ref_audio": "lisi.wav", "ref_text_file": "lisi.txt"}
}
```

文件路径相对于配置文件；对应录音、转写也放在 `references/`。可复制 `references/voices.example.json` 为 `voices.json` 再修改，最多支持 32 个命名音色。`.env` 设置：

```dotenv
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
TTS_VOICES_FILE=/references/voices.json
TTS_DEFAULT_VOICE=zhangsan
```

如果用 hfd 下载，`TTS_MODEL` 改为容器内的 Base 模型目录，例如 `/models/Qwen3-TTS-12Hz-0.6B-Base`。默认音色留空时 Base 使用配置文件中的第一个名称，CustomVoice 使用 `vivian`。

合成请求通过 `"voice":"zhangsan"` 或 `"voice":"lisi"` 选择；`GET /health` 的 `data.backend.voices` 返回配置名称，`reference_versions` 返回每个音色的参考哈希。未知名称返回 404。同一文本使用不同 `voice` 会分别生成和缓存。

每个参考录音不超过 30 秒，转写非空且不超过 2048 字符；启动时给每个录音追加 0.5 秒静音并分别预计算参考特征，后续直接复用对应 `voice_clone_prompt`。所有音色的参考特征保存在 CPU 内存；生成时只将选中的音色特征放入 GPU，生成保持单并发。模型只加载一份；增加音色主要增加 CPU 特征内存和启动耗时，特征提取及推理峰值显存仍需远端验证。更换音色配置、录音、转写或默认音色后重新创建 backend；参考内容哈希进入缓存键。配置多个参考音色需要 Base 模型，CustomVoice 使用内置音色；不会同时加载 Base 和 CustomVoice。

## 请求和持续播放

接口：`POST /v1/audio/speech`。

```json
{
  "input": "浙海一二三号船，请减速慢行。",
  "voice": "vivian",
  "language": "Chinese",
  "instruct": "",
  "speed": 1.0,
  "response_format": "pcm_framed"
}
```

文本精确匹配，保留空格和标点，最多 2048 字符。相同有效默认值和明确参数会共享任务；不同文本、音色、语言、指令、模型版本、参考版本、采样配置不会混用。两种传输封装共享相同的 PCM 结果。

使用交付的客户端边接收边保存 WAV：

```bash
python examples/http_streaming_client.py --url http://SERVER:8000 \
  --text '浙海一二三号船，请减速慢行。' --output announcement.wav
```

客户端安装 `sounddevice` 后可持续播放，不需要安装模型或 Web 服务依赖：

```bash
pip install sounddevice
python examples/http_streaming_client.py --url http://SERVER:8000 \
  --text '浙海一二三号船，请减速慢行。' --play --output announcement.wav
```

克隆模式可省略 `--voice`，或传 `--voice clone`。客户端使用一个持续的 `RawOutputStream`，解析不完整帧头和采样字节；只在收到完成帧后认定成功。如果流失败，会删除本次创建的不完整 WAV；已经播放的部分无法撤回。已存在的输出文件不会被覆盖。

如果接入方已经处理 HTTP 传输错误，可以请求原始 PCM：

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"浙海一二三号船，请减速慢行。","response_format":"pcm"}' \
  --output announcement.pcm
```

原始 PCM 失败时网关抛出异常，让 HTTP 流异常终止，绝不向 PCM 插入 JSON。不同代理可能改变连接终止行为；因此可靠的业务完成确认使用默认 `pcm_framed`。

## 二进制帧协议

成功响应头：

```text
Content-Type: application/vnd.qwen.pcm-stream   # pcm_framed
X-Audio-Protocol: qwen-pcm-v1
X-Audio-Sample-Rate: 24000                     # 以实际响应为准
X-Audio-Channels: 1
X-Audio-Sample-Format: s16le
X-Accel-Buffering: no
Cache-Control: no-store
```

每个帧的头是 **1 字节类型 + 4 字节无符号大端长度**，后面紧跟对应长度的 payload。PCM payload 使用有符号 16 位小端、单声道。

| 类型 | 值 | payload |
|---|---|---|
| 音频 | 1 | 不超过 256 KiB、包含完整采样的 PCM |
| 完成 | 2 | 长度 0，明确表示生成成功 |
| 失败 | 3 | UTF-8 JSON：数字 `code`、中文 `message`、`data` |

帧与 HTTP/TCP 读取边界无关。读取者必须保留不完整帧头、payload 和采样字节；断开时缺少完成帧就是失败。失败帧只是已开始的二进制协议内事件，HTTP 状态仍为已发送的 200，不能再修改；其中的 `code=500` 表示生成错误事件，不能当作新的 HTTP 响应。

开始发送前的校验、准入或首块生成失败均使用统一 JSON：`{"code":400,"message":"请求参数校验失败","data":...}`，数字 code 与 HTTP 状态一致。查询同样使用统一格式。流开始后绝不将裸 JSON 拼进音频。

## 缓存和资源限制

| 环境变量 | 默认值 | 行为 |
|---|---:|---|
| `TTS_CACHE_TTL_SECONDS` | 300 | 从完成与最后一次请求访问的较晚者计算，命中续期 |
| `TTS_CACHE_HOST_DIR` | `./cache` | 宿主机缓存目录，Compose 挂载到网关 `/cache` |
| `TTS_CACHE_DIR` | `/cache/audio`（Compose） | 网关读写 PCM 文件和完成元数据的容器内目录 |
| `TTS_CACHE_MAX_MB` | 256 | 磁盘 PCM 缓存和活跃任务预留容量，单位 MB |
| `TTS_MAX_READERS` | 256 | 最多 256 个音频读取连接，同文本连接也计数 |
| `TTS_TASK_TIMEOUT_SECONDS` | 120 | 网关从创建任务开始计时，包含排队；后端从生成开始硬计时 |
| `TTS_SEND_TIMEOUT_SECONDS` | 15 | 每个网络发送操作的独立超时 |
| `TTS_CACHE_CLEANUP_SECONDS` | 5 | 无访问时定期清理 |
| `TTS_CHUNK_SIZE` | 8 | 每个生成块的 codec token 数，需按远端首包与听感调整 |
| `TTS_MAX_NEW_TOKENS` | 1024 | 到达上限而未遇到 EOS 时报告失败，不缓存截断音频 |
| `TTS_MAX_SEQ_LEN` | 2048 | CUDA Graph 序列缓存上限，需大于生成 token 上限 |

缓存容量填写正整数，`1 MB = 1024 * 1024` 字节。已有 `.env` 请将旧的 `TTS_CACHE_MAX_BYTES` 改为 `TTS_CACHE_MAX_MB` 并换算数值，例如 `268435456` 字节改为 `256` MB。单任务输出和等待任务数不再提供配置项，可删除旧的 `TTS_MAX_OUTPUT_BYTES`、`TTS_MAX_OUTPUT_MB`、`TTS_MAX_PENDING_TASKS`；这些变量不再读取。健康接口中的字节统计仍以 bytes 返回，方便精确核对文件大小。

内部仍保护单任务最多 8 MB PCM（24 kHz 下约 175 秒）和一个执行任务之外最多 8 个等待任务。缓存总容量小于 8 MB 时，单任务上限自动降到缓存总容量，避免无法预留空间。相同请求共享任务，不重复占用等待名额。

首次准入原子创建共享 producer，并按最大输出量预留磁盘容量；完成后按实际文件大小释放多余预留。活跃生成和活跃读取的数据计入容量，不能淘汰；容量不足优先淘汰无人读取的已完成 LRU 条目，再拒绝新任务。实际磁盘空间不足或不可写时返回中文错误。这里限制的是 PCM 文件与任务预留，不包含文件系统分配和 JSON 元数据开销，也不是整个 Python 进程 RSS 上限。

网关收到 PCM 后立即追加到磁盘文件，写入完成后唤醒读取者。每个读取者从文件偏移 0 开始，补发已有音频后等待新数据；读取块最多 256 KiB，发送块边界可以与后端生成块边界不同，PCM 字节顺序保持一致。网关只在内存维护任务索引和临时读写块，不长期保存音频字节列表，也不创建每客户端音频队列。某个或全部客户端断开时，共享生成仍继续直到完成或到达任务上限。

Compose 默认将项目 `./cache` 挂载到网关 `/cache`，实际文件位于宿主机 `./cache/audio/`。如需固定 Linux 路径，在 `.env` 设置 `TTS_CACHE_HOST_DIR=/data/qwen3-tts/cache`，启动前执行 `mkdir -p /data/qwen3-tts/cache` 并保证容器可写。该目录必须只供一个网关实例使用，不使用 tmpfs；运行期间不要手工修改缓存文件。

生成成功后同步 PCM 文件并原子保存 JSON 完成元数据；JSON 只记录请求键哈希、文件大小、完成时间和最后访问时间，不保存原始请求文本。容器重启或重新创建后从同一挂载目录恢复未过期的成功缓存，命中仍会持久化续期。TTL 使用持久化的系统时间，停机时间计入过期；`TTS_CACHE_TTL_SECONDS` 决定过期时长，重启不会重置最后访问时间，也没有独立的五分钟门控。排队和任务超时仍使用单调时间。启动时删除过期、文件大小不匹配、缺少完成元数据的半成品以及遗留临时文件；容量调小时按 LRU 清理。

模型和音色版本仍参与请求键，版本一致才命中旧文件。网关重启后需先从后端确认当前版本；运行期间后端短暂不可用时，继续使用上一次确认的版本提供缓存。更换缓存挂载路径会使用新目录，原目录不自动搬迁。磁盘缓存可在重启后保留，但正在生成的任务和客户端连接不能恢复。

推理线程及后端网络桥只有两个帧的缓冲，GPU 同时只执行一个不同请求。Python 线程不能安全中断卡住的 CUDA 调用，后端达到硬超时会退出进程，由 Docker `restart: unless-stopped` 重启，释放 GPU；此时网关报告失败，健康检查短暂返回 503。超时后的下一任务可能遇到后端仍忙或重启而失败，需要由调用方重试，不能悄悄重叠执行两个推理。

后端暂时不可用时，网关按上一次确认的模型与音色版本继续提供尚未过期的缓存；未命中的新任务返回 503。健康接口仍报告依赖不可用。后端恢复后，网关重新获取实际版本；如果模型或参考内容变更，使用新请求键，不混用旧版本结果。

必须保持单网关进程，一个缓存目录只供一个网关使用，不能让多个 worker 或副本同时读写同一目录；本实现不引入 Redis 或跨进程广播。

## 代理与远端验收

如使用 Nginx，将 [nginx-streaming.conf](nginx-streaming.conf) 的 location 放入现有 server；按实际网络修改 upstream。除网关响应头外，还需要关闭代理响应缓存和 gzip，避免中间层攒整段音频。

CPU 行为测试已交付，但本机未执行。远端可使用网关镜像的已安装依赖运行它，不启动 GPU 模型：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/tests:/checks:ro" -v "$PWD/examples:/app/examples:ro" \
  gateway python -m unittest discover -s /checks -p test_streaming_service.py -v
```

这些用例覆盖 100 个相同请求只调用一次 producer、第三块后的晚加入、断开隔离、滑动 TTL、活跃读取保护、LRU、容量预留、有界排队、输出和总耗时上限、失败后重试、发送超时和帧拆分，以及磁盘缓存重启恢复、续期持久化、过期/损坏/半成品清理和磁盘写入失败。它们不证明真实 GPU 性能。

真实 GPU 验收按根目录 [计划](../qwen3-tts-streaming-plan.md) 的表格执行。后端日志记录首块、生成结束和 GPU allocated/reserved 峰值；网关记录排队、收到首块、发送首块和完成；客户端记录收到首块和完成。不同主机的单调时间不能直接相减，应分别使用本机耗时，并在需要跨机对齐时同步时钟或抓包。

需要确认：首块到达客户端时后端仍在生成、100 个相同请求仅触发一次后端推理、晚加入完整无重复、中文船名与数字及音频块边界听感、稳定运行首包 P50/P95、空闲与峰值显存、不同文本的排队耗时，以及经过实际反向代理后的失败事件。Docker 构建和 GPU 运行尚未验收。
