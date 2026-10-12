# 内置音色和自定义音色

## 内置音色：CustomVoice

下载 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 或 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 的完整模型，按 [部署文档](deployment.md) 挂载。

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
TTS_DEFAULT_VOICE=vivian
```

不需要参考录音，也不需要 voices.json。音色列表来自当前模型配置，官方模型包含 vivian、serena、uncle_fu、ryan、aiden、ono_anna、sohee、eric、dylan。Vivian 是内置音色名称，通过接口查询最终可用列表：

```bash
curl http://127.0.0.1:8000/v1/audio/voices
```

```json
{
  "code": 200,
  "message": "音色列表查询成功",
  "rows": [
    {"voice": "vivian", "type": "CustomVoice", "default": true}
  ]
}
```

上面只展示一项。列表没有分页，因此没有 total。0.6B 不接受非空 instruct；1.7B CustomVoice 可通过 instruct 传入语气指令。

## 多个自定义音色：Base + 参考音色

使用 Base 模型，不使用 CustomVoice 模型：

```bash
bash hfd.sh Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir /data/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base
```

准备目录：

```text
references/
  voices.json
  zhangsan.wav
  zhangsan.txt
  lisi.wav
  lisi.txt
```

voices.json：

```json
{
  "zhangsan": {"ref_audio": "zhangsan.wav", "ref_text_file": "zhangsan.txt"},
  "lisi": {"ref_audio": "lisi.wav", "ref_text_file": "lisi.txt"}
}
```

配置：

```dotenv
TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-Base
TTS_DEFAULT_VOICE=zhangsan
TTS_VOICES_FILE=/references/voices.json
```

最多配置 32 个音色。名称大小写按配置调用，不能只以大小写区分两个音色；不能包含斜杠、反斜杠或空字符。参考转写为 UTF-8、1–2048 字符，必须对应录音内容。录音大于 0 秒、不超过 30 秒，1–8 声道、8–192 kHz，建议使用干净的单人语音。

启动时逐个提取 ICL 特征并存入 backend-state 卷，然后启动官方引擎。多个音色共用同一套 Base 权重，不会每个音色加载一个完整模型；参考特征与并发状态仍消耗资源。重启会校验并复用已生成特征。修改参考文件、转写或模型后会生成新身份，旧音频缓存不会误命中。旧 profiles 目录不自动删除，频繁修改音色时可在停服务后清理不再使用的版本。

调用不同音色使用不同的 `voice`，不是全部写 clone：

```bash
curl -N http://127.0.0.1:8000/v1/audio/speech/sse \
  -H 'Content-Type: application/json' \
  -d '{"input":"前方船舶请注意。","voice":"zhangsan"}'
curl -N http://127.0.0.1:8000/v1/audio/speech/sse \
  -H 'Content-Type: application/json' \
  -d '{"input":"请减速慢行。","voice":"lisi"}'
```

自定义音色返回相同的流式音频协议。查询音色列表时返回 `voice: zhangsan`、`type: Base` 等条目。省略 voice 或传 null 时使用默认音色。没有单独的“创建音色”HTTP 接口；配置文件是音色来源，修改后重启 backend 生效。

## 当前范围

一个 backend 部署加载一个模型，选择 CustomVoice 或 Base。Base 当前用于参考音色克隆，不接受 instruct。VoiceDesign 和在线上传音色未开放。如需不同模型并存，应分别部署服务与缓存目录。
