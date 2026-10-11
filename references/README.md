# 克隆参考文件

自定义参考音色需要 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 或 1.7B Base 模型，也支持 hfd 下载的完整 Base 模型目录。

完整的 hfd 下载、Docker 挂载和调用步骤见 [Linux 自定义克隆音色部署说明](../deploy/voice-clone-linux.md)。

多个音色先将 `voices.example.json` 复制为 `voices.json`，给每个声音取不同名称并配置录音与转写：

```json
{
  "zhangsan": {"ref_audio": "zhangsan.wav", "ref_text_file": "zhangsan.txt"},
  "lisi": {"ref_audio": "lisi.wav", "ref_text_file": "lisi.txt"}
}
```

相对路径以 `voices.json` 所在目录为准。每个音色都准备自己的文件，然后在 `.env` 配置：

```dotenv
TTS_VOICES_FILE=/references/voices.json
TTS_DEFAULT_VOICE=zhangsan
```

请求分别传 `"voice": "zhangsan"` 或 `"voice": "lisi"`。默认音色留空时使用配置文件中的第一个名字；健康接口的 `data.backend.voices` 返回这些名字，`reference_versions` 返回各自的参考内容哈希。最多配置 32 个音色，名称不能重复。

不设置 `TTS_VOICES_FILE` 时沿用单音色配置，名称为 `clone`：

- `voice.wav`：不超过 30 秒的清晰参考录音。
- `voice.txt`：录音对应的 UTF-8 中文或其他语言转写，非空且不超过 2048 字符。

文件以只读挂载提供给推理容器，镜像不包含参考录音。每个参考音色启动时分别追加 0.5 秒静音并预计算 ICL 克隆特征；特征保存在 CPU 内存，生成时只将选中的音色特征放入 GPU。模型只加载一份，生成保持单并发。
更换配置、参考文件或默认音色后重新创建 backend 容器；健康接口返回参考内容哈希，后续请求使用新的请求键。
