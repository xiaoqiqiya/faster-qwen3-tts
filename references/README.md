# 参考音色文件

只有 Base 克隆模型需要参考文件。把文件放在本目录，Compose 会只读挂载为容器内的 `/references`。

```text
references/
  voices.json
  zhangsan.wav
  zhangsan.txt
  lisi.wav
  lisi.txt
```

复制 `voices.example.json` 为 `voices.json`，修改录音和转写路径。相对路径以配置文件所在目录为基准。音色名不能仅大小写不同。

转写使用 UTF-8，内容与录音一致；录音大于 0 秒且不超过 30 秒，支持 1–8 声道和 8–192 kHz 采样率。启动时转换为单声道，追加 0.5 秒静音并预计算 ICL 音色特征。录音和转写变化后，重启 backend。

详情见 [音色部署文档](../docs/voices.md)。参考录音与个人配置默认被 Git 忽略。
