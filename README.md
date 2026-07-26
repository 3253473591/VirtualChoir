[中文](README.md) | [English](readme_en.md)

**Windows 桌面端虚拟合唱空间渲染工具** — 将多轨独唱录音导入虚拟声场，在三维房间中摆放歌手位置，配置麦克风阵列与混响参数，渲染出具有空间感的立体声合唱混音。

> **音频来源说明：** 本项目围绕**虚拟歌姬导出的歌声**（48 kHz / 32-bit Mono WAV）设计，暂未针对真人录音优化。但空间渲染管线不对音源类型做区分——只要格式相符，同样可以导入**单声道乐器的同规格音频**，在虚拟房间中模拟交响乐、室内乐等器乐合奏的声场布局。

---

## 开始

```powershell
git clone https://github.com/3253473591/VirtualChoir.git
pip install -r requirements.txt
python -m virtual_choir
```

> PyTorch 需单独安装（CPU 版或 CUDA 版），请从 https://download.pytorch.org/whl/torch/ 获取。

---

## 功能

| 功能 | 说明 |
|------|------|
| **三维定位** | 在 2D 俯视房间图中拖拽放置歌手，支持 X / Y / Z 坐标及独立增益 |
| **房间声学模拟** | 自定义房间长宽高、RT60 混响时间、混响增益 |
| **麦克风阵列** | 2–6 支麦克风，可调节间距与高度，模拟不同拾音方式 |
| **AI 空间推荐** | 对接 Google Gemini Native API 或 OpenAI 兼容接口，根据音频内容推荐歌手布局 |
| **随机偏移** | 导入 MIDI 文件，对歌声施加随机起音偏移（±5 ms），模拟真人合唱的非完美同步 |
| **音色差异化** | 复制轨道时自动施加音色变化（共振峰偏移、音高微调、EQ 曲线、颤音、气息混合） |

---

## 常见问题

**Q：AI 分析没有返回结果？**

A：请检查：① API 密钥是否正确配置；② 网络是否可正常访问 AI 服务；③ 音频是否包含足够长的有声段落（AI 仅分析最长 10 秒的纯有声片段，静音过长的文件可能无有效输出）。

**Q：工程中的人声位置数据存在哪里？**

A：所有歌手坐标、房间参数、麦克风配置等保存在工程的 `project_config.json` 中。API 密钥存储在 Windows 凭据管理器（Credential Manager）中，**不会**写入工程文件，便于工程分享。

**Q：如何清除渲染缓存强制重新渲染？**

A：删除工程目录下的 `.render_cache/` 文件夹，下次渲染时将全部重新处理。

---

技术栈：Python 3.11 · PySide6 · NumPy / SciPy · pyroomacoustics · librosa · OpenVPI PC-NSF-HiFiGAN · torchcrepe · soundfile · sounddevice
*本项目仅供学习和研究使用。使用他人录音作品进行混音时请确保拥有相关授权。*
