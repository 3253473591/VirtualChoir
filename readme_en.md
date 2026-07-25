[中文](README.md) | [English](readme_en.md)

**Windows desktop virtual choir spatial rendering tool** — import multi-track solo vocal recordings into a virtual acoustic space, position singers in a 3D room, configure microphone arrays and reverb parameters, then render a spatially immersive stereo choir mix.

> **Audio source notes:** This project is designed around **vocals exported from virtual singers** (48 kHz / 32-bit Mono WAV) and has not been optimized for live human recordings. That said, the spatial rendering pipeline is source-agnostic — as long as the format matches, you can also import **mono instrument tracks at the same specs** to simulate symphony orchestras, chamber ensembles, and other instrumental arrangements within the virtual room.

---

## Getting Started

```powershell
git clone https://github.com/3253473591/VirtualChoir.git
pip install -r requirements.txt
python -m virtual_choir
```

> PyTorch must be installed separately (CPU or CUDA version). Download from https://download.pytorch.org/whl/torch/.

---

## Features

| Feature | Description |
|---------|-------------|
| **3D Positioning** | Drag singers on a top-down room map; supports X / Y / Z coordinates and per-track gain |
| **Room Acoustics** | Configurable room dimensions (L×W×H), RT60 reverb time, and reverb gain |
| **Microphone Array** | 2–6 virtual microphones with adjustable spacing and height to model different pickup patterns |
| **AI Spatial Recommendations** | Connects to Google Gemini Native API or any OpenAI-compatible endpoint; analyzes audio content to suggest singer layouts |
| **Randomized Timing Offsets** | Import MIDI files to apply random onset jitter (±5 ms), simulating the natural timing imperfections of a human choir |
| **Timbre Variation** | Automatically applies vocal differentiation (formant shift, pitch detune, EQ curves, vibrato, breath mix) when duplicating tracks |

---

## FAQ

**Q: AI analysis returns no results?**

A: Check: ① that your API key is correctly configured; ② that the network can reach your AI provider; ③ that the audio contains sufficient voiced content (the AI only analyzes up to 10 seconds of detected voiced segments; files that are mostly silence may produce no output).

**Q: Where is singer positioning data stored?**

A: All singer coordinates, room parameters, and microphone settings are saved in the project's `project_config.json`. API keys are stored in the Windows Credential Manager and are **never** written to project files, so projects can be shared safely.

**Q: How do I clear the render cache to force a full re-render?**

A: Delete the `.render_cache/` folder inside the project directory. The next render will reprocess everything from scratch.

---

Tech stack: Python 3.11 · PySide6 · NumPy / SciPy · pyroomacoustics · librosa · pyworld · torchcrepe · soundfile · sounddevice

*This project is for learning and research purposes only. Please ensure you have the appropriate rights before mixing third-party recordings.*
