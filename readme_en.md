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
| 3D Positioning | Drag singers on a top-down room map; supports X / Y / Z coordinates and per-track gain |
| Room Acoustics | Configurable room dimensions (L×W×H), RT60 reverb time, and reverb gain |
| Microphone Array | 2–6 virtual microphones with adjustable spacing and height to model different pickup patterns |
| AI Spatial Recommendations | Connects to Google Gemini Native API or any OpenAI-compatible endpoint; analyzes audio content to suggest singer layouts |
| Randomized Timing Offsets | Import MIDI files to apply random onset/offset jitter, simulating the natural timing imperfections of a human choir |
| Timbre Variation | Automatically applies vocal differentiation (formant shift, pitch detune, EQ curves, vibrato, breath mix) via OpenVPI PC-NSF-HiFiGAN neural vocoder when duplicating tracks |
| Voice Style and Articulation | Choose popular, bel canto, or child voice processing; presets 3/4/5 provide low, medium, and high articulation differences using lyric-aware consonant-to-vowel boundaries |

---

## Timbre Variation

Choose “Generate differentiated copies” from a track menu to set the copy count, voice style, and variation preset. Voice styles change the processing profile and do not require additional bel canto or child voice training data:

| Voice style | Processing focus |
|-------------|------------------|
| Popular | Keeps consonants clear while adding a moderate articulation contour |
| Bel canto | Softens sharp consonants and emphasizes connected vowel onsets and tails |
| Child | Enhances consonant transients and high-frequency clarity with shorter, brighter vowel entries |

Articulation intensity is tied to the variation preset: presets 1 and 2 leave articulation unchanged, while presets 3, 4, and 5 correspond to low, medium, and high differences. The feature requires a lyric-bearing MIDI assignment for the track. Lyrics such as `jia` or the Chinese character `家` are converted to pinyin with `pypinyin`, and the actual consonant onset is detected near the MIDI note timestamp. Without lyric MIDI, the existing pitch and timbre variation pipeline still works and only articulation processing is skipped.

To generate a listening comparison with levels 1, 3, and 5 and three copies per level:

```powershell
python tools\timbre_variation_comparison.py singer.wav --midi singer.mid --voice-style child
```

The tool writes a `manifest.json` containing the source audio, preset levels, voice style, and generated files. `--voice-style` accepts `popular`, `bel_canto`, or `child`.

---

## FAQ

**Q: AI analysis returns no results?**

A: Check: ① that your API key is correctly configured; ② that the network can reach your AI provider; ③ that the audio contains sufficient voiced content (the AI only analyzes up to 10 seconds of detected voiced segments; files that are mostly silence may produce no output).

**Q: Where is singer positioning data stored?**

A: All singer coordinates, room parameters, and microphone settings are saved in the project's `project_config.json`. API keys are stored in the Windows Credential Manager and are **never** written to project files, so projects can be shared safely.

**Q: How do I clear the render cache to force a full re-render?**

A: Delete the `.render_cache/` folder inside the project directory. The next render will reprocess everything from scratch.

---

Tech stack: Python 3.11 · PySide6 · NumPy / SciPy · pyroomacoustics · librosa · OpenVPI PC-NSF-HiFiGAN (neural vocoder for timbre variation re-synthesis) · torchcrepe · pypinyin · soundfile · sounddevice

*This project is for learning and research purposes only. Please ensure you have the appropriate rights before mixing third-party recordings.*
