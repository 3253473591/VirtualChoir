from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import ChoirError


def read_source_wav(path: Path) -> np.ndarray:
    """Validate and decode the immutable 48 kHz / 32-bit mono source."""
    if path.suffix.lower() != ".wav": raise ChoirError("AUDIO_FORMAT_UNSUPPORTED", "扩展名必须为 .wav")
    if not path.exists(): raise ChoirError("AUDIO_NOT_FOUND", str(path))
    try:
        info = sf.info(path)
        if info.channels != 1 or info.samplerate != 48000 or info.subtype not in {"FLOAT", "PCM_32"}:
            raise ChoirError("AUDIO_FORMAT_UNSUPPORTED", f"需要 48kHz/32-bit/mono，实际为 {info.samplerate}Hz/{info.subtype}/{info.channels}ch")
        data, rate = sf.read(path, dtype="float32", always_2d=False)
    except ChoirError: raise
    except PermissionError as exc: raise ChoirError("AUDIO_PERMISSION_DENIED", str(exc)) from exc
    except Exception as exc: raise ChoirError("AUDIO_FORMAT_UNSUPPORTED", str(exc)) from exc
    if rate != 48000 or data.size == 0: raise ChoirError("AUDIO_EMPTY")
    if not np.isfinite(data).all(): raise ChoirError("AUDIO_NON_FINITE")
    return np.asarray(data, dtype=np.float32)


AI_SAMPLE_RATE = 44100
AI_MAX_SECONDS = 10


def preprocess_for_ai(path: Path) -> tuple[bytes, dict[str, int | float | bool]]:
    """Produce at most ten seconds of voiced 44.1 kHz / 16-bit dry audio."""
    source = read_source_wav(path)
    return preprocess_samples_for_ai(source)


def preprocess_samples_for_ai(source: np.ndarray) -> tuple[bytes, dict[str, int | float | bool]]:
    """Encode an in-memory 48 kHz dry layer for an AI request."""
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
        raise ChoirError("AI_AUDIO_PREPROCESS_FAILED")
    resampled = resample_poly(source.astype(np.float64), 147, 160, window=("kaiser", 8.6)).astype(np.float32)
    if resampled.size == 0 or not np.isfinite(resampled).all(): raise ChoirError("AI_AUDIO_PREPROCESS_FAILED")
    selected = _select_voiced_audio(resampled, AI_SAMPLE_RATE, AI_MAX_SECONDS)
    if selected.size == 0:
        selected = resampled[: min(len(resampled), AI_SAMPLE_RATE * AI_MAX_SECONDS)]
    peak = float(np.max(np.abs(selected)))
    clipped = bool(peak > 1.0)
    pcm = np.rint(np.clip(selected, -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO(); sf.write(out, pcm, AI_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return out.getvalue(), {
        "sample_rate_hz": AI_SAMPLE_RATE, "bit_depth": 16, "channels": 1,
        "duration_s": len(selected) / AI_SAMPLE_RATE, "peak": peak, "clipped": clipped,
    }


def _select_voiced_audio(audio: np.ndarray, sample_rate: int, max_seconds: int) -> np.ndarray:
    """Keep sequential voiced regions and crossfade joins up to the AI limit."""
    target = sample_rate * max_seconds
    frame = max(1, int(sample_rate * 0.05))
    frame_count = int(np.ceil(len(audio) / frame))
    padded = np.pad(audio, (0, frame_count * frame - len(audio)))
    energy = np.sqrt(np.mean(padded.reshape(frame_count, frame) ** 2, axis=1))
    threshold = max(float(np.max(energy)) * 0.08, 1e-5)
    active = energy >= threshold
    regions: list[tuple[int, int]] = []
    start = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            regions.append((start * frame, min(index * frame, len(audio))))
            start = None
    if start is not None:
        regions.append((start * frame, len(audio)))
    if not regions:
        return audio[:target]

    pieces: list[np.ndarray] = []
    remaining = target
    for left, right in regions:
        if remaining <= 0:
            break
        piece = audio[left:min(right, left + remaining)]
        if len(piece):
            pieces.append(piece)
            remaining -= len(piece)
    if not pieces:
        return audio[:target]
    join = min(int(sample_rate * 0.005), *(len(piece) // 2 for piece in pieces))
    selected = pieces[0].copy()
    for piece in pieces[1:]:
        if join > 0:
            fade = np.linspace(0.0, 1.0, join, endpoint=False, dtype=np.float32)
            selected[-join:] = selected[-join:] * (1.0 - fade) + piece[:join] * fade
            selected = np.concatenate((selected, piece[join:]))
        else:
            selected = np.concatenate((selected, piece))
    return selected[:target]


def write_preview(path: Path, stereo_48k: np.ndarray) -> None:
    resampled = resample_poly(stereo_48k.astype(np.float64), 147, 160, axis=0, window=("kaiser", 8.6))
    pcm = np.rint(np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True); sf.write(path, pcm, 44100, subtype="PCM_16")


def write_export(path: Path, stereo_48k: np.ndarray) -> None:
    if not np.isfinite(stereo_48k).all(): raise ChoirError("RENDER_FAILED", "输出包含非有限样本")
    path.parent.mkdir(parents=True, exist_ok=True); sf.write(path, stereo_48k.astype(np.float32), 48000, subtype="FLOAT")


def play_preview(path: Path) -> None:
    """Start non-blocking stereo preview playback on the default output device."""
    data, rate = load_playback_audio(path)
    play_audio_segment(data, rate)


def load_playback_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a rendered mix or stem for the preview transport controls."""
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        if data.size == 0 or data.shape[1] not in {1, 2} or not np.isfinite(data).all():
            raise ValueError("preview audio is empty or invalid")
        return np.asarray(data, dtype=np.float32), int(rate)
    except Exception as exc: raise ChoirError("PLAYBACK_DEVICE_ERROR", str(exc)) from exc


def play_audio_segment(data: np.ndarray, sample_rate: int, start_frame: int = 0) -> None:
    """Play from a frame offset, allowing the GUI to implement pause and seek."""
    try:
        import sounddevice as sd
        start_frame = max(0, min(int(start_frame), len(data)))
        if start_frame < len(data):
            sd.play(data[start_frame:], sample_rate, blocking=False)
    except Exception as exc: raise ChoirError("PLAYBACK_DEVICE_ERROR", str(exc)) from exc


def stop_playback() -> None:
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass
