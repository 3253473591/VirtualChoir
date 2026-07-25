"""Preset-based vocal differentiation for virtual choir copies.

Each copy uses the same user-selected preset but a distinct deterministic
seed. CREPE supplies the pitch contour, WORLD resynthesizes the formant and
pitch variations, and librosa locates a high-energy vowel-like region.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .audio import read_source_wav
from .cuda_acceleration import apply_pitch_cents, fft_convolve as cuda_fft_convolve
from .errors import ChoirError
from .models import TimbreVariationConfig
from .naturalization import LyricUnit, parse_midi_notes

_FFT_FILTER_THRESHOLD_SAMPLES = 480_000
_FFT_FILTER_IMPULSE_SAMPLES = 8_192
_CUDA_FFT_THRESHOLD_SAMPLES = 1_000_000
_WORLD_FRAME_PERIOD_MS = 5.0
_CREPE_HOP_SAMPLES = 240
# WORLD is intentionally blended with the original recording to retain phase,
# transients, and breath detail that a full vocoder resynthesis cannot recover.
# Adjust this single value for future voicing tests: 0.70 means 70% WORLD.
WORLD_WET_MIX = 0.70


@dataclass(frozen=True)
class _WorldAnalysis:
    f0: np.ndarray
    spectral_envelope: np.ndarray
    aperiodicity: np.ndarray
    crepe_f0: np.ndarray
    crepe_periodicity: np.ndarray

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_variations(
    source_path: Path,
    copy_count: int,
    output_dir: Path,
    config: TimbreVariationConfig | None = None,
    cancel_event: threading.Event | None = None,
    progress=None,
    midi_path: Path | None = None,
    midi_track_index: int | None = None,
) -> list[Path]:
    """Generate *copy_count* timbre-differentiated WAV copies.

    Returns a list of absolute paths to the written 48 kHz / 32-bit mono WAVs.
    Metadata (JSON) is written alongside each copy.
    """
    if copy_count < 1 or copy_count > 64:
        raise ChoirError("PROJECT_SCHEMA_ERROR", "副本数量必须在 1-64 之间")

    config = config or TimbreVariationConfig()
    config.validate()

    source = read_source_wav(source_path)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_units, vibrato_mode = _load_vibrato_notes(
        midi_path, len(source) / 48000, midi_track_index,
    )

    stem = source_path.stem
    written: list[Path] = []
    # CREPE/WORLD analysis is shared by every copy. The expensive per-copy
    # synthesis and post-processing are independent and can use CPU cores.
    if progress:
        progress(2, "正在分析音高与共振峰…")
    analysis = _analyze_voice(source, 48000, cancel_event)
    analysis_cache: dict[str, _WorldAnalysis] = {"world": analysis}
    jobs = []
    for idx in range(1, copy_count + 1):
        seed = _derive_seed(source_sha, idx)
        rng = np.random.default_rng(seed)
        params = _sample_params(rng, config)
        params["vibrato_mode"] = vibrato_mode
        params["vibrato_midi_note_count"] = len(midi_units or ())
        jobs.append((idx, seed, params))

    processed: dict[int, np.ndarray] = {}
    worker_count = min(copy_count, max(1, min(8, os.cpu_count() or 1)))
    try:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="choir-var") as pool:
            futures = {
                pool.submit(
                    _process_copy, source, params, analysis_cache, seed, cancel_event, midi_units
                ): (idx, seed, params)
                for idx, seed, params in jobs
            }
            completed = 0
            for future in as_completed(futures):
                _check_cancel(cancel_event)
                idx, _seed, _params = futures[future]
                processed[idx] = future.result()
                completed += 1
                if progress:
                    progress(
                        2 + round(completed / copy_count * 90),
                        f"正在并行生成副本 {completed}/{copy_count}…",
                    )
    except ChoirError:
        _remove_generated_files(written)
        raise
    except Exception as exc:
        _remove_generated_files(written)
        raise ChoirError("RENDER_FAILED", f"差异化处理失败：{exc}") from exc

    for idx, seed, params in jobs:
        _check_cancel(cancel_event)
        out_path = _find_available_path(output_dir, stem, idx)
        _write_wav(out_path, processed[idx])
        _write_meta(out_path.with_suffix(".json"), idx, seed, config.preset_level, params)
        written.append(out_path)

    if progress:
        progress(100, f"已生成 {copy_count} 份差异化副本")

    return written


def _process_copy(
    source: np.ndarray,
    params: dict,
    analysis_cache: dict[str, _WorldAnalysis],
    seed: int,
    cancel_event: threading.Event | None,
    midi_units: tuple[LyricUnit, ...] | None,
) -> np.ndarray:
    """Run one independent copy with a deterministic per-copy generator."""
    _check_cancel(cancel_event)
    return _apply_timbre_variation(
        source, params, 48000, np.random.default_rng(seed), analysis_cache,
        cancel_event=cancel_event, midi_units=midi_units,
    )


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ChoirError("RENDER_CANCELLED")


def _remove_generated_files(written: list[Path], pending_path: Path | None = None) -> None:
    paths = list(written)
    if pending_path is not None:
        paths.append(pending_path)
    for wav_path in paths:
        for path in (wav_path, wav_path.with_suffix(".json")):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Seed derivation & parameter sampling
# ---------------------------------------------------------------------------


def _derive_seed(source_sha256: str, copy_index: int) -> int:
    payload = f"{source_sha256}:{copy_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 64) - 1)


def _sample_params(rng: np.random.Generator, config: TimbreVariationConfig) -> dict:
    def _uniform(pair):
        return float(rng.uniform(pair[0], pair[1]))

    return {
        "formant_shift": round(_uniform(config.formant_shift_range), 4),
        "pitch_shift_cents": round(_uniform(config.pitch_shift_cents_range), 2),
        "pitch_line_cents": round(_uniform(config.pitch_line_cents_range), 2),
        "vowel_onset_db": round(_uniform(config.vowel_onset_db_range), 2),
        "dynamic_db": round(_uniform(config.dynamic_db_range), 2),
        "eq_mid_db": round(_uniform(config.eq_mid_db_range), 2),
        "eq_high_db": round(_uniform(config.eq_high_db_range), 2),
        "breath_mix": round(_uniform(config.breath_mix_range), 4),
        "vibrato_depth_cents": round(_uniform(config.vibrato_depth_cents_range), 2),
        "vibrato_rate_hz": round(_uniform(config.vibrato_rate_hz_range), 3),
        "vibrato_note_probability": round(float(config.vibrato_note_probability), 3),
    }


# ---------------------------------------------------------------------------
# Timbre variation pipeline — each step is individually guarded so one
# failure doesn't lose the whole copy.
# ---------------------------------------------------------------------------


def _apply_timbre_variation(
    source: np.ndarray,
    params: dict,
    sample_rate: int,
    rng: np.random.Generator,
    analysis_cache: dict[str, _WorldAnalysis] | None = None,
    cancel_event: threading.Event | None = None,
    midi_units: tuple[LyricUnit, ...] | None = None,
) -> np.ndarray:
    """Apply WORLD resynthesis followed by EQ, energy, dynamics and breath."""
    _check_cancel(cancel_event)
    if analysis_cache is not None and "world" in analysis_cache:
        analysis = analysis_cache["world"]
    else:
        analysis = _analyze_voice(source, sample_rate, cancel_event)
        if analysis_cache is not None:
            analysis_cache["world"] = analysis

    audio = _resynthesize_world(
        analysis, params, sample_rate, rng, cancel_event, midi_units=midi_units,
    )
    audio = _match_length(audio, len(source))
    dry = source.astype(np.float32, copy=False)
    audio = dry * np.float32(1.0 - WORLD_WET_MIX) + audio * np.float32(WORLD_WET_MIX)
    _check_cancel(cancel_event)

    # Preserve the existing post-processing controls after WORLD synthesis.
    _check_cancel(cancel_event)
    if abs(params["eq_mid_db"]) > 0.05 or abs(params["eq_high_db"]) > 0.05:
        try:
            audio = _apply_eq(audio, sample_rate, params["eq_mid_db"], params["eq_high_db"])
        except ChoirError:
            raise
        except Exception:
            pass

    _check_cancel(cancel_event)
    try:
        audio = _apply_vowel_energy_gain(audio, sample_rate, params["vowel_onset_db"])
        audio = _apply_dynamic_variation(audio, sample_rate, params["dynamic_db"], rng)
    except Exception:
        pass

    # Breath is intentionally post-vocoder so it remains distinct from WORLD's
    # aperiodicity component.
    _check_cancel(cancel_event)
    if params["breath_mix"] > 1e-6:
        try:
            audio = _mix_breath(
                audio, sample_rate, params["breath_mix"], rng, cancel_event
            )
        except ChoirError:
            raise
        except Exception:
            pass

    _check_cancel(cancel_event)
    np.clip(audio, -2.0, 2.0, out=audio)
    return audio.astype(np.float32, copy=False)


def _optional_dependencies():
    """Load heavyweight audio dependencies only when the feature is used."""
    try:
        import librosa
        import pyworld
        import torch
        import torchcrepe
    except Exception as exc:
        raise ChoirError(
            "RENDER_DEPENDENCY_MISSING",
            "差异化需要 librosa、pyworld、torch 和 torchcrepe；请安装 requirements.txt",
        ) from exc
    return librosa, pyworld, torch, torchcrepe


def _analyze_voice(
    source: np.ndarray, sample_rate: int, cancel_event: threading.Event | None
) -> _WorldAnalysis:
    """Analyze one source once; all copies reuse these expensive features."""
    _check_cancel(cancel_event)
    _librosa, pyworld, torch, torchcrepe = _optional_dependencies()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = "full" if device == "cuda" else "tiny"
        tensor = torch.from_numpy(source.astype(np.float32, copy=False)).unsqueeze(0)
        with torch.no_grad():
            crepe_f0, periodicity = torchcrepe.predict(
                tensor, sample_rate, _CREPE_HOP_SAMPLES, 50, 1100,
                model=model, batch_size=2048, device=device, return_periodicity=True,
            )
        crepe_f0_np = np.asarray(crepe_f0.squeeze(0).cpu(), dtype=np.float64)
        periodicity_np = np.asarray(periodicity.squeeze(0).cpu(), dtype=np.float64)
        f0, time_axis = pyworld.harvest(source.astype(np.float64), sample_rate, frame_period=_WORLD_FRAME_PERIOD_MS)
        spectral = pyworld.cheaptrick(source.astype(np.float64), f0, time_axis, sample_rate)
        aperiodicity = pyworld.d4c(source.astype(np.float64), f0, time_axis, sample_rate)
    except Exception as exc:
        raise ChoirError("RENDER_FAILED", f"CREPE/WORLD 音频分析失败：{exc}") from exc
    _check_cancel(cancel_event)
    return _WorldAnalysis(f0, spectral, aperiodicity, crepe_f0_np, periodicity_np)


def _resynthesize_world(
    analysis: _WorldAnalysis,
    params: dict,
    sample_rate: int,
    rng: np.random.Generator,
    cancel_event: threading.Event | None,
    midi_units: tuple[LyricUnit, ...] | None = None,
) -> np.ndarray:
    """Use CREPE's contour to perturb WORLD F0 and warp its spectral envelope."""
    _unused_librosa, pyworld, _torch, _torchcrepe = _optional_dependencies()
    frame_count = len(analysis.f0)
    world_times = np.arange(frame_count, dtype=np.float64) * (_WORLD_FRAME_PERIOD_MS / 1000.0)
    crepe_times = np.arange(len(analysis.crepe_f0), dtype=np.float64) * (_CREPE_HOP_SAMPLES / sample_rate)
    voiced_crepe = np.where(analysis.crepe_periodicity >= 0.2, analysis.crepe_f0, np.nan)
    valid = np.isfinite(voiced_crepe) & (voiced_crepe > 0)
    f0 = analysis.f0.copy()
    if valid.any():
        crepe_on_world = np.interp(world_times, crepe_times[valid], voiced_crepe[valid])
        f0 = np.where(analysis.f0 > 0, crepe_on_world, 0.0)

    line = _smooth_random_line(frame_count, params["pitch_line_cents"], rng, midi_units)
    vibrato = _build_vibrato_cents(f0, params, rng, midi_units)
    cents = params["pitch_shift_cents"] + line + vibrato
    f0 = _apply_f0_cents(f0, cents)
    spectral = _warp_spectral_envelope(analysis.spectral_envelope, params["formant_shift"], sample_rate)
    _check_cancel(cancel_event)
    try:
        output = pyworld.synthesize(f0, spectral, analysis.aperiodicity, sample_rate, _WORLD_FRAME_PERIOD_MS)
    except Exception as exc:
        raise ChoirError("RENDER_FAILED", f"WORLD 重合成失败：{exc}") from exc
    return output.astype(np.float32, copy=False)


def _match_length(audio: np.ndarray, expected_length: int) -> np.ndarray:
    """WORLD frame rounding must not change the copy's timeline length."""
    if len(audio) >= expected_length:
        return audio[:expected_length]
    return np.pad(audio, (0, expected_length - len(audio)))


def _smooth_random_line(
    length: int,
    amplitude_cents: float,
    rng: np.random.Generator,
    midi_units: tuple[LyricUnit, ...] | None = None,
) -> np.ndarray:
    amplitude_cents = abs(amplitude_cents)
    if length < 2 or amplitude_cents < 1e-6:
        return np.zeros(length, dtype=np.float64)
    if midi_units:
        curve = np.zeros(length, dtype=np.float64)
        centers = np.array([
            ((unit.start_s + unit.end_s) / 2) * 1000 / _WORLD_FRAME_PERIOD_MS
            for unit in midi_units
        ], dtype=np.float64)
        centers = np.clip(centers, 0, length - 1)
        values = rng.uniform(-amplitude_cents, amplitude_cents, len(centers))
        left = max(0, int(round(midi_units[0].start_s * 1000 / _WORLD_FRAME_PERIOD_MS)))
        right = min(length, int(round(midi_units[-1].end_s * 1000 / _WORLD_FRAME_PERIOD_MS)))
        if right > left:
            curve[left:right] = np.interp(
                np.arange(left, right), centers, values, left=values[0], right=values[-1]
            )
            # A short fade keeps the MIDI window edge from introducing a step.
            fade = min(6, max(1, (right - left) // 4))
            curve[left:left + fade] *= np.linspace(0.0, 1.0, fade)
            curve[right - fade:right] *= np.linspace(1.0, 0.0, fade)
        return curve
    anchors = rng.uniform(-amplitude_cents, amplitude_cents, max(2, int(np.ceil(length / 40)) + 1))
    line = np.interp(np.arange(length), np.linspace(0, length - 1, len(anchors)), anchors)
    return signal.savgol_filter(line, min(length // 2 * 2 - 1, 31), 2) if length >= 5 else line


def _load_vibrato_notes(
    midi_path: Path | None, audio_duration_s: float, midi_track_index: int | None = None,
) -> tuple[tuple[LyricUnit, ...] | None, str]:
    """Use MIDI note boundaries when available; invalid input falls back safely."""
    if midi_path is None:
        return None, "voiced_segments"
    try:
        return tuple(parse_midi_notes(
            midi_path, audio_duration_s, midi_track_index=midi_track_index,
        )), "midi_notes"
    except (ChoirError, OSError, ValueError):
        return None, "voiced_segments"


def _build_vibrato_cents(
    f0: np.ndarray,
    params: dict,
    rng: np.random.Generator,
    midi_units: tuple[LyricUnit, ...] | None,
) -> np.ndarray:
    """Build a per-frame vibrato curve without changing the global transpose."""
    curve = np.zeros(len(f0), dtype=np.float64)
    if not len(f0) or params["vibrato_depth_cents"] <= 0:
        return curve
    if midi_units:
        for unit in midi_units:
            left = max(0, int(round(unit.start_s * 1000 / _WORLD_FRAME_PERIOD_MS)))
            right = min(len(curve), int(round(unit.end_s * 1000 / _WORLD_FRAME_PERIOD_MS)))
            _add_vibrato_window(curve, left, right, params, rng)
    else:
        _add_fallback_vibrato(curve, f0, params, rng)
    return curve


def _add_fallback_vibrato(
    curve: np.ndarray, f0: np.ndarray, params: dict, rng: np.random.Generator,
) -> None:
    """Split contiguous voiced frames into natural-sized expression sections."""
    voiced = np.asarray(f0 > 0, dtype=np.int8)
    changes = np.diff(np.pad(voiced, (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    min_frames = max(1, int(round(0.35 * 1000 / _WORLD_FRAME_PERIOD_MS)))
    for start, stop in zip(starts, stops):
        left = int(start)
        while left < stop:
            section = int(round(rng.uniform(0.8, 2.0) * 1000 / _WORLD_FRAME_PERIOD_MS))
            right = min(int(stop), left + max(section, min_frames))
            _add_vibrato_window(curve, left, right, params, rng)
            left = right


def _add_vibrato_window(
    curve: np.ndarray, left: int, right: int, params: dict, rng: np.random.Generator,
) -> None:
    """Add one delayed, tapered vibrato gesture to a note or voiced section."""
    if rng.random() > params["vibrato_note_probability"]:
        return
    frame_s = _WORLD_FRAME_PERIOD_MS / 1000.0
    minimum = max(1, int(round(0.22 / frame_s)))
    if right - left < minimum:
        return
    delay = min(int(round(rng.uniform(0.08, 0.20) / frame_s)), (right - left) // 4)
    left += delay
    length = right - left
    if length < minimum:
        return
    depth = params["vibrato_depth_cents"] * rng.uniform(0.2, 1.0)
    rate = params["vibrato_rate_hz"] * rng.uniform(0.85, 1.15)
    phase = rng.uniform(0.0, 2 * np.pi)
    frames = np.arange(length, dtype=np.float64)
    envelope = np.ones(length, dtype=np.float64)
    taper = min(max(1, int(round(0.10 / frame_s))), length // 3)
    if taper:
        envelope[:taper] = np.sin(np.linspace(0.0, np.pi / 2, taper, endpoint=False)) ** 2
        envelope[-taper:] = np.cos(np.linspace(0.0, np.pi / 2, taper, endpoint=True)) ** 2
    curve[left:right] += depth * envelope * np.sin(2 * np.pi * rate * frames * frame_s + phase)


def _apply_f0_cents(f0: np.ndarray, cents: np.ndarray) -> np.ndarray:
    """Prefer CUDA for the vector pitch conversion, retaining a NumPy fallback."""
    gpu_result = apply_pitch_cents(f0, cents)
    if gpu_result is not None:
        return np.asarray(gpu_result, dtype=np.float64)
    return np.where(f0 > 0, f0 * np.exp2(cents / 1200.0), 0.0)


def _warp_spectral_envelope(spectral: np.ndarray, shift_ratio: float, sample_rate: int) -> np.ndarray:
    if abs(shift_ratio) < 1e-8:
        return spectral.copy()
    bins = spectral.shape[1]
    frequencies = np.linspace(0.0, sample_rate / 2, bins)
    source_frequencies = np.clip(frequencies / (1.0 + shift_ratio), 0.0, sample_rate / 2)
    warped = np.empty_like(spectral)
    # The interpolation positions are identical for every WORLD frame. Process
    # rows in bounded batches to remove the per-frame Python loop without
    # allocating a copy of the entire multi-minute spectral envelope at once.
    positions = source_frequencies / (sample_rate / 2) * (bins - 1)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, bins - 1)
    fraction = (positions - lower).astype(spectral.dtype, copy=False)
    for start in range(0, len(spectral), 256):
        stop = min(start + 256, len(spectral))
        rows = spectral[start:stop]
        warped[start:stop] = rows[:, lower] * (1.0 - fraction) + rows[:, upper] * fraction
    return warped


def _apply_vowel_energy_gain(audio: np.ndarray, sample_rate: int, gain_db: float) -> np.ndarray:
    """Apply a gain shape at the rise into the strongest RMS energy region."""
    if abs(gain_db) < 0.01 or not len(audio):
        return audio
    librosa, _pyworld, _torch, _torchcrepe = _optional_dependencies()
    hop_length = 512
    rms = librosa.feature.rms(y=audio.astype(np.float32), frame_length=2048, hop_length=hop_length)[0]
    if not len(rms) or float(np.max(rms)) < 1e-8:
        return audio
    peak_frame = int(np.argmax(rms))
    onset_frame = peak_frame
    threshold = float(rms[peak_frame]) * 0.3
    while onset_frame > 0 and rms[onset_frame - 1] > threshold:
        onset_frame -= 1
    left = min(len(audio), onset_frame * hop_length)
    right = min(len(audio), left + max(1, int(sample_rate * 0.12)))
    gain = np.ones(len(audio), dtype=np.float32)
    shape = np.sin(np.linspace(0.0, np.pi, right - left, endpoint=False, dtype=np.float32))
    gain[left:right] += shape.astype(np.float32) * (np.float32(10 ** (gain_db / 20.0)) - 1.0)
    return audio * gain


def _apply_dynamic_variation(
    audio: np.ndarray, sample_rate: int, amplitude_db: float, rng: np.random.Generator
) -> np.ndarray:
    if abs(amplitude_db) < 0.01 or not len(audio):
        return audio
    points = max(2, int(np.ceil(len(audio) / (sample_rate * 0.4))) + 1)
    db = rng.uniform(-abs(amplitude_db), abs(amplitude_db), points)
    envelope = np.interp(np.arange(len(audio)), np.linspace(0, len(audio) - 1, points), db)
    return (audio * np.power(10.0, envelope / 20.0)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# (1) Formant shift — multi-band EQ with frequency-shifted centre frequencies
# ---------------------------------------------------------------------------

# Nominal formant centre frequencies for an average adult voice (Hz).
# We use four peaking filters.  A positive *shift_ratio* moves each centre
# frequency slightly higher (→ brighter / "smaller" timbre); negative → darker.
_FORMANT_BANDS: list[tuple[float, float]] = [
    (500.0, 1.2),     # F1 region — centre 500 Hz, Q ≈ 1.2
    (1500.0, 1.5),    # F2 region
    (2500.0, 2.0),    # F3 region
    (3700.0, 2.5),    # F4 region
]

# Per-band gain magnitudes (dB) when shifting.  Gains alternate in sign to
# preserve overall spectral balance while changing timbre colouration.
_FORMANT_GAIN_DB = 1.8   # max gain applied per band for a 100 % shift


def _formant_shift_eq(audio: np.ndarray, sr: int, shift_ratio: float) -> np.ndarray:
    """Shift formant perception using frequency-warped peaking EQ bands.

    This replaces the per-frame LPC pole-warping approach.  For micro-shifts
    (±5 %) the perceptual result is equivalent while running orders of
    magnitude faster and with guaranteed numerical stability.
    """
    nyq = sr / 2
    sos_all = []

    for f0, Q in _FORMANT_BANDS:
        # Warp the centre frequency
        f_shifted = f0 * (1.0 + shift_ratio)
        f_norm = max(0.001, min(0.999, f_shifted / nyq))

        # Gain: positive shift → boost upper formants, cut lower ones
        if f0 < 800:
            gain_db = -_FORMANT_GAIN_DB * shift_ratio / 0.05
        elif f0 < 2000:
            gain_db = _FORMANT_GAIN_DB * shift_ratio / 0.05 * 0.5
        else:
            gain_db = _FORMANT_GAIN_DB * shift_ratio / 0.05

        sos = _peaking_eq_sos(f_norm, gain_db, Q)
        sos_all.append(sos)

    if not sos_all:
        return audio

    sos = np.vstack(sos_all)
    return _apply_sos_filter(sos, audio).astype(audio.dtype, copy=False)


# ---------------------------------------------------------------------------
# (2) Pitch micro-shift — resample + time-stretch
# ---------------------------------------------------------------------------


def _pitch_shift_cents(audio: np.ndarray, cents: float) -> np.ndarray:
    """Shift pitch by *cents* (100 cents = 1 semitone) preserving duration.

    Resamples by ratio = 2^(cents/1200), then linearly stretches back
    to original length.  For micro-adjustments (≤ ±20 cents) the artefacts
    are negligible.
    """
    ratio = 2.0 ** (cents / 1200.0)
    n = len(audio)

    n_new = max(1, int(round(n / ratio)))
    indices = np.linspace(0, n - 1, n_new)
    resampled = np.interp(indices, np.arange(n, dtype=np.float64), audio)

    result = np.interp(
        np.linspace(0, n_new - 1, n),
        np.arange(n_new, dtype=np.float64),
        resampled,
    )
    return result.astype(audio.dtype, copy=False)


# ---------------------------------------------------------------------------
# (3) Multi-band EQ — cascade of biquad (second-order sections)
# ---------------------------------------------------------------------------


def _apply_eq(
    audio: np.ndarray, sr: int, mid_db: float, high_db: float
) -> np.ndarray:
    """Apply mid-band (800–3000 Hz) and high-shelf (>4000 Hz) EQ."""
    nyq = sr / 2
    sos_list = []

    # Mid peaking filter: centre ~1500 Hz, Q ≈ 1.0
    if abs(mid_db) > 0.05:
        sos_list.append(_peaking_eq_sos(1500 / nyq, mid_db, Q=1.0))

    # High shelf: corner ~4000 Hz
    if abs(high_db) > 0.05:
        sos_list.append(_high_shelf_sos(4000 / nyq, high_db, Q=1.2))

    if not sos_list:
        return audio

    sos = np.vstack(sos_list)
    return _apply_sos_filter(sos, audio).astype(audio.dtype, copy=False)


def _apply_sos_filter(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Apply small EQ cascades efficiently without changing short-file behavior."""
    if len(audio) < _FFT_FILTER_THRESHOLD_SAMPLES:
        return signal.sosfilt(sos, audio)

    # The peaking and shelf filters used here decay well before 8,192 samples.
    # Convolving their impulse response in the frequency domain avoids a long,
    # single-core IIR pass for multi-minute source recordings.
    dtype = audio.dtype if np.issubdtype(audio.dtype, np.floating) else np.float32
    sos = sos.astype(dtype, copy=False)
    impulse = np.zeros(_FFT_FILTER_IMPULSE_SAMPLES, dtype=dtype)
    impulse[0] = 1.0
    response = signal.sosfilt(sos, impulse)
    if len(audio) >= _CUDA_FFT_THRESHOLD_SAMPLES:
        gpu_result = cuda_fft_convolve(audio, response)
        if gpu_result is not None:
            return gpu_result[:len(audio)]
    return signal.fftconvolve(audio, response, mode="full")[:len(audio)]


def _peaking_eq_sos(f0_norm: float, gain_db: float, Q: float = 1.0) -> np.ndarray:
    """Second-order peaking EQ section (Robert Bristow-Johnson formula)."""
    A = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * f0_norm
    alpha = np.sin(omega) / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(omega)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(omega)
    a2 = 1.0 - alpha / A

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _high_shelf_sos(f0_norm: float, gain_db: float, Q: float = 1.2) -> np.ndarray:
    """Second-order high-shelf section."""
    A = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * f0_norm
    alpha = np.sin(omega) / (2.0 * Q)

    b0 = A * ((A + 1.0) + (A - 1.0) * np.cos(omega) + 2.0 * np.sqrt(A) * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * np.cos(omega))
    b2 = A * ((A + 1.0) + (A - 1.0) * np.cos(omega) - 2.0 * np.sqrt(A) * alpha)
    a0 = (A + 1.0) - (A - 1.0) * np.cos(omega) + 2.0 * np.sqrt(A) * alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * np.cos(omega))
    a2 = (A + 1.0) - (A - 1.0) * np.cos(omega) - 2.0 * np.sqrt(A) * alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


# ---------------------------------------------------------------------------
# (4) Breath overlay — band-pass filtered pink noise
# ---------------------------------------------------------------------------


def _mix_breath(
    audio: np.ndarray,
    sr: int,
    mix_ratio: float,
    rng: np.random.Generator,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """Mix band-pass filtered pink noise (2–6 kHz) into non-silent segments."""
    n = len(audio)

    # Pink noise (~1/f spectrum)
    white = rng.standard_normal(n).astype(np.float64)
    _check_cancel(cancel_event)
    pink = _pink_noise(white, cancel_event)

    # Band-pass 2000–6000 Hz (breath / fricative band)
    nyq = sr / 2
    sos_bp = signal.butter(4, [2000 / nyq, 6000 / nyq], btype="band", output="sos")
    filtered = signal.sosfilt(sos_bp, pink)
    _check_cancel(cancel_event)

    # Scale to target mix ratio relative to signal RMS
    rms_signal = float(np.sqrt(np.mean(audio ** 2)))
    rms_noise = float(np.sqrt(np.mean(filtered ** 2)))
    if rms_noise < 1e-15:
        rms_noise = 1.0
    scaled = filtered * (rms_signal * mix_ratio / rms_noise)

    # Gate: only apply breath during non-silent segments
    envelope = _rms_envelope(np.abs(audio), sr, window_ms=30.0)
    _check_cancel(cancel_event)
    threshold = float(np.max(envelope)) * 0.02
    gate = np.where(envelope > threshold, 1.0, 0.0)

    # Smooth gate transitions (5 ms)
    smooth_samples = int(round(0.005 * sr))
    if smooth_samples > 1:
        kernel = np.hanning(smooth_samples * 2)
        kernel = kernel / kernel.sum()
        if len(gate) >= _CUDA_FFT_THRESHOLD_SAMPLES:
            gpu_gate = cuda_fft_convolve(gate, kernel)
            if gpu_gate is not None:
                start = (len(kernel) - 1) // 2
                gate = gpu_gate[start:start + len(gate)]
            else:
                gate = signal.fftconvolve(gate, kernel, mode="same")
        else:
            gate = signal.fftconvolve(gate, kernel, mode="same")

    _check_cancel(cancel_event)
    return (audio + scaled * gate).astype(audio.dtype, copy=False)


def _pink_noise(
    white: np.ndarray, cancel_event: threading.Event | None = None
) -> np.ndarray:
    """Generate lightweight pink-tilted noise without octave-sized buffers."""
    _check_cancel(cancel_event)
    # The former Voss approximation allocated and repeated one full-size array
    # per octave.  This single-pole filter is O(n), keeps a gentle low-frequency
    # tilt, and is followed by the dedicated 2-6 kHz breath band-pass anyway.
    pink = signal.lfilter([0.05], [1.0, -0.95], white)
    _check_cancel(cancel_event)
    rms = float(np.sqrt(np.mean(pink ** 2)))
    if rms > 1e-15:
        pink /= rms
    return pink


def _rms_envelope(signal_abs: np.ndarray, sr: int, window_ms: float = 30.0) -> np.ndarray:
    """Smoothed RMS envelope of the absolute signal."""
    window = int(round(window_ms * sr / 1000))
    if window < 1:
        window = 1
    # This is exactly np.convolve(signal_abs, ones(window) / window,
    # mode="same"), but calculated as a zero-padded moving sum in O(n).
    # A 30 ms window at 48 kHz has 1,440 taps, so direct convolution is far
    # too expensive for multi-minute source files.
    left_padding = window // 2
    right_padding = window - 1 - left_padding
    padded = np.pad(signal_abs, (left_padding, right_padding))
    cumulative = np.empty(len(padded) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(padded, dtype=np.float64, out=cumulative[1:])
    return (cumulative[window:] - cumulative[:-window]) / window


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _find_available_path(output_dir: Path, stem: str, start_index: int) -> Path:
    """Find an unused file name like ``stem_副本N.wav``."""
    idx = start_index
    while True:
        candidate = output_dir / f"{stem}_副本{idx}.wav"
        if not candidate.exists():
            return candidate
        idx += 1


def _write_wav(path: Path, data: np.ndarray) -> None:
    """Write 48 kHz / 32-bit float / mono WAV."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), 48000, subtype="FLOAT")


def _write_meta(
    path: Path, copy_index: int, seed: int, preset_level: int, params: dict
) -> None:
    """Write differentiation metadata JSON alongside the WAV."""
    data = {
        "copy_index": copy_index,
        "seed": seed,
        "preset_level": preset_level,
        "engine": "crepe_world_librosa_v2",
        "params": params,
    }
    try:
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
        )
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(temp_name, path)
    except Exception:
        pass  # metadata is optional; don't fail the whole operation
