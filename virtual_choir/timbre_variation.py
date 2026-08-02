"""Preset-based vocal differentiation for virtual choir copies.

Each copy uses the same user-selected preset but a distinct deterministic
seed. CREPE supplies the pitch contour and OpenVPI PC-NSF-HiFiGAN renders it
from the original Mel spectrum. Librosa locates a high-energy vowel-like
region for the remaining post-processing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import unicodedata
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
from .timbre_vocoder import (
    MODEL_HOP_SAMPLES, VocoderAnalysis, analyze_voice as analyze_vocoder_voice,
    synthesize as synthesize_vocoder,
)

_FFT_FILTER_THRESHOLD_SAMPLES = 480_000
_FFT_FILTER_IMPULSE_SAMPLES = 8_192
_CUDA_FFT_THRESHOLD_SAMPLES = 1_000_000
_F0_FRAME_PERIOD_MS = MODEL_HOP_SAMPLES * 1000.0 / 44_100
_COPY_SUFFIX = "\u526f\u672c"
JITTER_MIN_INTERVAL_MS = 80.0
JITTER_MAX_INTERVAL_MS = 180.0

_PINYIN_INITIALS = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
)
_INITIAL_CLASS = {
    "b": "stop", "p": "stop", "d": "stop", "t": "stop", "g": "stop", "k": "stop",
    "j": "affricate", "q": "affricate", "zh": "affricate", "ch": "affricate", "z": "affricate", "c": "affricate",
    "f": "fricative", "h": "fricative", "x": "fricative", "sh": "fricative", "s": "fricative", "r": "fricative",
    "m": "sonorant", "n": "sonorant", "l": "sonorant",
}
_MAX_CONSONANT_MS = {
    "zero": 15.0,
    "stop": 55.0,
    "affricate": 80.0,
    "fricative": 85.0,
    "sonorant": 55.0,
}
_ARTICULATION_PROFILES = {
    "popular": {
        "consonant_gain_db": 0.7,
        "consonant_high_db": 0.8,
        "vowel_entry_db": 0.2,
        "tail_gain_db": -0.2,
        "vowel_window_ms": 70.0,
    },
    "bel_canto": {
        "consonant_gain_db": -0.9,
        "consonant_high_db": -1.0,
        "vowel_entry_db": 0.8,
        "tail_gain_db": 0.5,
        "vowel_window_ms": 115.0,
    },
    "child": {
        "consonant_gain_db": 1.5,
        "consonant_high_db": 1.8,
        "vowel_entry_db": 0.4,
        "tail_gain_db": -0.6,
        "vowel_window_ms": 55.0,
    },
}


@dataclass(frozen=True)
class ArticulationRegion:
    """Detected consonant-to-vowel boundary for one lyric-bearing MIDI note."""

    note_index: int
    lyric: str
    pinyin: str
    initial: str
    initial_class: str
    final_class: str
    consonant_start_s: float
    vowel_start_s: float
    note_end_s: float


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
        raise ChoirError("PROJECT_SCHEMA_ERROR", "??????? 1-64 ??")

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
    # CREPE/Mel analysis is shared. The generator itself is serialized because
    # the CUDA model is shared by all copy jobs.
    if progress:
        progress(2, "正在分析音高与频谱…")
    analysis = _analyze_voice(source, 48000, cancel_event)
    analysis_cache: dict[str, object] = {"vocoder": analysis}
    if midi_units and config.articulation_strength > 0.0:
        analysis_cache["articulation"] = _build_articulation_regions(
            source, 48000, analysis.f0, midi_units, cancel_event,
        )
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
                        f"正在生成副本 {completed}/{copy_count}…",
                    )
    except ChoirError:
        _remove_generated_files(written)
        raise
    except Exception as exc:
        _remove_generated_files(written)
        raise ChoirError("RENDER_FAILED", f"宸紓鍖栧鐞嗗け璐ワ細{exc}") from exc

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
    analysis_cache: dict[str, object],
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
        "voice_style": config.voice_style,
        "articulation_strength": round(float(config.articulation_strength), 3),
        "formant_shift": round(_uniform(config.formant_shift_range), 4),
        "pitch_shift_cents": round(_uniform(config.pitch_shift_cents_range), 2),
        "pitch_line_cents": round(_uniform(config.pitch_line_cents_range), 2),
        "pitch_redraw_mix": round(float(config.pitch_redraw_mix), 3),
        "jitter_cents": round(_uniform(config.jitter_cents_range), 3),
        "vowel_onset_db": round(_uniform(config.vowel_onset_db_range), 2),
        "dynamic_db": round(_uniform(config.dynamic_db_range), 2),
        "eq_mid_db": round(_uniform(config.eq_mid_db_range), 2),
        "eq_high_db": round(_uniform(config.eq_high_db_range), 2),
        "breath_mix": round(_uniform(config.breath_mix_range), 4),
        "vibrato_depth_cents": round(_uniform(config.vibrato_depth_cents_range), 2),
        "vibrato_rate_hz": round(_uniform(config.vibrato_rate_hz_range), 3),
        "vibrato_depth_cents_range": list(config.vibrato_depth_cents_range),
        "vibrato_rate_hz_range": list(config.vibrato_rate_hz_range),
        "vibrato_note_probability": round(float(config.vibrato_note_probability), 3),
        "vibrato_activation_probabilities": list(config.vibrato_activation_probabilities),
        "onset_scoop_depth_cents_range": list(config.onset_scoop_depth_cents_range),
        "onset_scoop_duration_ms_range": list(config.onset_scoop_duration_ms_range),
        "onset_scoop_note_probability": round(float(config.onset_scoop_note_probability), 3),
    }


# ---------------------------------------------------------------------------
# Timbre variation pipeline - each step is individually guarded so one
# failure doesn't lose the whole copy.
# ---------------------------------------------------------------------------


def _apply_timbre_variation(
    source: np.ndarray,
    params: dict,
    sample_rate: int,
    rng: np.random.Generator,
    analysis_cache: dict[str, object] | None = None,
    cancel_event: threading.Event | None = None,
    midi_units: tuple[LyricUnit, ...] | None = None,
) -> np.ndarray:
    """Apply PC-NSF-HiFiGAN rendering followed by colour and dynamics."""
    _check_cancel(cancel_event)
    if analysis_cache is not None and "vocoder" in analysis_cache:
        analysis = analysis_cache["vocoder"]
    else:
        analysis = _analyze_voice(source, sample_rate, cancel_event)
        if analysis_cache is not None:
            analysis_cache["vocoder"] = analysis

    audio = _resynthesize_pc_nsf_hifigan(
        analysis, params, sample_rate, rng, cancel_event, midi_units=midi_units,
    )
    audio = _match_length(audio, len(source))
    _check_cancel(cancel_event)

    regions = analysis_cache.get("articulation") if analysis_cache is not None else None
    if params.get("articulation_strength", 0.0) > 0.0 and isinstance(regions, tuple):
        try:
            audio, decisions = _apply_articulation_variation(
                audio, sample_rate, regions, params, rng, cancel_event,
            )
            params["articulation_decisions"] = decisions
        except ChoirError:
            raise
        except Exception:
            params["articulation_decisions"] = [{"mode": "processing_skipped"}]

    if abs(params["formant_shift"]) > 1e-6:
        audio = _formant_shift_eq(audio, sample_rate, params["formant_shift"])

    # Preserve the existing post-processing controls after neural synthesis.
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

    # Breath remains post-vocoder so it stays independently controllable.
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


def _analyze_voice(
    source: np.ndarray, sample_rate: int, cancel_event: threading.Event | None,
) -> VocoderAnalysis:
    """Delegate heavyweight feature extraction to the vocoder boundary."""
    if sample_rate != 48_000:
        raise ChoirError("RENDER_FAILED", "???????? 48 kHz ??")
    return analyze_vocoder_voice(source, cancel_event)


def _resynthesize_pc_nsf_hifigan(
    analysis: VocoderAnalysis,
    params: dict,
    sample_rate: int,
    rng: np.random.Generator,
    cancel_event: threading.Event | None,
    midi_units: tuple[LyricUnit, ...] | None = None,
) -> np.ndarray:
    """Build the external F0 line and delegate waveform rendering."""
    f0 = analysis.f0.copy()
    if midi_units:
        # MIDI supplies note boundaries for gesture selection only.  It must not
        # become an absolute pitch target: a valid MIDI can be octave-shifted
        # or offset from the recorded take.
        cents = _build_midi_pitch_transform_cents(f0, params, rng, midi_units)
    else:
        line = _smooth_random_line(len(f0), params["pitch_line_cents"], rng)
        vibrato = _build_vibrato_cents(f0, params, rng, None)
        jitter = _build_jitter_line(len(f0), params.get("jitter_cents", 0.0), rng)
        cents = params["pitch_shift_cents"] + line + vibrato + jitter
    return synthesize_vocoder(analysis, _apply_f0_cents(f0, cents), cancel_event)


def _match_length(audio: np.ndarray, expected_length: int) -> np.ndarray:
    """Neural framing and sample-rate conversion must keep the timeline."""
    if len(audio) >= expected_length:
        return audio[:expected_length]
    return np.pad(audio, (0, expected_length - len(audio)))


def _build_articulation_regions(
    source: np.ndarray,
    sample_rate: int,
    f0: np.ndarray,
    midi_units: tuple[LyricUnit, ...],
    cancel_event: threading.Event | None,
) -> tuple[ArticulationRegion, ...]:
    """Locate the consonant-to-vowel boundary around lyric-bearing notes.

    MIDI timing is only an anchor.  In vocal synthesis exports, an initial
    consonant can begin shortly before the note timestamp, so the boundary is
    located from the source waveform's energy, spectral balance and voiced F0.
    """
    if not len(source) or not midi_units:
        return ()
    times, energy, high_ratio, flux, voiced = _articulation_features(source, sample_rate, f0)
    if not len(times):
        return ()

    regions: list[ArticulationRegion] = []
    previous_end = 0.0
    for unit in midi_units:
        _check_cancel(cancel_event)
        pinyin = _lyric_to_pinyin(unit.lyric)
        if pinyin is None:
            previous_end = max(previous_end, unit.end_s)
            continue
        initial = _pinyin_initial(pinyin)
        initial_class = _INITIAL_CLASS.get(initial, "zero")
        vowel_start = _detect_vowel_start(
            times, energy, high_ratio, flux, voiced,
            note_start_s=unit.start_s,
            note_end_s=unit.end_s,
            previous_note_end_s=previous_end,
            initial_class=initial_class,
        )
        maximum = _MAX_CONSONANT_MS[initial_class] / 1000.0
        consonant_start = vowel_start if initial_class == "zero" else max(
            previous_end, unit.start_s - maximum, vowel_start - maximum,
        )
        if vowel_start <= consonant_start + 0.003:
            consonant_start = vowel_start
        regions.append(ArticulationRegion(
            note_index=unit.index,
            lyric=unit.lyric or "",
            pinyin=pinyin,
            initial=initial,
            initial_class=initial_class,
            final_class=_pinyin_final_class(pinyin, initial),
            consonant_start_s=round(float(consonant_start), 5),
            vowel_start_s=round(float(vowel_start), 5),
            note_end_s=round(float(unit.end_s), 5),
        ))
        previous_end = max(previous_end, unit.end_s)
    return tuple(regions)


def _articulation_features(
    source: np.ndarray, sample_rate: int, f0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build lightweight 5 ms features shared by every generated copy."""
    hop = max(1, int(round(sample_rate * 0.005)))
    frame = max(256, int(round(sample_rate * 0.025)))
    overlap = max(0, frame - hop)
    _freqs, times, spectrum = signal.stft(
        source.astype(np.float64), fs=sample_rate, window="hann", nperseg=frame,
        noverlap=overlap, boundary=None, padded=False,
    )
    if spectrum.ndim != 2 or not spectrum.shape[1]:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty, np.empty(0, dtype=bool)
    magnitude = np.abs(spectrum)
    power = magnitude * magnitude
    energy = np.sqrt(np.mean(power, axis=0))
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    speech_band = (frequencies >= 80.0) & (frequencies <= min(8_000.0, sample_rate / 2 - 1.0))
    high_band = (frequencies >= 2_000.0) & (frequencies <= min(8_000.0, sample_rate / 2 - 1.0))
    total = np.sum(power[speech_band], axis=0) + 1e-12
    high_ratio = np.sum(power[high_band], axis=0) / total
    flux = np.zeros(len(times), dtype=np.float64)
    if magnitude.shape[1] > 1:
        flux[1:] = np.sum(np.maximum(magnitude[:, 1:] - magnitude[:, :-1], 0.0), axis=0)
        scale = float(np.percentile(flux[1:], 95)) if len(flux) > 1 else 0.0
        if scale > 1e-12:
            flux /= scale
    if len(f0):
        f0_times = np.arange(len(f0), dtype=np.float64) * _F0_FRAME_PERIOD_MS / 1000.0
        voiced = np.interp(times, f0_times, (f0 > 0).astype(np.float64), left=0.0, right=0.0) >= 0.5
    else:
        voiced = np.zeros(len(times), dtype=bool)
    return times, energy, high_ratio, flux, voiced


def _detect_vowel_start(
    times: np.ndarray,
    energy: np.ndarray,
    high_ratio: np.ndarray,
    flux: np.ndarray,
    voiced: np.ndarray,
    *,
    note_start_s: float,
    note_end_s: float,
    previous_note_end_s: float,
    initial_class: str,
) -> float:
    maximum = _MAX_CONSONANT_MS[initial_class] / 1000.0
    left = max(previous_note_end_s, note_start_s - maximum)
    right = min(note_end_s, note_start_s + min(0.18, max(0.05, (note_end_s - note_start_s) * 0.55)))
    candidates = np.flatnonzero((times >= left) & (times <= right))
    if not len(candidates):
        return float(min(max(note_start_s, 0.0), note_end_s))

    local_energy = energy[candidates]
    floor = float(np.percentile(local_energy, 15))
    peak = float(np.percentile(local_energy, 90))
    energy_threshold = floor + (peak - floor) * 0.22
    # A vowel has periodic energy and normally less high-frequency dominance
    # than a preceding fricative/affricate.  Require two adjacent frames to
    # avoid reacting to one sharp consonant transient.
    for position, frame_index in enumerate(candidates[:-1]):
        next_index = candidates[position + 1]
        sustained = energy[frame_index] >= energy_threshold and energy[next_index] >= energy_threshold
        periodic = voiced[frame_index] and voiced[next_index]
        vowelish = high_ratio[frame_index] < 0.68 or flux[frame_index] < 0.65
        if sustained and periodic and vowelish:
            return float(times[frame_index])

    voiced_candidates = candidates[(voiced[candidates]) & (energy[candidates] >= energy_threshold)]
    if len(voiced_candidates):
        return float(times[voiced_candidates[0]])
    energetic = candidates[energy[candidates] >= energy_threshold]
    if len(energetic):
        return float(times[energetic[0]])
    return float(min(max(note_start_s, left), right))


def _lyric_to_pinyin(lyric: str | None) -> str | None:
    if not lyric:
        return None
    text = lyric.strip().lower()
    ascii_tokens = re.findall(r"[a-zv\u00fc\u00fc]+[0-9]*", text)
    if ascii_tokens:
        return _normalise_pinyin(ascii_tokens[0])
    try:
        from pypinyin import Style, lazy_pinyin

        syllables = lazy_pinyin(text, style=Style.NORMAL, strict=False)
    except (ImportError, ValueError):
        return None
    return _normalise_pinyin(syllables[0]) if syllables else None


def _normalise_pinyin(value: str) -> str | None:
    decomposed = unicodedata.normalize("NFD", value.lower())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    token = re.sub(r"[^a-zv\u00fc]", "", plain).replace("\u00fc", "v")
    return token or None


def _pinyin_initial(pinyin: str) -> str:
    return next((item for item in _PINYIN_INITIALS if pinyin.startswith(item)), "")


def _pinyin_final_class(pinyin: str, initial: str) -> str:
    final = pinyin[len(initial):]
    if final.endswith(("ng", "n", "m")):
        return "nasal"
    return "vowel"


def _apply_articulation_variation(
    audio: np.ndarray,
    sample_rate: int,
    regions: tuple[ArticulationRegion, ...],
    params: dict,
    rng: np.random.Generator,
    cancel_event: threading.Event | None,
) -> tuple[np.ndarray, list[dict]]:
    """Apply style-aware, local articulation changes without altering lyrics."""
    strength = float(params.get("articulation_strength", 0.0))
    profile = _ARTICULATION_PROFILES.get(str(params.get("voice_style", "popular")), _ARTICULATION_PROFILES["popular"])
    if strength <= 0.0 or not regions:
        return audio, []

    result = audio.astype(np.float32, copy=True)
    decisions: list[dict] = []
    for region in regions:
        _check_cancel(cancel_event)
        variation = float(rng.uniform(0.75, 1.25)) * strength
        consonant_left = int(round(region.consonant_start_s * sample_rate))
        consonant_right = int(round(region.vowel_start_s * sample_rate))
        consonant_gain = profile["consonant_gain_db"] * variation
        high_gain = profile["consonant_high_db"] * variation
        if consonant_right - consonant_left >= max(8, int(sample_rate * 0.008)):
            _apply_local_high_shelf(result, consonant_left, consonant_right, sample_rate, high_gain)
            _apply_local_gain(result, consonant_left, consonant_right, consonant_gain, mode="taper")

        vowel_left = consonant_right
        vowel_right = min(len(result), vowel_left + int(round(profile["vowel_window_ms"] * sample_rate / 1000.0)))
        vowel_gain = profile["vowel_entry_db"] * variation
        if vowel_right - vowel_left >= 8:
            _apply_local_gain(result, vowel_left, vowel_right, vowel_gain, mode="decay")

        tail_right = min(len(result), int(round(region.note_end_s * sample_rate)))
        tail_window = int(round((70.0 if region.final_class == "nasal" else 45.0) * sample_rate / 1000.0))
        tail_left = max(vowel_left, tail_right - tail_window)
        tail_gain = profile["tail_gain_db"] * variation
        if tail_right - tail_left >= 8:
            _apply_local_gain(result, tail_left, tail_right, tail_gain, mode="rise")

        decisions.append({
            "note_index": region.note_index,
            "lyric": region.lyric,
            "pinyin": region.pinyin,
            "initial": region.initial,
            "initial_class": region.initial_class,
            "final_class": region.final_class,
            "consonant_start_s": region.consonant_start_s,
            "vowel_start_s": region.vowel_start_s,
            "consonant_gain_db": round(float(consonant_gain), 3),
            "consonant_high_db": round(float(high_gain), 3),
            "vowel_entry_db": round(float(vowel_gain), 3),
            "tail_gain_db": round(float(tail_gain), 3),
        })
    return result, decisions


def _apply_local_high_shelf(
    audio: np.ndarray, left: int, right: int, sample_rate: int, gain_db: float,
) -> None:
    if abs(gain_db) < 0.02 or right <= left:
        return
    segment = audio[left:right].copy()
    filtered = _apply_eq(segment, sample_rate, 0.0, gain_db)
    mask = _local_mask(len(segment), mode="taper")
    audio[left:right] = segment * (1.0 - mask) + filtered * mask


def _apply_local_gain(
    audio: np.ndarray, left: int, right: int, gain_db: float, *, mode: str,
) -> None:
    if abs(gain_db) < 0.02 or right <= left:
        return
    mask = _local_mask(right - left, mode=mode)
    ratio = np.float32(10.0 ** (gain_db / 20.0))
    audio[left:right] *= 1.0 + mask * (ratio - 1.0)


def _local_mask(length: int, *, mode: str) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    phase = np.linspace(0.0, np.pi / 2, length, endpoint=True, dtype=np.float32)
    if mode == "decay":
        return np.cos(phase) ** 2
    if mode == "rise":
        return np.sin(phase) ** 2
    fade = min(max(1, length // 4), int(round(0.006 * 48_000)))
    mask = np.ones(length, dtype=np.float32)
    mask[:fade] = np.sin(np.linspace(0.0, np.pi / 2, fade, endpoint=False, dtype=np.float32)) ** 2
    mask[-fade:] = np.cos(np.linspace(0.0, np.pi / 2, fade, endpoint=True, dtype=np.float32)) ** 2
    return mask


def _build_midi_pitch_transform_cents(
    f0: np.ndarray,
    params: dict,
    rng: np.random.Generator,
    midi_units: tuple[LyricUnit, ...],
) -> np.ndarray:
    """Redraw the CREPE contour within MIDI note boundaries.

    MIDI gives timing and note-local gesture boundaries, never an absolute F0
    target.  This preserves a singer's recorded register even when the MIDI
    arrangement is transposed or octave-displaced from the audio take.
    """
    result = np.full(len(f0), float(params.get("pitch_shift_cents", 0.0)), dtype=np.float64)
    redraw_mix = float(params.get("pitch_redraw_mix", 0.0))
    amplitude = abs(float(params.get("pitch_line_cents", 0.0)))
    jitter_amplitude = abs(float(params.get("jitter_cents", 0.0)))
    decisions: list[dict] = []

    for unit in midi_units:
        left = max(0, int(round(unit.start_s * 1000 / _F0_FRAME_PERIOD_MS)))
        right = min(len(f0), int(round(unit.end_s * 1000 / _F0_FRAME_PERIOD_MS)))
        if right <= left:
            continue
        segment = f0[left:right]
        voiced = np.isfinite(segment) & (segment > 0)
        if not voiced.any():
            continue

        # Analyse the source gesture around its own local register rather than
        # the MIDI pitch.  This retains CREPE-guided vibrato classification but
        # cannot accidentally apply an octave-sized correction.
        original_offset = np.zeros(right - left, dtype=np.float64)
        reference_hz = float(np.median(segment[voiced]))
        original_offset[voiced] = 1200.0 * np.log2(segment[voiced] / reference_hz)
        redraw_offset = _build_note_redraw_curve(right - left, amplitude, rng)
        vibrato, decision = _build_crepe_guided_vibrato(
            original_offset, voiced, params, rng,
            duration_s=max(0.0, unit.end_s - unit.start_s),
        )
        onset_scoop, scoop_decision = _build_onset_scoop_cents(
            voiced, params, rng,
        )
        jitter = _build_jitter_line(right - left, jitter_amplitude, rng)
        # This is the same additive, source-F0-relative redraw model used by
        # the standalone listening experiment.  The preset percentage controls
        # how much of the generated contour is applied.
        target_delta = redraw_mix * (redraw_offset + vibrato + jitter)
        target_delta *= _voiced_boundary_envelope(voiced)
        note_result = result[left:right]
        # The scoop is intentionally added after the normal boundary envelope.
        # It needs to start below the local CREPE target at the first voiced
        # frame instead of being faded in from zero like generic F0 edits.
        note_result[voiced] += target_delta[voiced] + onset_scoop[voiced]
        result[left:right] = note_result
        decision.update({
            "midi_pitch": int(unit.pitch),
            "midi_pitch_role": "note_boundary_only",
            "redraw_mix": round(redraw_mix, 3),
            "start_s": round(float(unit.start_s), 4),
            "end_s": round(float(unit.end_s), 4),
            "onset_scoop": scoop_decision,
        })
        decisions.append(decision)

    # Metadata is intentionally compact but records the generated note
    # behaviours so a rendered copy can be inspected after listening.
    params["pitch_redraw_decisions"] = decisions
    params["midi_pitch_role"] = "note_boundaries_only"
    return result


def _build_onset_scoop_cents(
    voiced: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Create a brief low-to-target physiological onset gesture.

    The gesture is anchored at the first actual CREPE-voiced frame, not the
    MIDI event time.  This keeps consonants and unvoiced attacks out of the F0
    edit while using MIDI only for the per-note window.
    """
    curve = np.zeros(len(voiced), dtype=np.float64)
    probability = float(params.get("onset_scoop_note_probability", 0.0))
    depth_range = params.get("onset_scoop_depth_cents_range", (0.0, 0.0))
    duration_range = params.get("onset_scoop_duration_ms_range", (0.0, 0.0))
    depth_low, depth_high = (abs(float(value)) for value in depth_range)
    duration_low, duration_high = (float(value) for value in duration_range)
    decision = {
        "enabled": False,
        "probability": round(probability, 3),
    }
    if probability <= 0.0 or depth_high <= 0.0 or duration_high <= 0.0:
        decision["mode"] = "preset_disabled"
        return curve, decision
    if rng.random() >= probability:
        decision["mode"] = "not_selected"
        return curve, decision

    depth = float(rng.uniform(min(depth_low, depth_high), max(depth_low, depth_high)))
    duration_ms = float(rng.uniform(min(duration_low, duration_high), max(duration_low, duration_high)))

    starts = np.flatnonzero(voiced)
    if not len(starts):
        decision["mode"] = "unvoiced_note"
        return curve, decision
    start = int(starts[0])
    unvoiced_after_start = np.flatnonzero(~voiced[start:])
    voiced_stop = start + int(unvoiced_after_start[0]) if len(unvoiced_after_start) else len(voiced)
    requested_frames = max(2, int(round(duration_ms / _F0_FRAME_PERIOD_MS)) + 1)
    stop = min(voiced_stop, start + requested_frames)
    if stop - start < 2:
        decision["mode"] = "voiced_onset_too_short"
        return curve, decision

    # Start below the note's CREPE-relative target and arrive exactly at it.
    curve[start:stop] = np.linspace(-depth, 0.0, stop - start, dtype=np.float64)
    decision.update({
        "enabled": True,
        "mode": "low_to_target",
        "depth_cents": round(depth, 3),
        "duration_ms": round((stop - start - 1) * _F0_FRAME_PERIOD_MS, 3),
    })
    return curve, decision


def _voiced_boundary_envelope(voiced: np.ndarray) -> np.ndarray:
    """Fade F0 edits around actual voiced-segment boundaries.

    MIDI note boundaries can precede the singer's consonant onset.  CREPE's
    voiced mask is the reliable guard for avoiding F0 changes on those brief
    transients, which are especially audible in neural-vocoder output.
    """
    envelope = np.zeros(len(voiced), dtype=np.float64)
    changes = np.diff(np.pad(np.asarray(voiced, dtype=np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops):
        envelope[start:stop] = 1.0
        fade = min(3, max(1, (stop - start) // 3))
        envelope[start:start + fade] *= np.linspace(0.0, 1.0, fade)
        envelope[stop - fade:stop] *= np.linspace(1.0, 0.0, fade)
    return envelope


def _build_note_redraw_curve(length: int, amplitude_cents: float, rng: np.random.Generator) -> np.ndarray:
    """Create a low-frequency, non-periodic note contour in cents."""
    if length <= 0 or amplitude_cents < 1e-6:
        return np.zeros(length, dtype=np.float64)
    if length < 8:
        return np.zeros(length, dtype=np.float64)
    anchor_count = max(2, min(6, int(np.ceil(length / 40)) + 1))
    anchors = rng.uniform(-amplitude_cents, amplitude_cents, anchor_count)
    curve = np.interp(np.arange(length), np.linspace(0, length - 1, anchor_count), anchors)
    window = min(length // 2 * 2 - 1, 31)
    if window >= 5:
        curve = signal.savgol_filter(curve, window, 2)
    # Avoid a per-note transposition and ease back to the source F0 at note
    # boundaries, where an F0 step is particularly audible to the vocoder.
    curve -= float(np.mean(curve))
    # Preserve consonant onsets and releases from F0 redraw artifacts.
    fade = min(12, max(1, length // 3))
    curve[:fade] *= np.linspace(0.0, 1.0, fade)
    curve[-fade:] *= np.linspace(1.0, 0.0, fade)
    return curve


def _build_jitter_line(length: int, amplitude_cents: float, rng: np.random.Generator) -> np.ndarray:
    """Generate small, aperiodic and short-correlated human pitch jitter."""
    if length <= 1 or amplitude_cents < 1e-6:
        return np.zeros(length, dtype=np.float64)
    # Slower 80-180 ms anchors keep the motion human without turning into FM.
    spacing = max(4, int(round(rng.uniform(
        JITTER_MIN_INTERVAL_MS / 1000.0,
        JITTER_MAX_INTERVAL_MS / 1000.0,
    ) * 1000 / _F0_FRAME_PERIOD_MS)))
    anchors = rng.uniform(-amplitude_cents, amplitude_cents, max(2, int(np.ceil(length / spacing)) + 1))
    curve = np.interp(np.arange(length), np.linspace(0, length - 1, len(anchors)), anchors)
    if length >= 5:
        window = min(length // 2 * 2 - 1, 7)
        if window >= 5:
            curve = signal.savgol_filter(curve, window, 2)
    fade = min(12, max(1, length // 3))
    curve[:fade] *= np.linspace(0.0, 1.0, fade)
    curve[-fade:] *= np.linspace(1.0, 0.0, fade)
    return curve


def _analyze_crepe_vibrato(offset_cents: np.ndarray, voiced: np.ndarray) -> tuple[float, float]:
    """Estimate periodic vibrato depth/rate after removing slow pitch motion."""
    values = offset_cents[voiced]
    if len(values) < 60:  # 300 ms is too short for a reliable vibrato reading.
        return 0.0, 0.0
    window = min(len(values) // 2 * 2 - 1, 101)
    trend = signal.savgol_filter(values, window, 2) if window >= 5 else np.full_like(values, np.mean(values))
    residual = values - trend
    frequencies = np.fft.rfftfreq(len(residual), _F0_FRAME_PERIOD_MS / 1000.0)
    spectrum = np.abs(np.fft.rfft(residual))
    band = (frequencies >= 3.5) & (frequencies <= 8.0)
    if not band.any() or float(np.max(spectrum[band])) < 1e-6:
        return 0.0, 0.0
    rate = float(frequencies[band][np.argmax(spectrum[band])])
    depth = float(np.percentile(np.abs(residual), 90))
    return depth, rate


def _build_crepe_guided_vibrato(
    original_offset: np.ndarray,
    voiced: np.ndarray,
    params: dict,
    rng: np.random.Generator,
    duration_s: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Select a note gesture from CREPE's detected original vibrato."""
    curve = np.zeros(len(original_offset), dtype=np.float64)
    if duration_s is None:
        duration_s = len(curve) * _F0_FRAME_PERIOD_MS / 1000.0
    source_depth, source_rate = _analyze_crepe_vibrato(original_offset, voiced)
    category = 0 if source_depth < 3.0 else 1 if source_depth < 8.0 else 2 if source_depth < 20.0 else 3
    decision = {
        "source_vibrato": ("flat", "light", "natural", "strong")[category],
        "source_depth_cents": round(source_depth, 3),
        "source_rate_hz": round(source_rate, 3),
    }
    if duration_s <= 0.250:
        decision.update({"mode": "short_note_suppressed", "enabled": False})
        return curve, decision

    probabilities = params.get("vibrato_activation_probabilities", (0.7, 0.6, 0.5, 0.4))
    if rng.random() > float(probabilities[category]):
        decision.update({"mode": "not_selected", "enabled": False})
        return curve, decision

    low_depth, high_depth = params["vibrato_depth_cents_range"] if "vibrato_depth_cents_range" in params else (
        max(0.0, params.get("vibrato_depth_cents", 0.0) * 0.6), params.get("vibrato_depth_cents", 0.0),
    )
    low_rate, high_rate = params["vibrato_rate_hz_range"] if "vibrato_rate_hz_range" in params else (
        params.get("vibrato_rate_hz", 5.0), params.get("vibrato_rate_hz", 5.0),
    )
    base_depth = float(rng.uniform(low_depth, high_depth))
    base_rate = float(rng.uniform(low_rate, high_rate))
    if category == 0:
        mode = str(rng.choice(("full", "late", "early_flat"), p=(0.45, 0.40, 0.15)))
        depth, rate = base_depth, base_rate
    else:
        modes = ("stable", "soften", "strengthen", "slower", "faster", "full", "early_flat", "late")
        mode = str(rng.choice(modes, p=(0.18, 0.18, 0.14, 0.10, 0.10, 0.10, 0.10, 0.10)))
        source_rate = source_rate if source_rate > 0 else base_rate
        depth = float(np.clip(source_depth, low_depth, high_depth))
        rate = float(np.clip(source_rate, low_rate, high_rate))
        if mode == "stable":
            depth = 0.0
        elif mode == "soften":
            depth *= float(rng.uniform(0.40, 0.75))
        elif mode == "strengthen":
            depth = min(high_depth, max(base_depth, source_depth * rng.uniform(1.25, 1.65)))
        elif mode == "slower":
            rate = float(np.clip(source_rate * rng.uniform(0.75, 0.90), low_rate, high_rate))
        elif mode == "faster":
            rate = float(np.clip(source_rate * rng.uniform(1.10, 1.30), low_rate, high_rate))

    if depth <= 1e-6:
        decision.update({"mode": mode, "enabled": True, "target_depth_cents": 0.0, "target_rate_hz": round(rate, 3)})
        return curve, decision

    envelope = _vibrato_envelope(len(curve), mode, rng)
    frames = np.arange(len(curve), dtype=np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    curve[voiced] = depth * envelope[voiced] * np.sin(
        2.0 * np.pi * rate * frames[voiced] * (_F0_FRAME_PERIOD_MS / 1000.0) + phase
    )
    decision.update({
        "mode": mode,
        "enabled": True,
        "target_depth_cents": round(float(depth), 3),
        "target_rate_hz": round(float(rate), 3),
    })
    return curve, decision


def _vibrato_envelope(length: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float64)
    if mode == "early_flat":
        start = int(round(length * rng.uniform(0.45, 0.60)))
        stop = min(length, int(round(length * rng.uniform(0.65, 0.85))))
        if stop > start:
            envelope[start:stop] = np.linspace(1.0, 0.0, stop - start)
            envelope[stop:] = 0.0
    elif mode == "late":
        start = int(round(length * rng.uniform(0.15, 0.35)))
        stop = min(length, start + max(1, int(round(length * 0.18))))
        envelope[:start] = 0.0
        if stop > start:
            envelope[start:stop] = np.linspace(0.0, 1.0, stop - start)
    edge = min(10, max(1, length // 8))
    envelope[:edge] *= np.linspace(0.0, 1.0, edge)
    envelope[-edge:] *= np.linspace(1.0, 0.0, edge)
    return envelope


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
            ((unit.start_s + unit.end_s) / 2) * 1000 / _F0_FRAME_PERIOD_MS
            for unit in midi_units
        ], dtype=np.float64)
        centers = np.clip(centers, 0, length - 1)
        values = rng.uniform(-amplitude_cents, amplitude_cents, len(centers))
        left = max(0, int(round(midi_units[0].start_s * 1000 / _F0_FRAME_PERIOD_MS)))
        right = min(length, int(round(midi_units[-1].end_s * 1000 / _F0_FRAME_PERIOD_MS)))
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
        try:
            units = parse_midi_notes(
                midi_path, audio_duration_s, require_lyrics=True,
                midi_track_index=midi_track_index,
            )
        except ChoirError:
            # Keep all existing no-lyric MIDI workflows working.  Lyrics are
            # optional for F0 variation, but preferred for articulation.
            units = parse_midi_notes(
                midi_path, audio_duration_s, midi_track_index=midi_track_index,
            )
        return tuple(units), "midi_notes"
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
            left = max(0, int(round(unit.start_s * 1000 / _F0_FRAME_PERIOD_MS)))
            right = min(len(curve), int(round(unit.end_s * 1000 / _F0_FRAME_PERIOD_MS)))
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
    min_frames = max(1, int(round(0.35 * 1000 / _F0_FRAME_PERIOD_MS)))
    for start, stop in zip(starts, stops):
        left = int(start)
        while left < stop:
            section = int(round(rng.uniform(0.8, 2.0) * 1000 / _F0_FRAME_PERIOD_MS))
            right = min(int(stop), left + max(section, min_frames))
            _add_vibrato_window(curve, left, right, params, rng)
            left = right


def _add_vibrato_window(
    curve: np.ndarray, left: int, right: int, params: dict, rng: np.random.Generator,
) -> None:
    """Add one delayed, tapered vibrato gesture to a note or voiced section."""
    if rng.random() > params["vibrato_note_probability"]:
        return
    frame_s = _F0_FRAME_PERIOD_MS / 1000.0
    # Notes under 250 ms retain the random pitch-line offset but do not receive
    # periodic vibrato, which otherwise reads as an artificial flutter.
    minimum = max(1, int(round(0.25 / frame_s)))
    if right - left <= minimum:
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


def _apply_vowel_energy_gain(audio: np.ndarray, sample_rate: int, gain_db: float) -> np.ndarray:
    """Apply a gain shape at the rise into the strongest RMS energy region."""
    if abs(gain_db) < 0.01 or not len(audio):
        return audio
    try:
        import librosa
    except Exception as exc:
        raise ChoirError("RENDER_DEPENDENCY_MISSING", "音色差异化需要 librosa；请安装 requirements.txt") from exc
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
# (1) Formant shift 鈥?multi-band EQ with frequency-shifted centre frequencies
# ---------------------------------------------------------------------------

# Nominal formant centre frequencies for an average adult voice (Hz).
# We use four peaking filters.  A positive *shift_ratio* moves each centre
# frequency slightly higher (鈫?brighter / "smaller" timbre); negative 鈫?darker.
_FORMANT_BANDS: list[tuple[float, float]] = [
    (500.0, 1.2),     # F1 region 鈥?centre 500 Hz, Q 鈮?1.2
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
    (卤5 %) the perceptual result is equivalent while running orders of
    magnitude faster and with guaranteed numerical stability.
    """
    nyq = sr / 2
    sos_all = []

    for f0, Q in _FORMANT_BANDS:
        # Warp the centre frequency
        f_shifted = f0 * (1.0 + shift_ratio)
        f_norm = max(0.001, min(0.999, f_shifted / nyq))

        # Gain: positive shift 鈫?boost upper formants, cut lower ones
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
# (2) Pitch micro-shift 鈥?resample + time-stretch
# ---------------------------------------------------------------------------


def _pitch_shift_cents(audio: np.ndarray, cents: float) -> np.ndarray:
    """Shift pitch by *cents* (100 cents = 1 semitone) preserving duration.

    Resamples by ratio = 2^(cents/1200), then linearly stretches back
    to original length.  For micro-adjustments (鈮?卤20 cents) the artefacts
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
# (3) Multi-band EQ 鈥?cascade of biquad (second-order sections)
# ---------------------------------------------------------------------------


def _apply_eq(
    audio: np.ndarray, sr: int, mid_db: float, high_db: float
) -> np.ndarray:
    """Apply mid-band (800鈥?000 Hz) and high-shelf (>4000 Hz) EQ."""
    nyq = sr / 2
    sos_list = []

    # Mid peaking filter: centre ~1500 Hz, Q 鈮?1.0
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
# (4) Breath overlay 鈥?band-pass filtered pink noise
# ---------------------------------------------------------------------------


def _mix_breath(
    audio: np.ndarray,
    sr: int,
    mix_ratio: float,
    rng: np.random.Generator,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """Mix band-pass filtered pink noise (2鈥? kHz) into non-silent segments."""
    n = len(audio)

    # Pink noise (~1/f spectrum)
    white = rng.standard_normal(n).astype(np.float64)
    _check_cancel(cancel_event)
    pink = _pink_noise(white, cancel_event)

    # Band-pass 2000鈥?000 Hz (breath / fricative band)
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
    """Find an unused copy filename without depending on console encoding."""
    idx = start_index
    while True:
        candidate = output_dir / f"{stem}_{_COPY_SUFFIX}{idx}.wav"
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
        "engine": "crepe_pc_nsf_hifigan_v1",
        "vocoder_checkpoint_license": "CC BY-NC-SA 4.0",
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
