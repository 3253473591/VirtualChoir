from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import read_source_wav
from .cuda_acceleration import create_linear_sampler
from .errors import ChoirError
from .models import NaturalizationConfig, midi_assignment_for_track

NATURALIZATION_VERSION = "1.3.1"
OFFSET_STDDEV_MS = 5.0
OFFSET_LIMIT_MS = 15.0
TAIL_GAP_THRESHOLD_S = 0.020
SHORT_NOTE_TAIL_OFFSET_THRESHOLD_S = 0.400
TAIL_LEFT_MEAN_MS = -12.0
TAIL_LEFT_STDDEV_MS = 20.0
TAIL_LEFT_RANGE_MS = (-30.0, 0.0)
TAIL_RIGHT_MEAN_MS = 30.0
TAIL_RIGHT_STDDEV_MS = 18.0
TAIL_RIGHT_RANGE_MS = (0.0, 80.0)
# A larger right-hand component makes the two-sided distribution right-skewed.
TAIL_RIGHT_PROBABILITY = 0.65
_INTERPOLATION_CHUNK_SAMPLES = 262_144
_CUDA_INTERPOLATION_THRESHOLD_SAMPLES = 1_000_000
Progress = Callable[[int, str], None]


@dataclass(frozen=True)
class LyricUnit:
    """One first-track MIDI note with optional verbatim lyric metadata."""

    index: int
    pitch: int
    start_s: float
    end_s: float
    lyric: str | None
    lyric_time_s: float | None


@dataclass(frozen=True)
class UnitOffset:
    note_index: int
    pitch: int
    note_start_s: float
    note_end_s: float
    lyric: str | None
    lyric_time_s: float | None
    offset_ms: float
    end_offset_ms: float = 0.0


@dataclass(frozen=True)
class MidiTrackInfo:
    """Inspectable MIDI track metadata for internal-track selection."""

    index: int
    note_count: int
    lowest_pitch: int | None
    highest_pitch: int | None


def resolve_midi_path(project_dir: Path, midi_path: str) -> Path:
    candidate = Path(midi_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (project_dir / candidate).resolve()
        root = project_dir.resolve()
        if resolved != root and root not in resolved.parents:
            raise ChoirError("NATURALIZATION_INPUT_MISSING", "MIDI 路径超出工程目录")
    if not resolved.is_file():
        raise ChoirError("NATURALIZATION_INPUT_MISSING", str(resolved))
    return resolved


def parse_midi_notes(
    path: Path,
    audio_duration_s: float,
    *,
    require_lyrics: bool = False,
    midi_track_index: int | None = None,
) -> list[LyricUnit]:
    """Read one monophonic note timeline, skipping empty tracks by default."""
    try:
        division, tracks = _read_midi_tracks(path.read_bytes())
        parsed_tracks = [_read_track(track) for track in tracks]
    except ChoirError:
        raise
    except (OSError, ValueError, IndexError, struct.error) as exc:
        raise ChoirError("NATURALIZATION_MIDI_INVALID", str(exc)) from exc

    if midi_track_index is not None:
        if not 0 <= midi_track_index < len(parsed_tracks):
            raise ChoirError("NATURALIZATION_MIDI_INVALID", "所选 MIDI 内部音轨不存在")
        selected_index = midi_track_index
    else:
        selected_index = next(
            (index for index, parsed in enumerate(parsed_tracks) if parsed[3]), None
        )
        if selected_index is None:
            raise ChoirError("NATURALIZATION_ALIGNMENT_FAILED", "MIDI 没有音符事件")

    _selected_tempos, lyrics, text_events, notes = parsed_tracks[selected_index]
    tempos = [tempo for parsed in parsed_tracks for tempo in parsed[0]]
    track_label = f"MIDI 第 {selected_index + 1} 轨"
    if not notes:
        raise ChoirError("NATURALIZATION_ALIGNMENT_FAILED", f"{track_label} 没有音符事件")
    if require_lyrics and not lyrics:
        lyrics = text_events
    if require_lyrics and not lyrics:
        # Vocal-synth exports often keep lyrics in a dedicated metadata track
        # while notes are in separate singer tracks. Lyrics are metadata only,
        # so using all text events preserves note timing without rejecting it.
        lyrics = [
            item for parsed in parsed_tracks for item in (parsed[1] or parsed[2])
        ]
    if require_lyrics and not lyrics:
        raise ChoirError("NATURALIZATION_MIDI_INVALID", f"{track_label} 及其他 MIDI 轨均没有歌词事件")

    tempo_map = _tempo_map(tempos)
    lyric_events = sorted(lyrics, key=lambda item: item[0])
    note_events = sorted(notes, key=lambda item: (item[0], item[2], item[1]))
    if any(left[0] == right[0] for left, right in zip(note_events, note_events[1:])):
        raise ChoirError("NATURALIZATION_ALIGNMENT_FAILED", f"{track_label} 包含同时起音，无法作为单声部时间线")

    units: list[LyricUnit] = []
    for start_tick, end_tick, pitch in note_events:
        start_s = _tick_seconds(start_tick, division, tempo_map)
        if start_s >= audio_duration_s:
            continue
        end_s = min(audio_duration_s, _tick_seconds(end_tick, division, tempo_map))
        if end_s <= start_s:
            continue
        matching = next(
            ((tick, text) for tick, text in lyric_events if start_tick <= tick < end_tick),
            None,
        )
        lyric_time_s = None
        lyric = None
        if matching is not None:
            lyric_time_s = _tick_seconds(matching[0], division, tempo_map)
            lyric = matching[1].strip().strip("\x00") or None
        units.append(LyricUnit(len(units) + 1, pitch, start_s, end_s, lyric, lyric_time_s))
    if not units:
        raise ChoirError("NATURALIZATION_ALIGNMENT_FAILED", f"{track_label} 音符均在 WAV 有效时长之外")
    return units


def inspect_midi_tracks(path: Path) -> list[MidiTrackInfo]:
    """Return all physical MIDI tracks, including empty tempo/control tracks."""
    try:
        _division, tracks = _read_midi_tracks(path.read_bytes())
        result = []
        for index, track in enumerate(tracks):
            _tempos, _lyrics, _text, notes = _read_track(track)
            pitches = [note[2] for note in notes]
            result.append(MidiTrackInfo(
                index, len(notes), min(pitches) if pitches else None,
                max(pitches) if pitches else None,
            ))
        return result
    except ChoirError:
        raise
    except (OSError, ValueError, IndexError, struct.error) as exc:
        raise ChoirError("NATURALIZATION_MIDI_INVALID", str(exc)) from exc


def parse_lyric_midi(
    path: Path, config: NaturalizationConfig, audio_duration_s: float,
    midi_track_index: int | None = None,
) -> list[LyricUnit]:
    """Read the lyric-bearing MIDI timeline required by timing naturalization."""
    config.validate()
    return parse_midi_notes(
        path, audio_duration_s, require_lyrics=True,
        midi_track_index=midi_track_index,
    )


def naturalize_track(
    source_path: Path,
    track_id: str,
    config: NaturalizationConfig,
    project_dir: Path,
    cache_dir: Path,
    cancel_event: threading.Event | None = None,
    progress: Progress | None = None,
) -> tuple[np.ndarray, list[UnitOffset]]:
    """Return a cached, length-preserving random-offset dry-vocal layer."""
    config.validate()
    _check_cancel(cancel_event)
    source = read_source_wav(source_path)
    assignment = midi_assignment_for_track(track_id, config)
    if assignment is None:
        return source, []
    midi_path = resolve_midi_path(project_dir, assignment.midi_path)
    cache_key = _cache_key(source_path, midi_path, track_id, config)
    audio_cache = cache_dir / f"{cache_key}.npy"
    manifest_cache = cache_dir / f"{cache_key}.json"
    if audio_cache.is_file() and manifest_cache.is_file():
        try:
            cached = np.load(audio_cache, allow_pickle=False).astype(np.float32)
            records = [
                UnitOffset(**item)
                for item in json.loads(manifest_cache.read_text(encoding="utf-8"))["offsets"]
            ]
            if len(cached) == len(source) and np.isfinite(cached).all():
                _check_cancel(cancel_event)
                return cached, records
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    _check_cancel(cancel_event)
    if progress:
        selected = (
            f"第 {assignment.midi_track_index + 1} 轨"
            if assignment.midi_track_index is not None else "首条有音符音轨"
        )
        progress(3, f"正在分析 {track_id} 的 MIDI {selected}")
    units = parse_lyric_midi(
        midi_path, config, len(source) / 48000, assignment.midi_track_index,
    )
    offsets = _choose_offsets(
        units, track_id, config.random_seed,
        audio_duration_s=len(source) / 48000,
    )
    processed = apply_local_time_shifts(source, offsets, 48000, cancel_event)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _check_cancel(cancel_event)
    _atomic_npy(audio_cache, processed)
    try:
        _check_cancel(cancel_event)
    except ChoirError:
        audio_cache.unlink(missing_ok=True)
        raise
    _atomic_json(manifest_cache, {
        "version": NATURALIZATION_VERSION,
        "track_id": track_id,
        "distribution": {
            "onset": {
                "type": "truncated_normal",
                "mean_ms": 0.0,
                "stddev_ms": OFFSET_STDDEV_MS,
                "range_ms": [-OFFSET_LIMIT_MS, OFFSET_LIMIT_MS],
            },
            "tail": {
                "type": "right_skewed_two_sided_truncated_normal",
                "minimum_gap_ms": TAIL_GAP_THRESHOLD_S * 1000,
                "left": {
                    "mean_ms": TAIL_LEFT_MEAN_MS,
                    "stddev_ms": TAIL_LEFT_STDDEV_MS,
                    "range_ms": list(TAIL_LEFT_RANGE_MS),
                },
                "right": {
                    "mean_ms": TAIL_RIGHT_MEAN_MS,
                    "stddev_ms": TAIL_RIGHT_STDDEV_MS,
                    "range_ms": list(TAIL_RIGHT_RANGE_MS),
                    "probability": TAIL_RIGHT_PROBABILITY,
                },
            },
        },
        "offsets": [asdict(item) for item in offsets],
    })
    try:
        _check_cancel(cancel_event)
    except ChoirError:
        audio_cache.unlink(missing_ok=True)
        manifest_cache.unlink(missing_ok=True)
        raise
    return processed, offsets


def apply_local_time_shifts(
    source: np.ndarray,
    offsets: list[UnitOffset],
    sample_rate: int,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """Apply non-overlapping local time maps with smooth crossfaded edges."""
    if not offsets:
        return source.astype(np.float32, copy=True)
    result = source.astype(np.float32, copy=True)
    length = len(source)
    source_positions = np.arange(length, dtype=np.float64)
    cuda_sampler = (
        create_linear_sampler(source)
        if length >= _CUDA_INTERPOLATION_THRESHOLD_SAMPLES else None
    )
    starts = [record.note_start_s * sample_rate for record in offsets]
    for index, record in enumerate(offsets):
        _check_cancel(cancel_event)
        start = int(round(starts[index]))
        previous_start = starts[index - 1] if index else max(0.0, starts[index] - sample_rate * 0.05)
        next_start = starts[index + 1] if index + 1 < len(starts) else float(length)
        left = max(0, int(round((previous_start + starts[index]) / 2))) if index else int(round(previous_start))
        right = min(length, int(round((starts[index] + next_start) / 2))) if index + 1 < len(starts) else min(length, int(round(next_start)))
        if right - left >= 3:
            center = max(left + 1, min(start, right - 2))
        else:
            center = start
        if right - left < 2:
            continue

        shift = record.offset_ms * sample_rate / 1000.0
        tail_shift = record.end_offset_ms * sample_rate / 1000.0
        # Consonants carry the sharpest onset transients.  Keep their first
        # 20 ms untouched, then ease the timing offset into the vowel body.
        attack_end = min(right - 2, center + int(round(0.020 * sample_rate)))
        if attack_end <= center or right - attack_end < 3:
            continue
        output_positions = np.arange(left, right, dtype=np.float64)
        note_end = max(attack_end + 1, min(right - 2, int(round(record.note_end_s * sample_rate))))
        # Map the source note end to its randomized output time.  The local
        # region is bounded by neighbouring note starts, so a very long tail is
        # safely limited before it can overwrite the following consonant.
        tail_output = max(attack_end + 1, min(right - 2, note_end + tail_shift))
        if tail_output > attack_end + 1:
            map_output = np.array([left, attack_end, tail_output, right - 1], dtype=np.float64)
            map_input = np.array([left, attack_end - shift, note_end, right - 1], dtype=np.float64)
        else:
            map_output = np.array([left, attack_end, right - 1], dtype=np.float64)
            map_input = np.array([left, attack_end - shift, right - 1], dtype=np.float64)
        mapped = np.interp(output_positions, map_output, map_input)
        np.clip(mapped, 0, length - 1, out=mapped)
        warped = np.empty(right - left, dtype=np.float32)
        for chunk_start in range(0, len(mapped), _INTERPOLATION_CHUNK_SAMPLES):
            _check_cancel(cancel_event)
            chunk_end = min(chunk_start + _INTERPOLATION_CHUNK_SAMPLES, len(mapped))
            mapped_chunk = mapped[chunk_start:chunk_end]
            if cuda_sampler is not None:
                warped[chunk_start:chunk_end] = cuda_sampler.sample(mapped_chunk)
            else:
                warped[chunk_start:chunk_end] = np.interp(
                    mapped_chunk, source_positions, source
                )
        _check_cancel(cancel_event)

        blend = np.zeros(right - left, dtype=np.float32)
        local_attack_end = attack_end - left
        fade_samples = min(int(round(0.050 * sample_rate)), max(1, (right - attack_end) // 3))
        fade_end = min(len(blend), local_attack_end + fade_samples)
        blend[local_attack_end:fade_end] = np.sin(
            np.linspace(0.0, np.pi / 2, fade_end - local_attack_end, endpoint=False, dtype=np.float32)
        ) ** 2
        release_start = max(fade_end, len(blend) - fade_samples)
        blend[fade_end:release_start] = 1.0
        blend[release_start:] = np.cos(
            np.linspace(0.0, np.pi / 2, len(blend) - release_start, endpoint=True, dtype=np.float32)
        ) ** 2
        original = source[left:right]
        result[left:right] = original * (1.0 - blend) + warped * blend
    return result


def _choose_offsets(
    units: list[LyricUnit], track_id: str, random_seed: int,
    *, audio_duration_s: float | None = None,
) -> list[UnitOffset]:
    stable_track_seed = int.from_bytes(hashlib.sha256(track_id.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng((random_seed ^ stable_track_seed) & ((1 << 64) - 1))
    records = []
    for index, unit in enumerate(units):
        value = float(rng.normal(0.0, OFFSET_STDDEV_MS))
        while not -OFFSET_LIMIT_MS <= value <= OFFSET_LIMIT_MS:
            value = float(rng.normal(0.0, OFFSET_STDDEV_MS))
        next_unit = units[index + 1] if index + 1 < len(units) else None
        end_offset = 0.0
        # MIDI tick conversion can turn an exact 20 ms gap into
        # 0.020000000000000018, so retain the strict rule with a tiny tolerance.
        if (
            next_unit is not None
            and unit.end_s - unit.start_s > SHORT_NOTE_TAIL_OFFSET_THRESHOLD_S
            and next_unit.start_s - unit.end_s > TAIL_GAP_THRESHOLD_S + 1e-9
        ):
            end_offset = _choose_tail_offset(rng)
        records.append(UnitOffset(
            unit.index, unit.pitch, unit.start_s, unit.end_s,
            unit.lyric, unit.lyric_time_s, round(value, 3), round(end_offset, 3),
        ))
    return records


def _choose_tail_offset(rng: np.random.Generator) -> float:
    """Sample the requested two-sided, right-skewed tail-time distribution."""
    if float(rng.random()) < TAIL_RIGHT_PROBABILITY:
        mean, stddev, bounds = (
            TAIL_RIGHT_MEAN_MS, TAIL_RIGHT_STDDEV_MS, TAIL_RIGHT_RANGE_MS,
        )
    else:
        mean, stddev, bounds = (
            TAIL_LEFT_MEAN_MS, TAIL_LEFT_STDDEV_MS, TAIL_LEFT_RANGE_MS,
        )
    value = float(rng.normal(mean, stddev))
    while not bounds[0] <= value <= bounds[1]:
        value = float(rng.normal(mean, stddev))
    return value


def _read_midi_tracks(payload: bytes) -> tuple[int, list[bytes]]:
    if len(payload) < 14 or payload[:4] != b"MThd":
        raise ChoirError("NATURALIZATION_MIDI_INVALID", "不是标准 MIDI 文件")
    header_length = struct.unpack(">I", payload[4:8])[0]
    if header_length < 6:
        raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI 头长度无效")
    _format, track_count, division = struct.unpack(">HHH", payload[8:14])
    if track_count < 1:
        raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI 没有轨道")
    if division & 0x8000 or division == 0:
        raise ChoirError("NATURALIZATION_MIDI_INVALID", "不支持 SMPTE 时间格式")
    offset = 8 + header_length
    tracks = []
    for index in range(track_count):
        if payload[offset:offset + 4] != b"MTrk":
            raise ChoirError("NATURALIZATION_MIDI_INVALID", f"缺少第 {index + 1} 条 MTrk 数据块")
        if offset + 8 > len(payload):
            raise ChoirError("NATURALIZATION_MIDI_INVALID", f"第 {index + 1} 轨头数据不完整")
        size = struct.unpack(">I", payload[offset + 4:offset + 8])[0]
        offset += 8
        track = payload[offset:offset + size]
        if len(track) != size:
            raise ChoirError("NATURALIZATION_MIDI_INVALID", f"第 {index + 1} 轨数据不完整")
        tracks.append(track)
        offset += size
    return division, tracks


def _read_track(
    track: bytes,
) -> tuple[list[tuple[int, int]], list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, int, int]]]:
    position = 0
    tick = 0
    running: int | None = None
    tempos: list[tuple[int, int]] = []
    lyrics: list[tuple[int, str]] = []
    text_events: list[tuple[int, str]] = []
    notes: list[tuple[int, int, int]] = []
    active: dict[tuple[int, int], list[int]] = {}
    while position < len(track):
        delta, position = _read_varlen(track, position)
        tick += delta
        if position >= len(track):
            raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI 事件状态缺失")
        status = track[position]
        if status & 0x80:
            position += 1
            if status < 0xF0:
                running = status
        elif running is not None:
            status = running
        else:
            raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI running status 无效")
        if status == 0xFF:
            if position >= len(track):
                raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI meta 事件不完整")
            meta_type = track[position]
            length, position = _read_varlen(track, position + 1)
            data = track[position:position + length]
            if len(data) != length:
                raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI meta 数据不完整")
            position += length
            running = None
            if meta_type == 0x51 and len(data) == 3:
                tempos.append((tick, int.from_bytes(data, "big")))
            elif meta_type == 0x05:
                lyrics.append((tick, _decode_midi_text(data)))
            elif meta_type == 0x01:
                text_events.append((tick, _decode_midi_text(data)))
            elif meta_type == 0x2F:
                break
            continue
        if status in (0xF0, 0xF7):
            length, position = _read_varlen(track, position)
            position += length
            running = None
            continue
        event_type = status >> 4
        channel = status & 0x0F
        data_length = 1 if event_type in {0xC, 0xD} else 2
        data = track[position:position + data_length]
        if len(data) != data_length:
            raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI 事件数据不完整")
        position += data_length
        if event_type == 0x9 and data[1] > 0:
            active.setdefault((channel, data[0]), []).append(tick)
        elif event_type == 0x8 or (event_type == 0x9 and data[1] == 0):
            starts = active.get((channel, data[0]))
            if starts:
                start = starts.pop(0)
                notes.append((start, max(tick, start + 1), data[0]))
    for (_channel, pitch), starts in active.items():
        notes.extend((start, start + 1, pitch) for start in starts)
    return tempos, lyrics, text_events, notes


def _read_varlen(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if position >= len(data):
            raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI 可变长整数不完整")
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, position
    raise ChoirError("NATURALIZATION_MIDI_INVALID", "MIDI 可变长整数过长")


def _decode_midi_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
    for encoding in ("utf-8", "shift_jis", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _tempo_map(tempos: list[tuple[int, int]]) -> list[tuple[int, int]]:
    by_tick = {tick: tempo for tick, tempo in tempos if tempo > 0}
    by_tick.setdefault(0, 500000)
    return sorted(by_tick.items())


def _tick_seconds(tick: int, division: int, tempos: list[tuple[int, int]]) -> float:
    seconds = 0.0
    previous_tick, tempo = tempos[0]
    for change_tick, new_tempo in tempos[1:]:
        if change_tick >= tick:
            break
        seconds += (change_tick - previous_tick) * tempo / division / 1_000_000
        previous_tick, tempo = change_tick, new_tempo
    return seconds + (tick - previous_tick) * tempo / division / 1_000_000


def _cache_key(
    source_path: Path, midi_path: Path, track_id: str, config: NaturalizationConfig,
) -> str:
    payload = {
        "version": NATURALIZATION_VERSION,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "midi_sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
        "track_id": track_id,
        "config": asdict(config),
        "stddev_ms": OFFSET_STDDEV_MS,
        "limit_ms": OFFSET_LIMIT_MS,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_npy(target: Path, data: np.ndarray) -> None:
    handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npy", dir=target.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        np.save(temp, data.astype(np.float32), allow_pickle=False)
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _atomic_json(target: Path, data: dict) -> None:
    handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ChoirError("RENDER_CANCELLED")
