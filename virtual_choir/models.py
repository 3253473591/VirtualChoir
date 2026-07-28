from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ChoirError

EPSILON = 1e-6
TRACK_RE = re.compile(r"^singer_[1-9][0-9]*$")
NATURALIZATION_LANGUAGES = {"普通话", "日语", "英语", "韩语", "粤语", "多语种混合"}
SEGMENT_LANGUAGES = NATURALIZATION_LANGUAGES - {"多语种混合"}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ChoirError("PROJECT_SCHEMA_ERROR", f"{name} 必须是有限 number")
    return float(value)


def _number(value: Any, name: str, low: float, high: float, *, low_open=False, high_open=False) -> float:
    value = _finite(value, name)
    if value < low or value > high or (low_open and value <= low) or (high_open and value >= high):
        raise ChoirError("PROJECT_SCHEMA_ERROR", f"{name} 必须在允许范围内")
    return value


def grid_nodes(length: float, step: float) -> list[float]:
    """Return all X/Y snap targets, including the room boundary.

    The final boundary is deliberately a node even when the room dimension is
    not an exact multiple of the step, matching the room-view grid.
    """
    nodes = [0.0]
    k = 1
    while k * step < length:
        nodes.append(k * step)
        k += 1
    if abs(nodes[-1] - length) > EPSILON:
        nodes.append(length)
    return nodes


def snap_to_grid(value: float, length: float, step: float) -> float:
    """Snap an in-range X/Y coordinate to its nearest grid node.

    Equal-distance ties choose the lower coordinate so all input paths use a
    deterministic result.
    """
    nodes = grid_nodes(length, step)
    distance = min(abs(node - value) for node in nodes)
    return min(node for node in nodes if abs(abs(node - value) - distance) < EPSILON)


@dataclass
class Position:
    x_m: float
    y_m: float
    z_m: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        _exact_keys(data, {"x_m", "y_m", "z_m"}, "position")
        return cls(_finite(data["x_m"], "x_m"), _finite(data["y_m"], "y_m"), _finite(data["z_m"], "z_m"))


@dataclass
class RoomConfig:
    length_m: float = 8.0
    width_m: float = 6.0
    height_m: float = 3.0
    rt60_s: float = 0.55
    reverb_gain_db: float = -12.0
    bus_gain_db: float = 0.0
    grid_step_m: float = 0.5

    def validate(self) -> None:
        self.length_m = _number(self.length_m, "room.length_m", 0, 100, low_open=True)
        self.width_m = _number(self.width_m, "room.width_m", 0, 100, low_open=True)
        self.height_m = _number(self.height_m, "room.height_m", 0, 30, low_open=True)
        self.rt60_s = _number(self.rt60_s, "room.rt60_s", .2, 2)
        self.reverb_gain_db = _number(self.reverb_gain_db, "room.reverb_gain_db", -30, 0)
        self.bus_gain_db = _number(self.bus_gain_db, "room.bus_gain_db", -24, 12)
        self.grid_step_m = _number(self.grid_step_m, "room.grid_step_m", .01, min(self.length_m, self.width_m))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoomConfig":
        allowed = set(cls.__dataclass_fields__)
        _exact_keys(data, allowed, "room", required=allowed - {"bus_gain_db"})
        result = cls(**{**data, "bus_gain_db": data.get("bus_gain_db", 0.0)})
        result.validate(); return result


@dataclass
class MicrophoneInput:
    count: int = 2
    spacing_m: float = 0.6
    height_m: float = 1.7

    def validate(self, room: RoomConfig) -> None:
        if type(self.count) is not int or not 2 <= self.count <= 6:
            raise ChoirError("INVALID_MICROPHONE", "count 必须是 2 到 6 的整数")
        max_spacing = min(3.0, room.width_m / (self.count - 1))
        self.spacing_m = _number(
            self.spacing_m, "microphone.spacing_m", .2, max_spacing, high_open=True,
        )
        self.height_m = _number(self.height_m, "microphone.height_m", 0, room.height_m, low_open=True, high_open=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any], room: RoomConfig) -> "MicrophoneInput":
        _exact_keys(data, set(cls.__dataclass_fields__), "microphone")
        result = cls(**data); result.validate(room); return result


@dataclass
class TrackConfig:
    track_id: str
    file_name: str
    position: Position
    gain_db: float = 0.0
    enabled: bool = True
    source_path: str | None = None
    parent_source: str | None = None
    copy_index: int | None = None
    variation_preset: int | None = None

    def validate(self, room: RoomConfig) -> None:
        if not isinstance(self.track_id, str) or not TRACK_RE.fullmatch(self.track_id):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "track_id 格式无效")
        if not isinstance(self.file_name, str) or not self.file_name or len(self.file_name) > 255:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "file_name 无效")
        self.gain_db = _number(self.gain_db, "gain_db", -60, 12)
        if type(self.enabled) is not bool:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "enabled 必须是 boolean")
        if self.parent_source is not None and not isinstance(self.parent_source, str):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "parent_source 必须是字符串或 null")
        if self.copy_index is not None and (type(self.copy_index) is not int or self.copy_index < 1):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "copy_index 必须为正整数或 null")
        if self.variation_preset is not None and (
            type(self.variation_preset) is not int or not 1 <= self.variation_preset <= 5
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "variation_preset 必须为 1-5 或 null")
        validate_position(self.position, room)

    @classmethod
    def from_dict(cls, data: dict[str, Any], room: RoomConfig) -> "TrackConfig":
        allowed = {"track_id", "file_name", "position", "gain_db", "enabled", "source_path", "parent_source", "copy_index", "variation_preset"}
        _exact_keys(data, allowed, "track", required={"track_id", "file_name", "position", "gain_db", "enabled"})
        result = cls(**{**data, "position": Position.from_dict(data["position"])})
        result.validate(room); return result


@dataclass
class LanguageSegment:
    start_unit: int
    end_unit: int
    language: str

    def validate(self) -> None:
        if type(self.start_unit) is not int or type(self.end_unit) is not int:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "语言区段序号必须是整数")
        if self.start_unit < 1 or self.end_unit < self.start_unit:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "语言区段范围无效")
        if self.language not in SEGMENT_LANGUAGES:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "语言区段语言无效")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LanguageSegment":
        _exact_keys(data, {"start_unit", "end_unit", "language"}, "language_segment")
        result = cls(**data)
        result.validate()
        return result


@dataclass
class MidiAssignment:
    midi_path: str
    track_ids: list[str] = field(default_factory=list)
    midi_track_index: int | None = None

    def validate(self, project: "ProjectConfig | None" = None) -> None:
        if not isinstance(self.midi_path, str) or not self.midi_path.strip():
            raise ChoirError("NATURALIZATION_INPUT_MISSING", "MIDI 路径无效")
        if not isinstance(self.track_ids, list) or not all(
            isinstance(track_id, str) and TRACK_RE.fullmatch(track_id)
            for track_id in self.track_ids
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "MIDI 分配的 track_ids 必须为轨道 ID 列表")
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "MIDI 分配中包含重复轨道")
        if self.midi_track_index is not None and (
            type(self.midi_track_index) is not int or self.midi_track_index < 0
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "MIDI 内部音轨编号无效")
        if project is not None:
            known_ids = {track.track_id for track in project.tracks}
            invalid = set(self.track_ids) - known_ids
            if invalid:
                names = "、".join(sorted(invalid))
                raise ChoirError("NATURALIZATION_TRACK_MISMATCH", f"MIDI 分配包含无效轨道：{names}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MidiAssignment":
        _exact_keys(
            data, {"midi_path", "track_ids", "midi_track_index"},
            "midi_assignment", required={"midi_path"},
        )
        result = cls(
            data["midi_path"], list(data.get("track_ids", [])),
            data.get("midi_track_index"),
        )
        result.validate()
        return result


@dataclass(init=False)
class NaturalizationConfig:
    enabled: bool = False
    assignments: list[MidiAssignment] = field(default_factory=list)
    random_seed: int = 20260724

    def __init__(
        self,
        enabled: bool = False,
        assignments: list[MidiAssignment] | str | None = None,
        random_seed: int = 20260724,
        *,
        midi_path: str | None = None,
    ) -> None:
        # A string in the former second positional argument is the legacy
        # ``midi_path`` API. Keep accepting it while storing only assignments.
        if isinstance(assignments, str):
            midi_path = assignments
            assignments = None
        self.enabled = enabled
        self.assignments = list(assignments or [])
        if midi_path and not self.assignments:
            self.assignments = [MidiAssignment(midi_path, [])]
        self.random_seed = random_seed

    @property
    def midi_path(self) -> str | None:
        """Legacy single-MIDI view used by older integrations."""
        return self.assignments[0].midi_path if self.assignments else None

    @midi_path.setter
    def midi_path(self, value: str | None) -> None:
        if value:
            self.assignments = [MidiAssignment(value, [])]
        else:
            self.assignments = []

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "naturalization.enabled 必须是 boolean")
        if not isinstance(self.assignments, list) or not all(
            isinstance(assignment, MidiAssignment) for assignment in self.assignments
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "naturalization.assignments 必须为 MIDI 分配列表")
        for assignment in self.assignments:
            assignment.validate()
        assigned_track_ids = [
            track_id
            for assignment in self.assignments
            for track_id in assignment.track_ids
        ]
        if len(set(assigned_track_ids)) != len(assigned_track_ids):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "同一轨道不能分配给多个 MIDI")
        if type(self.random_seed) is not int or not -(2**63) <= self.random_seed < 2**63:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "naturalization.random_seed 无效")
        if self.enabled:
            if not self.assignments:
                raise ChoirError("NATURALIZATION_INPUT_MISSING", "请提供带歌词的 MIDI 文件")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NaturalizationConfig":
        # ``naturalization`` was previously language/phoneme based.  Preserve
        # old projects by deliberately discarding now-obsolete language and
        # selection fields while retaining the MIDI reference and random seed.
        allowed = set(cls.__dataclass_fields__) | {
            "midi_path",
            "language", "language_segments", "selection_ratio",
            "min_offset_ms", "max_offset_ms",
        }
        _exact_keys(data, allowed, "naturalization", required={"enabled"})
        assignments = [MidiAssignment.from_dict(item) for item in data.get("assignments", [])]
        if not assignments and data.get("midi_path"):
            assignments = [MidiAssignment(data["midi_path"], [])]
        result = cls(
            enabled=data["enabled"], assignments=assignments,
            random_seed=data.get("random_seed", 20260724),
        )
        result.validate()
        return result


def midi_for_track(track_id: str, config: NaturalizationConfig) -> str | None:
    """Return the first explicitly assigned MIDI, or the single-MIDI default."""
    assignment = midi_assignment_for_track(track_id, config)
    return assignment.midi_path if assignment else None


def midi_assignment_for_track(
    track_id: str, config: NaturalizationConfig,
) -> MidiAssignment | None:
    """Return the owning MIDI assignment, including its internal track choice."""
    for assignment in config.assignments:
        if track_id in assignment.track_ids:
            return assignment
    if len(config.assignments) == 1 and not config.assignments[0].track_ids:
        return config.assignments[0]
    return None


@dataclass
class TimbreVariationConfig:
    """Concrete variation ranges selected from one of the five presets."""

    preset_level: int = 3
    formant_shift_range: tuple[float, float] = (-0.05, 0.05)
    pitch_shift_cents_range: tuple[float, float] = (-4.0, 4.0)
    pitch_line_cents_range: tuple[float, float] = (-3.0, 3.0)
    pitch_redraw_mix: float = 0.70
    jitter_cents_range: tuple[float, float] = (0.7, 1.7)
    vowel_onset_db_range: tuple[float, float] = (-1.5, 1.5)
    dynamic_db_range: tuple[float, float] = (-1.5, 1.5)
    eq_mid_db_range: tuple[float, float] = (-1.5, 1.5)
    eq_high_db_range: tuple[float, float] = (-1.5, 1.5)
    breath_mix_range: tuple[float, float] = (0.003, 0.008)
    random_seed: int = 20260724
    vibrato_depth_cents_range: tuple[float, float] = (4.0, 18.0)
    vibrato_rate_hz_range: tuple[float, float] = (4.5, 6.5)
    vibrato_note_probability: float = 0.7
    # Activation likelihoods for CREPE-classified (flat, light, natural,
    # strong) source vibrato.  MIDI note boundaries make this per-note.
    vibrato_activation_probabilities: tuple[float, float, float, float] = (
        0.92, 0.78, 0.65, 0.50,
    )
    # A small low-to-target onset gesture, only available with MIDI note
    # boundaries.  Presets 1-2 disable it; presets 3-5 share this range.
    onset_scoop_depth_cents_range: tuple[float, float] = (8.0, 20.0)
    onset_scoop_duration_ms_range: tuple[float, float] = (35.0, 50.0)
    onset_scoop_note_probability: float = 0.70

    @classmethod
    def from_preset(cls, level: int) -> "TimbreVariationConfig":
        """Return fixed ranges for a user-facing differentiation preset."""
        presets = {
            1: ((-0.015, 0.015), (-1.5, 1.5), (-3.0, 3.0), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.001, 0.003)),
            2: ((-0.03, 0.03), (-2.5, 2.5), (-6.0, 6.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.002, 0.005)),
            3: ((-0.05, 0.05), (-4.0, 4.0), (-10.0, 10.0), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (0.003, 0.008)),
            4: ((-0.07, 0.07), (-6.0, 6.0), (-16.0, 16.0), (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5), (0.005, 0.011)),
            5: ((-0.10, 0.10), (-9.0, 9.0), (-24.0, 24.0), (-3.5, 3.5), (-4.0, 4.0), (-3.5, 3.5), (-3.5, 3.5), (0.008, 0.015)),
        }
        try:
            values = presets[level]
        except KeyError as exc:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "差异化预设必须为 1-5") from exc
        vibrato = {
            1: ((1.5, 4.5), (4.7, 5.4), 0.35, (0.75, 0.55, 0.35, 0.20), 0.50, (0.3, 0.8)),
            2: ((2.5, 7.0), (4.5, 5.8), 0.50, (0.85, 0.68, 0.50, 0.35), 0.60, (0.5, 1.2)),
            3: ((4.0, 11.0), (4.3, 6.2), 0.65, (0.92, 0.78, 0.65, 0.50), 0.70, (0.7, 1.7)),
            4: ((6.0, 17.0), (4.1, 6.6), 0.78, (0.96, 0.88, 0.78, 0.68), 0.80, (0.9, 2.4)),
            5: ((8.0, 25.0), (3.9, 7.0), 0.90, (0.99, 0.94, 0.88, 0.80), 0.95, (1.2, 3.2)),
        }[level]
        onset_scoop = (
            ((0.0, 0.0), (0.0, 0.0), 0.0)
            if level < 3
            else ((8.0, 20.0), (35.0, 50.0), 0.70)
        )
        return cls(
            preset_level=level,
            formant_shift_range=values[0],
            pitch_shift_cents_range=values[1],
            pitch_line_cents_range=values[2],
            vowel_onset_db_range=values[3],
            dynamic_db_range=values[4],
            eq_mid_db_range=values[5],
            eq_high_db_range=values[6],
            breath_mix_range=values[7],
            vibrato_depth_cents_range=vibrato[0],
            vibrato_rate_hz_range=vibrato[1],
            vibrato_note_probability=vibrato[2],
            vibrato_activation_probabilities=vibrato[3],
            pitch_redraw_mix=vibrato[4],
            jitter_cents_range=vibrato[5],
            onset_scoop_depth_cents_range=onset_scoop[0],
            onset_scoop_duration_ms_range=onset_scoop[1],
            onset_scoop_note_probability=onset_scoop[2],
        )

    def validate(self) -> None:
        for name, (lo, hi) in [
            ("formant_shift_range", self.formant_shift_range),
            ("pitch_shift_cents_range", self.pitch_shift_cents_range),
            ("pitch_line_cents_range", self.pitch_line_cents_range),
            ("jitter_cents_range", self.jitter_cents_range),
            ("vowel_onset_db_range", self.vowel_onset_db_range),
            ("dynamic_db_range", self.dynamic_db_range),
            ("eq_mid_db_range", self.eq_mid_db_range),
            ("eq_high_db_range", self.eq_high_db_range),
            ("breath_mix_range", self.breath_mix_range),
            ("vibrato_depth_cents_range", self.vibrato_depth_cents_range),
            ("vibrato_rate_hz_range", self.vibrato_rate_hz_range),
            ("onset_scoop_depth_cents_range", self.onset_scoop_depth_cents_range),
            ("onset_scoop_duration_ms_range", self.onset_scoop_duration_ms_range),
        ]:
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo > hi:
                raise ChoirError("PROJECT_SCHEMA_ERROR", f"TimbreVariationConfig.{name} 无效")
        if not isinstance(self.vibrato_note_probability, (int, float)) or not 0 <= self.vibrato_note_probability <= 1:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "TimbreVariationConfig.vibrato_note_probability 无效")
        if not isinstance(self.onset_scoop_note_probability, (int, float)) or not 0 <= self.onset_scoop_note_probability <= 1:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "TimbreVariationConfig.onset_scoop_note_probability 无效")
        if not isinstance(self.pitch_redraw_mix, (int, float)) or not 0 <= self.pitch_redraw_mix <= 1:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "TimbreVariationConfig.pitch_redraw_mix 无效")
        if len(self.vibrato_activation_probabilities) != 4 or any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1
            for value in self.vibrato_activation_probabilities
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "TimbreVariationConfig.vibrato_activation_probabilities 无效")
        if type(self.random_seed) is not int:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "TimbreVariationConfig.random_seed 无效")
        if type(self.preset_level) is not int or not 1 <= self.preset_level <= 5:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "TimbreVariationConfig.preset_level 无效")


@dataclass
class ProjectConfig:
    project_name: str = "virtual_choir"
    room: RoomConfig = field(default_factory=RoomConfig)
    microphone: MicrophoneInput = field(default_factory=MicrophoneInput)
    tracks: list[TrackConfig] = field(default_factory=list)
    ai_recommendations: list[dict[str, Any]] = field(default_factory=list)
    ai_conversation: list[dict[str, str]] = field(default_factory=list)
    naturalization: NaturalizationConfig = field(default_factory=NaturalizationConfig)
    next_track_sequence: int = 1
    created_at: str | None = None
    modified_at: str | None = None

    def microphone_positions(self) -> list[dict[str, float | str]]:
        center = self.room.width_m / 2
        offset = (self.microphone.count - 1) / 2
        return [
            {
                "id": f"microphone_{index + 1}",
                "x_m": center + (index - offset) * self.microphone.spacing_m,
                "y_m": 0.0,
                "z_m": self.microphone.height_m,
            }
            for index in range(self.microphone.count)
        ]

    def validate(self, project_dir: Path | None = None, *, require_sources: bool = False, require_track=False) -> None:
        self.room.validate(); self.microphone.validate(self.room)
        if type(self.next_track_sequence) is not int or self.next_track_sequence < 1:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "next_track_sequence 无效")
        if len(self.tracks) > 256: raise ChoirError("TRACK_LIMIT_EXCEEDED")
        self.naturalization.validate()
        if not isinstance(self.ai_recommendations, list) or not all(
            isinstance(item, dict) for item in self.ai_recommendations
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "ai_recommendations 必须为对象列表")
        if not isinstance(self.ai_conversation, list) or not all(
            isinstance(item, dict)
            and set(item) == {"role", "content"}
            and item["role"] in {"user", "assistant"}
            and isinstance(item["content"], str)
            and item["content"].strip()
            for item in self.ai_conversation
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "ai_conversation 格式无效")
        ids: set[str] = set()
        for track in self.tracks:
            track.validate(self.room)
            if track.track_id in ids: raise ChoirError("DUPLICATE_TRACK_ID", track.track_id)
            ids.add(track.track_id)
            if require_sources and project_dir:
                path = resolve_source_path(project_dir, track.source_path or track.file_name)
                if not path.is_file(): raise ChoirError("AUDIO_NOT_FOUND", str(path))
        for assignment in self.naturalization.assignments:
            assignment.validate(self)
        if require_track and not any(track.enabled for track in self.tracks):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "至少需要一条启用轨道")

    def add_track(self, source_path: Path, position: Position | None = None) -> TrackConfig:
        if len(self.tracks) >= 256: raise ChoirError("TRACK_LIMIT_EXCEEDED")
        track = TrackConfig(f"singer_{self.next_track_sequence}", source_path.name, position or Position(self.room.width_m/2, self.room.length_m/2, min(1.6, self.room.height_m)), 0.0, True, str(source_path))
        track.validate(self.room); self.tracks.append(track); self.next_track_sequence += 1
        return track

    def to_dict(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.created_at = self.created_at or now; self.modified_at = now
        tracks = []
        for track in self.tracks:
            item = asdict(track)
            for nil_key in ("source_path", "parent_source", "copy_index", "variation_preset"):
                if item.get(nil_key) is None:
                    item.pop(nil_key, None)
            tracks.append(item)
        result = {"schema_version": 1, "project": {"name": self.project_name, "sample_rate_hz": 48000, "source_bit_depth": 32, "source_format": "wav", "source_channels": 1, "created_at": self.created_at, "modified_at": self.modified_at}, "room": asdict(self.room), "microphone": asdict(self.microphone), "microphone_positions": self.microphone_positions(), "tracks": tracks, "next_track_sequence": self.next_track_sequence}
        if self.ai_recommendations:
            result["ai_recommendations"] = self.ai_recommendations
        if self.ai_conversation:
            result["ai_conversation"] = self.ai_conversation
        if self.naturalization.enabled or self.naturalization.assignments:
            result["naturalization"] = asdict(self.naturalization)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        _exact_keys(data, {"schema_version", "project", "room", "microphone", "microphone_positions", "tracks", "ai_recommendations", "ai_conversation", "naturalization", "next_track_sequence"}, "project_config", required={"schema_version", "project", "room", "microphone", "tracks", "next_track_sequence"})
        if data.get("schema_version") != 1: raise ChoirError("PROJECT_VERSION_UNSUPPORTED")
        project = data["project"]
        _exact_keys(project, {"name", "sample_rate_hz", "source_bit_depth", "source_format", "source_channels", "created_at", "modified_at"}, "project", required={"name", "sample_rate_hz", "source_bit_depth", "source_format", "source_channels"})
        if (project["sample_rate_hz"], project["source_bit_depth"], project["source_format"], project["source_channels"]) != (48000, 32, "wav", 1):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "源音频合同必须为 48kHz/32-bit/mono WAV")
        room = RoomConfig.from_dict(data["room"]); mic = MicrophoneInput.from_dict(data["microphone"], room)
        result = cls(
            project_name=project["name"], room=room, microphone=mic,
            tracks=[TrackConfig.from_dict(t, room) for t in data["tracks"]],
            ai_recommendations=data.get("ai_recommendations", []),
            ai_conversation=data.get("ai_conversation", []),
            naturalization=NaturalizationConfig.from_dict(data["naturalization"]) if "naturalization" in data else NaturalizationConfig(),
            next_track_sequence=data["next_track_sequence"],
            created_at=project.get("created_at"), modified_at=project.get("modified_at"),
        )
        expected = result.microphone_positions()
        if "microphone_positions" in data:
            legacy = [
                {"id": "microphone_left", "x_m": result.room.width_m / 2 - result.microphone.spacing_m / 2, "y_m": 0.0, "z_m": result.microphone.height_m},
                {"id": "microphone_right", "x_m": result.room.width_m / 2 + result.microphone.spacing_m / 2, "y_m": 0.0, "z_m": result.microphone.height_m},
            ]
            if data["microphone_positions"] != expected and not (
                result.microphone.count == 2 and data["microphone_positions"] == legacy
            ):
                raise ChoirError("PROJECT_SCHEMA_ERROR", "microphone_positions 必须匹配本地公式")
        result.validate(); return result


def _exact_keys(data: Any, allowed: set[str], label: str, required: set[str] | None = None) -> None:
    if not isinstance(data, dict): raise ChoirError("PROJECT_SCHEMA_ERROR", f"{label} 必须为对象")
    required = required or allowed
    if set(data) - allowed or required - set(data): raise ChoirError("PROJECT_SCHEMA_ERROR", f"{label} 字段不符合 schema")


def validate_position(position: Position, room: RoomConfig) -> None:
    for value, maximum, label in ((position.x_m, room.width_m, "x_m"), (position.y_m, room.length_m, "y_m"), (position.z_m, room.height_m, "z_m")):
        if not math.isfinite(value) or value < 0 or value > maximum: raise ChoirError("INVALID_COORDINATE", label)


def resolve_source_path(project_dir: Path, source_path: str) -> Path:
    candidate = Path(source_path)
    if candidate.is_absolute(): return candidate.resolve()
    resolved = (project_dir / candidate).resolve()
    if project_dir.resolve() not in resolved.parents and resolved != project_dir.resolve():
        raise ChoirError("AUDIO_NOT_FOUND", "source_path 超出工程目录")
    return resolved
