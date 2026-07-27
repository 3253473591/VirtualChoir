from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ChoirError
from .models import TRACK_RE, MicrophoneInput, Position, ProjectConfig, RoomConfig, validate_position
from .project_io import save_json


PRESET_VERSION = 1


@dataclass
class SingerPreset:
    track_id: str
    position: Position
    gain_db: float

    def validate(self, room: RoomConfig) -> None:
        if not isinstance(self.track_id, str) or not TRACK_RE.fullmatch(self.track_id):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手 track_id 无效")
        if (
            isinstance(self.gain_db, bool)
            or not isinstance(self.gain_db, (int, float))
            or not math.isfinite(self.gain_db)
            or not -60 <= self.gain_db <= 12
        ):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手 gain_db 无效")
        validate_position(self.position, room)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "position": asdict(self.position),
            "gain_db": self.gain_db,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SingerPreset":
        if not isinstance(data, dict) or set(data) != {"track_id", "position", "gain_db"}:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手字段不符合 schema")
        return cls(
            track_id=data["track_id"],
            position=Position.from_dict(data["position"]),
            gain_db=data["gain_db"],
        )


@dataclass
class ProjectPreset:
    singer_count: int
    room: RoomConfig
    microphone: MicrophoneInput
    singers: list[SingerPreset]

    @classmethod
    def from_project(cls, project: ProjectConfig) -> "ProjectPreset":
        return cls(
            singer_count=len(project.tracks),
            room=RoomConfig(
                length_m=project.room.length_m,
                width_m=project.room.width_m,
                height_m=project.room.height_m,
                rt60_s=project.room.rt60_s,
                reverb_gain_db=project.room.reverb_gain_db,
            ),
            microphone=MicrophoneInput(
                count=project.microphone.count,
                spacing_m=project.microphone.spacing_m,
                height_m=project.microphone.height_m,
            ),
            singers=[
                SingerPreset(
                    track_id=track.track_id,
                    position=Position(track.position.x_m, track.position.y_m, track.position.z_m),
                    gain_db=track.gain_db,
                )
                for track in project.tracks
            ],
        )

    def validate(self) -> None:
        if type(self.singer_count) is not int or self.singer_count < 0:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手数量无效")
        if len(self.singers) != self.singer_count:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手数量不匹配")
        self.room.validate()
        self.microphone.validate(self.room)
        if len({singer.track_id for singer in self.singers}) != len(self.singers):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手 track_id 重复")
        for singer in self.singers:
            singer.validate(self.room)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        # MIDI and all naturalization settings are intentionally excluded.
        return {
            "preset_version": PRESET_VERSION,
            "singer_count": self.singer_count,
            "room": {
                "length_m": self.room.length_m,
                "width_m": self.room.width_m,
                "height_m": self.room.height_m,
                "rt60_s": self.room.rt60_s,
                "reverb_gain_db": self.room.reverb_gain_db,
            },
            "microphone": asdict(self.microphone),
            # Audio file names and MIDI input are deliberately never exported.
            "singers": [singer.to_dict() for singer in self.singers],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProjectPreset":
        if not isinstance(data, dict) or set(data) != {
            "preset_version", "singer_count", "room", "microphone", "singers",
        }:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设字段不符合 schema")
        if data["preset_version"] != PRESET_VERSION:
            raise ChoirError("PROJECT_VERSION_UNSUPPORTED", "不支持的预设版本")
        room_data = data["room"]
        room_keys = {
            "length_m", "width_m", "height_m", "rt60_s", "reverb_gain_db",
        }
        if not isinstance(room_data, dict) or set(room_data) != room_keys:
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设房间字段不符合 schema")
        room = RoomConfig(**room_data)
        room.validate()
        microphone = MicrophoneInput.from_dict(data["microphone"], room)
        if not isinstance(data["singers"], list):
            raise ChoirError("PROJECT_SCHEMA_ERROR", "预设歌手必须为列表")
        preset = cls(
            singer_count=data["singer_count"],
            room=room,
            microphone=microphone,
            singers=[SingerPreset.from_dict(item) for item in data["singers"]],
        )
        preset.validate()
        return preset


def save_preset(project: ProjectConfig, path: Path) -> Path:
    preset = ProjectPreset.from_project(project)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path.parent, path.name, preset.to_dict())
    except (OSError, TypeError, ValueError) as exc:
        raise ChoirError("OUTPUT_WRITE_FAILED", str(exc)) from exc
    return path


def load_preset(path: Path) -> ProjectPreset:
    if not path.is_file():
        raise ChoirError("PROJECT_NOT_FOUND", str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChoirError("PROJECT_PARSE_ERROR", str(exc)) from exc
    except OSError as exc:
        raise ChoirError("PROJECT_NOT_FOUND", str(exc)) from exc
    return ProjectPreset.from_dict(data)


def apply_preset(project: ProjectConfig, preset: ProjectPreset) -> None:
    """Apply singer positions/gains by track ID, preserving audio names and MIDI."""
    preset.validate()
    if len(project.tracks) != preset.singer_count:
        raise ChoirError(
            "PROJECT_SCHEMA_ERROR",
            f"预设需要 {preset.singer_count} 名歌手，当前工程有 {len(project.tracks)} 名",
        )
    room = RoomConfig(**asdict(preset.room))
    room.grid_step_m = min(project.room.grid_step_m, room.length_m, room.width_m)
    project.room = room
    project.microphone = preset.microphone
    source_by_id = {singer.track_id: singer for singer in preset.singers}
    project_track_ids = {track.track_id for track in project.tracks}
    unmatched_sources = [singer for singer in preset.singers if singer.track_id not in project_track_ids]
    unmatched_tracks = [track for track in project.tracks if track.track_id not in source_by_id]
    for track in project.tracks:
        singer = source_by_id.get(track.track_id)
        if singer is None:
            continue
        track.position = Position(
            singer.position.x_m, singer.position.y_m, singer.position.z_m,
        )
        track.gain_db = float(singer.gain_db)
    # Singer count is the only cross-project compatibility condition. If two
    # projects use different singer IDs, pair their remaining singers by order.
    for track, singer in zip(unmatched_tracks, unmatched_sources):
        track.position = Position(
            singer.position.x_m, singer.position.y_m, singer.position.z_m,
        )
        track.gain_db = float(singer.gain_db)
    project.validate()
