from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .errors import ChoirError
from .models import ProjectConfig, resolve_source_path


# ---------------------------------------------------------------------------
# Subfolder names
# ---------------------------------------------------------------------------

AI_OUTPUT_DIR = "AI_Output"        # approved_config.json / ai_suggestion.json
MEDIA_DIR = "Media"                 # copied MIDI / WAV / generated copies


def _ensure_subdir(project_dir: Path, name: str) -> Path:
    """Return ``project_dir / name``, creating it when it does not exist."""
    target = project_dir / name
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Project save / load (unchanged contract)
# ---------------------------------------------------------------------------


def save_project(project: ProjectConfig, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    project.validate()
    target = directory / "project_config.json"
    _atomic_json(target, project.to_dict())
    return target


def load_project(path: Path) -> ProjectConfig:
    if not path.is_file(): raise ChoirError("PROJECT_NOT_FOUND", str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise ChoirError("PROJECT_PARSE_ERROR", str(exc)) from exc
    except OSError as exc: raise ChoirError("PROJECT_NOT_FOUND", str(exc)) from exc
    project = ProjectConfig.from_dict(data); project.validate(path.parent, require_sources=True)
    return project


# ---------------------------------------------------------------------------
# Generic JSON persistence
# ---------------------------------------------------------------------------


def save_json(directory: Path, filename: str, data: dict) -> Path:
    target = directory / filename; _atomic_json(target, data); return target


# ---------------------------------------------------------------------------
# AI output – written into the AI_Output subfolder so the project root
# stays clean and the user knows what to open.
# ---------------------------------------------------------------------------


def save_ai_json(project_dir: Path, filename: str, data: dict) -> Path:
    """Persist AI-related JSON into ``<project_dir>/AI_Output/<filename>``."""
    return save_json(_ensure_subdir(project_dir, AI_OUTPUT_DIR), filename, data)


# ---------------------------------------------------------------------------
# Media helpers – copy external files into the project Media directory
# so everything the project needs is self-contained.
# ---------------------------------------------------------------------------


def copy_to_media(project_dir: Path, source_path: Path) -> Path:
    """Copy *source_path* into ``<project_dir>/Media/``, preserving the
    original filename.  Returns the absolute path of the copy.

    If a file with the same name already exists and its content matches
    *source_path* (SHA-256), the existing file is reused.  Otherwise a
    unique name is chosen by appending a numeric suffix.
    """
    import hashlib

    media = _ensure_subdir(project_dir, MEDIA_DIR)
    dest = media / source_path.name

    # Fast path: identical file already present
    if dest.is_file():
        try:
            existing_digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            new_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if existing_digest == new_digest:
                return dest.resolve()
        except OSError:
            pass

    # Find a free name
    if not dest.exists():
        shutil.copy2(source_path, dest)
        return dest.resolve()

    stem, suffix = source_path.stem, source_path.suffix
    idx = 1
    while True:
        candidate = media / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(source_path, candidate)
            return candidate.resolve()
        idx += 1


def rename_media_source(
    project_dir: Path, project: ProjectConfig, track_id: str, new_name: str,
) -> tuple[Path, set[str]]:
    """Rename one physical Media WAV and synchronize every project reference."""
    track = next((item for item in project.tracks if item.track_id == track_id), None)
    if track is None:
        raise ChoirError("AUDIO_NOT_FOUND", track_id)
    new_name = new_name.strip()
    if Path(new_name).suffix == "":
        new_name = f"{new_name}.wav"
    if (
        not new_name or Path(new_name).name != new_name or len(new_name) > 255
        or Path(new_name).suffix.lower() != ".wav"
    ):
        raise ChoirError("OUTPUT_WRITE_FAILED", "新文件名必须是有效的 .wav 文件名，且不能包含目录")
    source = resolve_source_path(project_dir, track.source_path or track.file_name)
    media = (project_dir / MEDIA_DIR).resolve()
    if source.parent.resolve() != media:
        raise ChoirError("OUTPUT_WRITE_FAILED", "只能重命名工程 Media 目录中的源文件")
    target = (media / new_name).resolve()
    if target.parent != media:
        raise ChoirError("OUTPUT_WRITE_FAILED", "新文件名超出 Media 目录")
    if target == source:
        return source, {track_id}
    if target.exists():
        raise ChoirError("OUTPUT_WRITE_FAILED", f"Media 中已存在 {new_name}")

    affected: set[str] = set()
    old_resolved = source.resolve()
    for item in project.tracks:
        try:
            item_source = resolve_source_path(project_dir, item.source_path or item.file_name)
        except ChoirError:
            continue
        if item_source.resolve() == old_resolved:
            affected.add(item.track_id)
    try:
        source.rename(target)
    except OSError as exc:
        raise ChoirError("OUTPUT_WRITE_FAILED", str(exc)) from exc

    target_text = str(target)
    for item in project.tracks:
        if item.track_id in affected:
            item.file_name = target.name
            item.source_path = target_text
        if item.parent_source:
            try:
                parent = resolve_source_path(project_dir, item.parent_source)
            except ChoirError:
                parent = None
            if parent is not None and parent.resolve() == old_resolved:
                item.parent_source = target_text
    names = {item.track_id: item.file_name for item in project.tracks}
    for recommendation in project.ai_recommendations:
        for singer in recommendation.get("singers", []):
            if singer.get("track_id") in names:
                singer["note"] = names[singer["track_id"]]
    return target, affected


# ---------------------------------------------------------------------------
# Atomic write (internal)
# ---------------------------------------------------------------------------


def _atomic_json(target: Path, data: dict) -> None:
    try:
        handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, text=True)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
        os.replace(temp_name, target)
    except (OSError, TypeError, ValueError) as exc:
        try:
            if "temp_name" in locals(): Path(temp_name).unlink(missing_ok=True)
        finally: raise ChoirError("OUTPUT_WRITE_FAILED", str(exc)) from exc
