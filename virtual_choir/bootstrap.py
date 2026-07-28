"""First-run workspace and PC-NSF-HiFiGAN installation helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_DIR_NAME = "projects"
PRESETS_DIR_NAME = "presets"
MODEL_DIR_NAME = "models"
VOCODER_REPOSITORY_NAME = "SingingVocoders"
CHECKPOINT_NAME = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt"

_SOURCE_ARCHIVE_URL = (
    "https://github.com/openvpi/SingingVocoders/archive/refs/heads/main.zip"
)
_CHECKPOINT_ARCHIVE_URL = (
    "https://github.com/openvpi/SingingVocoders/releases/download/v1.0.0/"
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
)
_USER_AGENT = "VirtualChoir/0.1 (+https://github.com/3253473591/VirtualChoir)"


class WorkspaceInitializationError(RuntimeError):
    """Raised when the first-run model installation cannot complete."""


def workspace_root() -> Path:
    """Return the directory from which ``python -m virtual_choir`` was run."""
    return Path.cwd().resolve()


def vocoder_paths(root: Path | None = None) -> tuple[Path, Path]:
    """Return the installed model-source and checkpoint paths for a workspace."""
    model_root = (root or workspace_root()) / MODEL_DIR_NAME / VOCODER_REPOSITORY_NAME
    return (
        model_root / "models" / "nsf_HiFigan" / "models.py",
        model_root / "checkpoints" / CHECKPOINT_NAME,
    )


def initialize_workspace(root: Path | None = None) -> None:
    """Create the user workspace and install the vocoder once when necessary.

    Downloads are only attempted when a required model file is missing.  Archive
    extraction is constrained to the target directory so a remote archive can
    never write outside the user-selected workspace.
    """
    root = (root or workspace_root()).resolve()
    (root / PROJECT_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (root / PRESETS_DIR_NAME).mkdir(parents=True, exist_ok=True)

    model_source, checkpoint = vocoder_paths(root)
    if model_source.is_file() and checkpoint.is_file():
        return

    try:
        if not model_source.is_file():
            _install_model_source(model_source)
        if not checkpoint.is_file():
            _install_checkpoint(checkpoint)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        raise WorkspaceInitializationError(
            f"无法下载 PC-NSF-HiFiGAN 模型：{exc}"
        ) from exc


def _install_model_source(model_source: Path) -> None:
    with _download_archive(_SOURCE_ARCHIVE_URL, model_source.parent) as archive:
        # GitHub archives have a revision-prefixed top-level directory.  Keep
        # only its contents so the runtime path remains stable across commits.
        _extract_archive(archive, model_source.parents[2], strip_first_component=True)
    if not model_source.is_file():
        raise WorkspaceInitializationError("PC-NSF-HiFiGAN 源码归档不包含 models.py")


def _install_checkpoint(checkpoint: Path) -> None:
    with _download_archive(_CHECKPOINT_ARCHIVE_URL, checkpoint.parent) as archive:
        with zipfile.ZipFile(archive) as bundle:
            member = next(
                (item for item in bundle.infolist() if Path(item.filename).name == CHECKPOINT_NAME),
                None,
            )
            if member is None:
                raise WorkspaceInitializationError("PC-NSF-HiFiGAN 权重归档不包含预期 checkpoint")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{checkpoint.name}.", suffix=".tmp",
                dir=checkpoint.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(source, stream)
            os.replace(temporary, checkpoint)


class _download_archive:
    def __init__(self, url: str, parent: Path):
        self.url, self.parent = url, parent
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".virtual_choir_download_", suffix=".zip", dir=self.parent)
        self.path = Path(name)
        try:
            with os.fdopen(handle, "wb") as output, urllib.request.urlopen(
                urllib.request.Request(self.url, headers={"User-Agent": _USER_AGENT}), timeout=120,
            ) as response:
                shutil.copyfileobj(response, output)
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        return self.path

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def _extract_archive(archive: Path, target: Path, *, strip_first_component: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = PurePosixPath(member.filename).parts
            if strip_first_component:
                parts = parts[1:]
            if not parts or member.is_dir() or any(part in {"", ".", ".."} for part in parts):
                continue
            destination = target.joinpath(*parts)
            resolved = destination.resolve()
            if target.resolve() not in resolved.parents:
                raise WorkspaceInitializationError("模型归档包含非法路径")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(source, stream)
            os.replace(temporary, destination)
