from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import read_source_wav, write_export, write_preview
from .errors import ChoirError
from .models import ProjectConfig, TrackConfig, midi_for_track, resolve_source_path
from .naturalization import naturalize_track, resolve_midi_path

RENDERER_VERSION = "0.2.0"
Progress = Callable[[int, str], None]


class Renderer:
    def __init__(self, project: ProjectConfig, project_dir: Path, cache_dir: Path | None = None):
        project.validate(project_dir, require_sources=True, require_track=True)
        self.project, self.project_dir = project, project_dir
        self.cache_dir = cache_dir or project_dir / ".render_cache"
        self.cancel_event = threading.Event()
        self.warnings: list[str] = []

    def cancel(self) -> None: self.cancel_event.set()

    def render_preview(self, output_dir: Path, progress: Progress | None = None, accept_clip_risk=False) -> Path:
        self._check_disk_space(output_dir)
        stereo = self._mix(progress)
        self._check_peak(stereo, accept_clip_risk)
        target = output_dir / "preview.wav"
        write_preview(target, stereo)

        # Keep listening stems beside the mix so the preview transport can
        # switch tracks without requiring a separate export operation.
        tracks = [track for track in self.project.tracks if track.enabled]
        for index, track in enumerate(tracks):
            self._check_cancel()
            write_preview(output_dir / "stems" / f"{track.track_id}.wav", self._render_track(track))
            self._report(progress, 95 + round((index + 1) / len(tracks) * 4), f"正在生成分轨预览 {track.track_id}")
        self._report(progress, 100, "预览已生成")
        return target

    def export_stems(self, output_dir: Path, progress: Progress | None = None, accept_clip_risk=False) -> list[Path]:
        self._check_disk_space(output_dir)
        tracks = [t for t in self.project.tracks if t.enabled]; result: list[Path] = []
        for index, track in enumerate(tracks):
            self._check_cancel(); stereo = self._render_track(track); self._check_peak(stereo, accept_clip_risk)
            target = output_dir / f"{track.track_id}.wav"; write_export(target, stereo); result.append(target)
            self._report(progress, round((index + 1) / len(tracks) * 100), f"已导出 {track.track_id}")
        return result

    def export_mix(self, output_dir: Path, progress: Progress | None = None, accept_clip_risk=False) -> Path:
        self._check_disk_space(output_dir)
        stereo = self._mix(progress); self._check_peak(stereo, accept_clip_risk)
        target = output_dir / "mix.wav"; write_export(target, stereo); self._report(progress, 100, "混音已导出")
        return target

    def _mix(self, progress: Progress | None) -> np.ndarray:
        tracks = [t for t in self.project.tracks if t.enabled]; mix: np.ndarray | None = None
        for index, track in enumerate(tracks):
            self._check_cancel(); rendered = self._render_track(track)
            if mix is None: mix = rendered
            else:
                if len(rendered) > len(mix): mix = np.pad(mix, ((0, len(rendered)-len(mix)), (0, 0)))
                mix[:len(rendered)] += rendered
            self._report(progress, round((index + 1) / len(tracks) * 95), f"正在渲染 {track.track_id}")
        if mix is None:
            return np.empty((0, 2), dtype=np.float32)
        # This is a post-mix bus control: it affects preview/mix output only,
        # never individual stem renders or the AI analysis request.
        return mix * np.float32(10 ** (self.project.room.bus_gain_db / 20))

    def _render_track(self, track: TrackConfig) -> np.ndarray:
        key = self._cache_key(track); cached = self.cache_dir / f"{key}.npy"
        if cached.is_file(): return np.load(cached).astype(np.float32)
        source_path = resolve_source_path(self.project_dir, track.source_path or track.file_name)
        source = self._render_source(track, source_path)
        rirs = self._rirs(track)
        microphone_channels = [_fft_convolve(source, rir) for rir in rirs]
        stereo = _microphone_channels_to_stereo(microphone_channels) * np.float32(10 ** (track.gain_db / 20))
        self.cache_dir.mkdir(parents=True, exist_ok=True); np.save(cached, stereo)
        return stereo

    def _render_source(self, track: TrackConfig, source_path: Path) -> np.ndarray:
        """Read the local dry layer, preserving the raw WAV as a safe fallback."""
        config = self.project.naturalization
        if not config.enabled:
            return read_source_wav(source_path)
        if midi_for_track(track.track_id, config) is None:
            return read_source_wav(source_path)
        try:
            processed, offsets = naturalize_track(
                source_path, track.track_id, config, self.project_dir,
                self.project_dir / ".naturalization_cache", self.cancel_event,
            )
            if offsets:
                self.warnings.append(f"{track.track_id}：已应用 {len(offsets)} 个音符随机偏移")
            return processed
        except ChoirError as exc:
            if exc.code == "RENDER_CANCELLED":
                raise
            self.warnings.append(f"{track.track_id}：{exc.message}，本次已使用原始 WAV 继续渲染。")
            return read_source_wav(source_path)

    def _rirs(self, track: TrackConfig) -> list[np.ndarray]:
        try: import pyroomacoustics as pra
        except ImportError as exc: raise ChoirError("RENDER_DEPENDENCY_MISSING", "请在 py311_env 执行 pip install pyroomacoustics") from exc
        try:
            dimensions = [self.project.room.width_m, self.project.room.length_m, self.project.room.height_m]
            absorption, max_order = pra.inverse_sabine(self.project.room.rt60_s, dimensions)
            room = pra.ShoeBox(dimensions, fs=48000, materials=pra.Material(absorption), max_order=max_order)
            room.add_source([track.position.x_m, track.position.y_m, track.position.z_m])
            mic = self.project.microphone_positions()
            room.add_microphone_array(np.array([
                [item["x_m"] for item in mic],
                [item["y_m"] for item in mic],
                [item["z_m"] for item in mic],
            ], dtype=float))
            room.compute_rir(); rirs = [np.asarray(room.rir[i][0], dtype=np.float32) for i in range(2)]
            # keep direct sound; apply requested gain only to the reflected tail
            linear = 10 ** (self.project.room.reverb_gain_db / 20)
            for rir in rirs:
                direct = int(np.argmax(np.abs(rir)))
                rir[direct + 1:] *= linear
            return rirs
        except ChoirError: raise
        except Exception as exc: raise ChoirError("RENDER_FAILED", f"RIR 生成失败：{exc}") from exc

    def _cache_key(self, track: TrackConfig) -> str:
        source = resolve_source_path(self.project_dir, track.source_path or track.file_name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        naturalization = None
        if self.project.naturalization.enabled:
            assignments = []
            for assignment in self.project.naturalization.assignments:
                try:
                    midi = resolve_midi_path(self.project_dir, assignment.midi_path)
                    midi_sha256 = hashlib.sha256(midi.read_bytes()).hexdigest()
                except ChoirError:
                    midi_sha256 = "unavailable"
                assignments.append({
                    "midi_path": assignment.midi_path,
                    "midi_sha256": midi_sha256,
                    "track_ids": sorted(assignment.track_ids),
                })
            naturalization = {
                "config": asdict(self.project.naturalization),
                "assignments": assignments,
            }
        payload = {"source_sha256": digest, "naturalization": naturalization, "room": self.project.room.__dict__, "microphones": self.project.microphone_positions(), "position": track.position.__dict__, "gain_db": track.gain_db, "renderer_version": RENDERER_VERSION}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _check_peak(self, stereo: np.ndarray, accepted: bool) -> None:
        if stereo.size and float(np.max(np.abs(stereo))) > 1.0 and not accepted: raise ChoirError("AUDIO_CLIP_RISK", "请确认允许削波风险后重试")

    def _check_disk_space(self, output_dir: Path) -> None:
        """Ensure a conservative output estimate plus the contractual 2 GB margin."""
        output_dir.mkdir(parents=True, exist_ok=True)
        total_samples = sum(resolve_source_path(self.project_dir, t.source_path or t.file_name).stat().st_size for t in self.project.tracks if t.enabled)
        # Float WAV source bytes underestimate RIR tails; multiply by four and reserve 2 GiB.
        required = total_samples * 4 + 2 * 1024**3
        if shutil.disk_usage(output_dir).free < required: raise ChoirError("DISK_SPACE_LOW", f"至少需要 {required / 1024**3:.2f} GB 可用空间")

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set(): raise ChoirError("RENDER_CANCELLED")

    @staticmethod
    def _report(callback: Progress | None, percent: int, message: str) -> None:
        if callback: callback(max(0, min(100, percent)), message)


def _fft_convolve(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Use CUDA torch when available; NumPy FFT is the deterministic CPU fallback."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            n = len(source) + len(rir) - 1; size = 1 << (n - 1).bit_length()
            a = torch.as_tensor(source, device=device); b = torch.as_tensor(rir, device=device)
            out = torch.fft.irfft(torch.fft.rfft(a, n=size) * torch.fft.rfft(b, n=size), n=size)[:n]
            return out.cpu().numpy().astype(np.float32)
    except Exception:
        pass
    n = len(source) + len(rir) - 1; size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(source, size) * np.fft.rfft(rir, size), size)[:n].astype(np.float32)


def _stereo_channels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pad unequal RIR tails before building a two-channel render buffer."""
    length = max(len(left), len(right))
    if len(left) != length:
        left = np.pad(left, (0, length - len(left)))
    if len(right) != length:
        right = np.pad(right, (0, length - len(right)))
    return np.column_stack((left, right)).astype(np.float32)


def _microphone_channels_to_stereo(channels: list[np.ndarray]) -> np.ndarray:
    """Equal-power pan an ordered microphone array into a stereo deliverable."""
    if len(channels) < 2:
        raise ValueError("at least two microphone channels are required")
    length = max(len(channel) for channel in channels)
    padded = [
        np.pad(channel, (0, length - len(channel))) if len(channel) != length else channel
        for channel in channels
    ]
    pan = np.linspace(-1.0, 1.0, len(padded), dtype=np.float64)
    left_weights = np.sqrt((1.0 - pan) / 2.0)
    right_weights = np.sqrt((1.0 + pan) / 2.0)
    left = sum(channel * weight for channel, weight in zip(padded, left_weights)) / left_weights.sum()
    right = sum(channel * weight for channel, weight in zip(padded, right_weights)) / right_weights.sum()
    return np.column_stack((left, right)).astype(np.float32)
