from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..ai import AIAudioInput, AIClient, AIConfig
from ..audio import preprocess_for_ai, preprocess_samples_for_ai
from ..errors import ChoirError
from ..models import ProjectConfig, midi_for_track
from ..render import Renderer
from ..naturalization import naturalize_track
from ..timbre_variation import generate_variations
from ..models import TimbreVariationConfig


class RenderWorker(QObject):
    progress = Signal(int, str)
    completed = Signal(object)
    notice = Signal(str)
    failed = Signal(object, str, object)

    def __init__(self, project: ProjectConfig, project_dir: Path, output_dir: Path,
                 job: str, accept_clip_risk: bool = False):
        super().__init__()
        self.project, self.project_dir, self.output_dir = project, project_dir, output_dir
        self.job, self.accept_clip_risk, self.renderer = job, accept_clip_risk, None

    @Slot()
    def run(self):
        try:
            self.renderer = Renderer(self.project, self.project_dir)
            callback = lambda percent, message: self.progress.emit(percent, message)
            if self.job == "preview":
                result = self.renderer.render_preview(self.output_dir, callback, self.accept_clip_risk)
            elif self.job == "stems":
                result = self.renderer.export_stems(self.output_dir, callback, self.accept_clip_risk)
            else:
                result = self.renderer.export_mix(self.output_dir, callback, self.accept_clip_risk)
            for message in self.renderer.warnings:
                self.notice.emit(message)
            self.completed.emit(result)
        except ChoirError as exc:
            self.failed.emit(exc, self.job, self.output_dir)
        except Exception as exc:
            self.failed.emit(ChoirError("RENDER_FAILED", str(exc)), self.job, self.output_dir)

    def cancel(self):
        if self.renderer:
            self.renderer.cancel()


class AIAnalysisWorker(QObject):
    completed = Signal(object)
    failed = Signal(object)

    def __init__(
        self, config: AIConfig, sources: list[tuple[str, str, Path]], project: ProjectConfig,
        project_dir: Path | None,
    ):
        super().__init__()
        self.config, self.sources, self.project, self.project_dir = config, sources, project, project_dir

    @Slot()
    def run(self):
        try:
            audio_inputs = []
            track_metadata = []
            for track_id, note, source in self.sources:
                assigned_midi = midi_for_track(track_id, self.project.naturalization)
                if (
                    self.project.naturalization.enabled
                    and self.project_dir is not None
                    and assigned_midi is not None
                ):
                    # AI hears the same dry timing layer as preview/export, but
                    # never room simulation, microphones, position or gain.
                    dry, _offsets = naturalize_track(
                        source, track_id, self.project.naturalization, self.project_dir,
                        self.project_dir / ".naturalization_cache",
                    )
                    wav_bytes, metadata = preprocess_samples_for_ai(dry)
                    metadata["naturalization_applied"] = True
                else:
                    wav_bytes, metadata = preprocess_for_ai(source)
                    metadata["naturalization_applied"] = False
                audio_inputs.append(AIAudioInput(track_id, note, wav_bytes))
                track_metadata.append({"track_id": track_id, "note": note, **metadata})
            response = AIClient(self.config).analyze_audio_json(audio_inputs, self.project)
            self.completed.emit((response, {
                "tracks": track_metadata,
                "clipped": any(bool(item["clipped"]) for item in track_metadata),
            }))
        except ChoirError as exc:
            self.failed.emit(exc)
        except Exception as exc:
            self.failed.emit(ChoirError("AI_REQUEST_FAILED", str(exc)))


class AIChatWorker(QObject):
    completed = Signal(object)
    failed = Signal(object)

    def __init__(self, config: AIConfig, project: ProjectConfig, messages: list[dict[str, str]]):
        super().__init__()
        self.config, self.project, self.messages = config, project, messages

    @Slot()
    def run(self):
        try:
            self.completed.emit(AIClient(self.config).customize_json(self.messages, self.project))
        except ChoirError as exc:
            self.failed.emit(exc)
        except Exception as exc:
            self.failed.emit(ChoirError("AI_REQUEST_FAILED", str(exc)))


class DuplicateWorker(QObject):
    completed = Signal(object)   # list[Path] — generated WAV paths
    progress = Signal(int, str)  # percent, message
    failed = Signal(object)      # ChoirError

    def __init__(
        self, source_path: Path, copy_count: int, output_dir: Path, preset_level: int = 3,
        midi_path: Path | None = None, midi_track_index: int | None = None,
        voice_style: str = "popular",
    ):
        super().__init__()
        self.source_path = source_path
        self.copy_count = copy_count
        self.output_dir = output_dir
        self.preset_level = preset_level
        self.midi_path = midi_path
        self.midi_track_index = midi_track_index
        self.voice_style = voice_style
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    @Slot()
    def run(self):
        try:
            paths = generate_variations(
                self.source_path,
                self.copy_count,
                self.output_dir,
                config=TimbreVariationConfig.from_preset(self.preset_level, self.voice_style),
                cancel_event=self._cancel,
                progress=lambda p, m: self.progress.emit(p, m),
                midi_path=self.midi_path,
                midi_track_index=self.midi_track_index,
            )
            self.completed.emit(paths)
        except ChoirError as exc:
            self.failed.emit(exc)
        except Exception as exc:
            self.failed.emit(ChoirError("RENDER_FAILED", str(exc)))
