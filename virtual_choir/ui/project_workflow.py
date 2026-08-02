from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from ..audio import read_source_wav
from ..errors import ChoirError
from ..models import Position, TrackConfig, resolve_source_path
from ..naturalization import resolve_midi_path
from ..models import midi_assignment_for_track
from ..project_io import (
    MEDIA_DIR, copy_to_media, load_project, rename_media_source, save_ai_json, save_project,
)
from ..presets import apply_preset, load_preset, save_preset
from .dialogs import DuplicateTrackDialog
from .theme import DEFAULT_PROJECT_DIR
from .tracks import BatchTrackDialog
from .workers import DuplicateWorker

PRESET_DIR = Path.cwd() / "presets"

class ProjectWorkflowMixin:
    def export_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出工程预设",
            str(PRESET_DIR / "virtual_choir_preset.json"),
            "虚拟合唱预设 (*.virtual-choir-preset.json *.json)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".virtual-choir-preset.json")
        try:
            save_preset(self.project, target)
        except ChoirError as exc:
            self.error(exc)
            return
        self._status.showMessage(f"已导出预设：{target.name}", 8000)
        self._toast.show_message("预设已导出", "success")

    def import_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入工程预设",
            str(PRESET_DIR),
            "虚拟合唱预设 (*.virtual-choir-preset.json *.json)",
        )
        if not path:
            return
        try:
            preset = load_preset(Path(path))
            apply_preset(self.project, preset)
        except ChoirError as exc:
            self.error(exc)
            return
        self.changed(room=True, tracks=True, selection=True)
        self.room_view.fit_room()
        self._status.showMessage("预设已导入", 8000)
        self._toast.show_message("预设已导入", "success")

    def rename_track_source(self, track_id: str):
        if not self.project_dir:
            QMessageBox.information(self, "请先保存工程", "保存工程后才能重命名 Media 文件。")
            return
        track = self._track(track_id)
        new_name, accepted = QInputDialog.getText(
            self, "重命名 Media 文件", "新文件名（无需输入 .wav）：", text=Path(track.file_name).stem,
        )
        if not accepted or new_name.strip() == Path(track.file_name).stem:
            return
        try:
            target, affected = rename_media_source(
                self.project_dir, self.project, track_id, new_name,
            )
            if self.project.ai_recommendations:
                save_ai_json(
                    self.project_dir, "ai_suggestion.json",
                    {"recommendations": self.project.ai_recommendations},
                )
        except ChoirError as exc:
            self.error(exc)
            return
        self.changed(tracks=True, selection=True)
        self._status.showMessage(
            f"已重命名为 {target.name}，同步 {len(affected)} 条轨道引用", 8000,
        )

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入 WAV", "", "WAV 音频 (*.wav *.WAV)"
        )
        self._import_paths([Path(p) for p in paths])

    def _import_paths(self, paths: list[Path]):
        if paths and not self.project_dir and not self._ensure_project_saved("导入音频"):
            return
        results = []
        for path in paths:
            try:
                read_source_wav(path)
                # Copy the external WAV into the project Media directory so the
                # project is self-contained.
                if self.project_dir:
                    media_path = copy_to_media(self.project_dir, path)
                    self.project.add_track(media_path)
                results.append(f"{path.name}：已导入")
            except ChoirError as exc:
                results.append(f"{path.name}：{exc}")
        if paths:
            imported = len([result for result in results if result.endswith("已导入")])
            if imported:
                self.changed(tracks=True, selection=True)
                self._toast.show_message(f"已导入 {imported} 个文件", "success")

    def dragEnterEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        if any(path.suffix.lower() in {".wav", ".json"} for path in paths):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        project_paths = [path for path in paths if path.suffix.lower() == ".json"]
        if project_paths:
            self._open_project_path(project_paths[0])
        else:
            self._import_paths([path for path in paths if path.suffix.lower() == ".wav"])
        event.acceptProposedAction()

    def remove_track(self, track_id: str):
        source_track = self._track(track_id)
        source_references = {
            value for value in (
                source_track.track_id, source_track.file_name, source_track.source_path,
            ) if value
        }
        dependent = [
            track for track in self.project.tracks
            if track.parent_source and track.parent_source in source_references
        ]
        message = f"只会移除 {track_id} 的工程引用，不会删除源文件。"
        if dependent:
            message += f"\n\n注意：有 {len(dependent)} 个差异化副本仍会保留在工程中。"
        if QMessageBox.question(
            self, "移除轨道",
            f"{message}\n\n是否继续？"
        ) == QMessageBox.StandardButton.Yes:
            self.project.tracks = [t for t in self.project.tracks if t.track_id != track_id]
            for assignment in self.project.naturalization.assignments:
                assignment.track_ids = [value for value in assignment.track_ids if value != track_id]
            if self._selected_track_id == track_id:
                self._selected_track_id = None
            self.changed(tracks=True, selection=True)

    def batch_tracks(self):
        """Apply enable, disable, or removal to a checked set of tracks."""
        if not self.project.tracks:
            QMessageBox.information(self, "没有轨道", "请先导入至少一条音频轨道。")
            return

        dialog = BatchTrackDialog(self.project.tracks, self)
        if not dialog.exec() or not dialog.operation:
            return

        selected_ids = set(dialog.selected_track_ids())
        selected_count = len(selected_ids)
        if dialog.operation == "delete":
            if QMessageBox.question(
                self,
                "批量移除轨道",
                f"将从工程中移除 {selected_count} 条轨道引用，音频文件不会被删除。是否继续？",
            ) != QMessageBox.StandardButton.Yes:
                return
            self.project.tracks = [track for track in self.project.tracks if track.track_id not in selected_ids]
            for assignment in self.project.naturalization.assignments:
                assignment.track_ids = [
                    value for value in assignment.track_ids if value not in selected_ids
                ]
            action_text = "已移除"
        else:
            enabled = dialog.operation == "enable"
            for track in self.project.tracks:
                if track.track_id in selected_ids:
                    track.enabled = enabled
            action_text = "已启用" if enabled else "已禁用"

        self._selected_track_id = None
        self.changed(tracks=True, selection=True)
        message = f"{action_text} {selected_count} 条轨道"
        self._status.showMessage(message, 8000)
        self._toast.show_message(message, "success")

    def duplicate_track(self, track_id: str):
        """Open the duplicate dialog and kick off background generation."""
        try:
            source_track = self._track(track_id)
        except StopIteration:
            self._status.showMessage(f"未找到轨道 {track_id}", 5000)
            return

        midi_assignment = midi_assignment_for_track(track_id, self.project.naturalization)
        assigned_midi = midi_assignment.midi_path if midi_assignment else None
        if not assigned_midi:
            notice = QMessageBox(self)
            notice.setIcon(QMessageBox.Icon.Information)
            notice.setWindowTitle("建议导入 MIDI")
            notice.setText("当前轨道未分配 MIDI。")
            notice.setInformativeText(
                "导入 MIDI 后可按音符边界生成更自然的音高线和颤音行为；"
                "也可以继续，程序会使用无 MIDI 的平滑随机音高线。"
            )
            import_button = notice.addButton("导入 MIDI", QMessageBox.ButtonRole.ActionRole)
            continue_button = notice.addButton("仍然继续", QMessageBox.ButtonRole.AcceptRole)
            notice.exec()
            if notice.clickedButton() is import_button:
                self.choose_reference_midi()
                midi_assignment = midi_assignment_for_track(track_id, self.project.naturalization)
                assigned_midi = midi_assignment.midi_path if midi_assignment else None
            elif notice.clickedButton() is not continue_button:
                return
        dialog = DuplicateTrackDialog(
            source_track, self.project_dir, assigned_midi, self,
        )
        if not dialog.exec():
            return

        opts = dialog.result()
        if False and not assigned_midi:
            notice = QMessageBox(self)
            notice.setIcon(QMessageBox.Icon.Information)
            notice.setWindowTitle("未分配 MIDI")
            notice.setText("当前轨道未分配 MIDI。")
            notice.setInformativeText(
                "仍可继续生成差异化副本，但将使用无 MIDI 的平滑随机音高线，"
                "无法按音符应用逐音符颤音行为。"
            )
            notice.setStandardButtons(QMessageBox.StandardButton.Ok)
            notice.exec()
        if not self._ensure_project_saved("生成差异化副本"):
            return
        try:
            source_path = resolve_source_path(
                self.project_dir, source_track.source_path or source_track.file_name
            )
        except ChoirError as exc:
            self.error(exc)
            return

        if not source_path.is_file():
            self.error(ChoirError("AUDIO_NOT_FOUND", str(source_path)))
            return

        vibrato_midi_path = None
        if assigned_midi:
            try:
                vibrato_midi_path = resolve_midi_path(
                    self.project_dir, assigned_midi,
                )
            except ChoirError:
                # The duplication pipeline deliberately has a no-MIDI fallback.
                vibrato_midi_path = None

        output_dir = self.project_dir / MEDIA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Disable UI during generation
        self._set_duplicate_ui_enabled(False)
        self._overlay.show_overlay("正在生成差异化副本…")
        self.progress.setValue(0)
        self._progress_label.setText("准备生成副本…")

        self.stop_button.setEnabled(True)

        # Keep this immutable source snapshot on the window.  The completion
        # slot must be a QObject-bound method so Qt queues it onto the GUI
        # thread; a Python lambda has no receiver thread affinity.
        self._dup_source_track = deepcopy(source_track)
        self._dup_preset_level = opts["preset_level"]
        self._dup_voice_style = opts["voice_style"]
        self.dup_thread = QThread(self)
        self.dup_worker = DuplicateWorker(
            source_path, opts["copy_count"], output_dir, opts["preset_level"],
            vibrato_midi_path, midi_assignment.midi_track_index if midi_assignment else None,
            voice_style=opts["voice_style"],
        )
        self.dup_worker.moveToThread(self.dup_thread)
        # Worker signals are always delivered to MainWindow on the GUI thread.
        # Do not connect these signals to lambdas: lambdas do not have QObject
        # thread affinity, so PySide may execute them in the worker thread.
        self.dup_thread.started.connect(self.dup_worker.run)
        self.dup_worker.progress.connect(
            self._task_callbacks.duplicate_progress, Qt.ConnectionType.QueuedConnection
        )
        self.dup_worker.completed.connect(
            self._task_callbacks.duplicate_completed, Qt.ConnectionType.QueuedConnection
        )
        self.dup_worker.failed.connect(
            self._task_callbacks.duplicate_failed, Qt.ConnectionType.QueuedConnection
        )
        self.dup_worker.completed.connect(self.dup_thread.quit, Qt.ConnectionType.QueuedConnection)
        self.dup_worker.failed.connect(self.dup_thread.quit, Qt.ConnectionType.QueuedConnection)
        self.dup_worker.completed.connect(self.dup_worker.deleteLater)
        self.dup_worker.failed.connect(self.dup_worker.deleteLater)
        self.dup_thread.finished.connect(self._dup_cleanup, Qt.ConnectionType.QueuedConnection)
        self.dup_thread.start()

    def _dup_progress(self, percent: int, message: str):
        self.progress.setValue(percent)
        self._progress_label.setText(message)
        self._overlay.set_text(f"{message} ({percent}%)")

    @Slot(object)
    def _dup_completed(self, paths: list[Path]):
        """Register each generated copy as a new Track."""
        source_track = self._dup_source_track
        if source_track is None:
            self._dup_failed(ChoirError("RENDER_FAILED", "Missing duplicate source track"))
            return

        self._overlay.hide_overlay()
        added = 0
        for idx, wav_path in enumerate(paths, start=1):
            try:
                track = TrackConfig(
                    track_id=f"singer_{self.project.next_track_sequence}",
                    file_name=wav_path.name,
                    position=Position(
                        source_track.position.x_m,
                        source_track.position.y_m,
                        source_track.position.z_m,
                    ),
                    gain_db=source_track.gain_db,
                    enabled=True,
                    source_path=str(wav_path.resolve()),
                    parent_source=source_track.source_path or str(
                        resolve_source_path(
                            self.project_dir,
                            source_track.source_path or source_track.file_name,
                        )
                    ),
                    copy_index=idx,
                    variation_preset=self._dup_preset_level,
                    variation_style=self._dup_voice_style,
                )
                track.validate(self.project.room)
                self.project.tracks.append(track)
                self.project.next_track_sequence += 1
                added += 1
            except ChoirError as exc:
                self._status.showMessage(f"副本 {idx} 注册失败：{exc}", 8000)

        self.changed(tracks=True, selection=True)
        self._status.showMessage(f"已生成 {added} 份差异化副本", 8000)
        self._toast.show_message(f"已生成 {added} 份差异化副本", "success")

        self.progress.setValue(100)
        self._progress_label.setText("准备就绪")

    def _dup_failed(self, error: ChoirError):
        self._overlay.hide_overlay()
        self._set_duplicate_ui_enabled(True)
        self._progress_label.setText("准备就绪")
        if error.code == "RENDER_CANCELLED":
            self._status.showMessage("差异化生成已取消", 8000)
            self._toast.show_message("差异化生成已取消", "info")
        else:
            self.error(error)

    def _dup_cleanup(self):
        thread = self.dup_thread
        self.dup_worker = None
        self.dup_thread = None
        self._dup_source_track = None
        self._dup_preset_level = None
        self._dup_voice_style = None
        self._set_duplicate_ui_enabled(True)
        self.stop_button.setEnabled(False)
        self._overlay.hide_overlay()
        if thread is not None:
            thread.deleteLater()

    def _set_duplicate_ui_enabled(self, enabled: bool):
        self.preview_button.setEnabled(enabled)
        self.stems_button.setEnabled(enabled)
        self.mix_button.setEnabled(enabled)
        self.import_action.setEnabled(enabled)
        self.open_action.setEnabled(enabled)
        self.save_action.setEnabled(enabled)
        self.export_preset_action.setEnabled(enabled)
        self.import_preset_action.setEnabled(enabled)
        self.analyze_action.setEnabled(enabled)
        self.customize_ai_action.setEnabled(enabled)
        self._track_panel.setEnabled(enabled)
        self.room_view.setEnabled(enabled)

    def cancel_duplicate(self):
        if self.dup_worker is None or self.dup_thread is None or not self.dup_thread.isRunning():
            return
        self.stop_button.setEnabled(False)
        self.dup_worker.cancel()
        self._overlay.set_text("正在取消差异化生成…")
        self._progress_label.setText("取消中…")
        self._status.showMessage("正在请求取消差异化生成…")

    def _ensure_project_saved(self, operation: str) -> bool:
        """Prompt for a project directory, then persist before a file-producing job."""
        if self.project_dir:
            return True

        directory = QFileDialog.getExistingDirectory(
            self,
            f"保存工程后继续{operation}",
            str(DEFAULT_PROJECT_DIR),
        )
        if not directory:
            return False

        try:
            self.project_dir = Path(directory)
            save_project(self.project, self.project_dir)
            self.dirty = False
            self.refresh(room=True, tracks=True, selection=True)
            self._status.showMessage(f"工程已保存，继续{operation}", 5000)
            self._toast.show_message("工程已保存", "success")
            return True
        except ChoirError as exc:
            self.project_dir = None
            self.error(exc)
            return False

    def save(self):
        if not self.project_dir:
            directory = QFileDialog.getExistingDirectory(
                self, "选择工程目录", str(DEFAULT_PROJECT_DIR)
            )
            if not directory:
                return
            self.project_dir = Path(directory)
        try:
            save_project(self.project, self.project_dir)
            self.dirty = False
            self.refresh()
            self._status.showMessage("工程已保存", 5000)
            self._toast.show_message("工程已保存", "success")
        except ChoirError as exc:
            self.error(exc)

    def save_as(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择工程目录", str(self.project_dir or DEFAULT_PROJECT_DIR)
        )
        if not directory:
            return
        self.project_dir = Path(directory)
        self.save()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开工程", str(DEFAULT_PROJECT_DIR), "工程 JSON (project_config.json *.json)"
        )
        if not path:
            return
        self._open_project_path(Path(path))

    def _open_project_path(self, path: Path):
        try:
            self.project = load_project(path)
            self.project_dir = path.parent
            self.dirty = False
            self._ai_undo_project = None
            self.undo_ai_action.setEnabled(False)
            self._sync_ai_recommendations()
            self._project_revision += 1
            self.refresh(room=True, tracks=True, selection=True, preview_sources=True)
            self.room_view.fit_room()
            self._status.showMessage("工程已加载", 5000)
            self._toast.show_message("工程已加载", "info")
        except ChoirError as exc:
            self.error(exc)

    def _reset_layout(self):
        self._splitter.setSizes([320, 800, 320])

    def _toggle_left_panel(self):
        self._left_panel.setVisible(not self._left_panel.isVisible())
        self.toggle_left_action.setChecked(self._left_panel.isVisible())

    def _toggle_right_panel(self):
        self._track_panel.setVisible(not self._track_panel.isVisible())
        self.toggle_right_action.setChecked(self._track_panel.isVisible())
