from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout

from ..errors import ChoirError
from ..models import MidiAssignment, Position, validate_position
from ..naturalization import inspect_midi_tracks, resolve_midi_path
from ..project_io import copy_to_media
from .theme import DEFAULT_PROJECT_DIR
from .dialogs import MidiAssignmentDialog
from .widgets import MidiPathEdit

class WindowStateMixin:
    def refresh(
        self,
        *,
        room: bool = False,
        tracks: bool = False,
        selection: bool = False,
        preview_sources: bool = False,
        title: bool = True,
    ):
        """Refresh only the UI areas affected by a project change."""
        self._updating = True
        self._sync_naturalization_status()
        self._sync_track_dependent_controls()
        if room:
            self._sync_room_controls()
            self.room_view.set_project(self.project)
        else:
            self.room_view.update_tracks(self.project)
        if tracks:
            self._track_panel.refresh(self.project)
        else:
            for track in self.project.tracks:
                self._track_panel.update_track(track)
        if selection:
            self._track_panel.select_track(self._selected_track_id)
            self.room_view.select_singer(self._selected_track_id)
        if preview_sources and self.project_dir:
            self._refresh_preview_sources(self.project_dir)
        self._updating = False
        if title:
            self.setWindowTitle(f"虚拟合唱团{' *' if self.dirty else ''}")

    def _show_shortcuts(self):
        actions = (
            self.import_action, self.open_action, self.save_action, self.analyze_action,
        )
        lines = [
            f"{action.shortcut().toString()}\t{action.text()}"
            for action in actions if not action.shortcut().isEmpty()
        ]
        lines.append("空格\t播放/暂停预览")
        lines.append("滚轮\t缩放房间视图")
        QMessageBox.information(self, "快捷键", "\n".join(lines))

    def _sync_naturalization_status(self):
        config = self.project.naturalization
        has_tracks = bool(self.project.tracks)
        self._rebuild_midi_assignment_list()
        self.reference_midi_clear_button.setEnabled(bool(config.assignments))
        self.naturalization_enabled.blockSignals(True)
        self.naturalization_enabled.setChecked(config.enabled)
        self.naturalization_enabled.setEnabled(bool(config.assignments) and has_tracks)
        self.naturalization_enabled.blockSignals(False)
        if not has_tracks:
            self.naturalization_status.setText("导入音轨后可启用随机偏移")
            return
        if not config.enabled:
            self.naturalization_status.setText(self._midi_assignment_summary())
            return
        self.naturalization_status.setText("已启用 · " + self._midi_assignment_summary())

    def _rebuild_midi_assignment_list(self):
        while self.midi_list_layout.count():
            item = self.midi_list_layout.takeAt(0)
            if item.widget() is not None and item.widget() is not self.midi_empty_frame:
                item.widget().hide()
                item.widget().deleteLater()
        assignments = self.project.naturalization.assignments
        self.midi_empty_frame.setVisible(not assignments)
        if not assignments:
            self.reference_midi_input.setText("")
            self.midi_list_layout.addWidget(self.midi_empty_frame)
            return

        for index, assignment in enumerate(assignments):
            frame = QFrame()
            frame.setObjectName("midiAssignment")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(8, 8, 8, 8)
            input_box = MidiPathEdit()
            input_box.setText(Path(assignment.midi_path).name)
            input_box.setToolTip(f"{assignment.midi_path}\n拖入 MIDI 可替换此时间线")
            input_box.midi_dropped.connect(
                lambda path, item_index=index: self.replace_reference_midi(item_index, path)
            )
            layout.addWidget(input_box)
            row = QHBoxLayout()
            assign = QPushButton("分配轨道")
            assign.setObjectName("secondary")
            assign.clicked.connect(
                lambda _checked=False, item_index=index: self.assign_midi_tracks(item_index)
            )
            row.addWidget(assign)
            remove = QPushButton("移除")
            remove.setObjectName("secondary")
            remove.clicked.connect(
                lambda _checked=False, item_index=index: self.remove_reference_midi(item_index)
            )
            row.addWidget(remove)
            row.addStretch(1)
            layout.addLayout(row)
            label = QLabel(self._assignment_label(assignment))
            label.setStyleSheet("font-size: 11px; color: #57606a; border: none;")
            layout.addWidget(label)
            frame.setStyleSheet(
                "QFrame#midiAssignment { border: 1px solid #d0d7de; border-radius: 5px; }"
            )
            self.midi_list_layout.addWidget(frame)

    def _assignment_label(self, assignment: MidiAssignment) -> str:
        midi_track = (
            f"MIDI 音轨 {assignment.midi_track_index + 1}"
            if assignment.midi_track_index is not None else "自动选择首条有音符 MIDI 音轨"
        )
        if len(self.project.naturalization.assignments) == 1 and not assignment.track_ids:
            return f"{midi_track} · 负责所有已启用轨道"
        if assignment.track_ids:
            return f"{midi_track} · 负责 {len(assignment.track_ids)} 条轨道：{'、'.join(assignment.track_ids)}"
        return f"{midi_track} · 尚未分配轨道"

    def _midi_assignment_summary(self) -> str:
        assignments = self.project.naturalization.assignments
        if not assignments:
            return "未选择 MIDI，副本将使用有声段颤音"
        enabled_ids = {track.track_id for track in self.project.tracks if track.enabled}
        if len(assignments) == 1 and not assignments[0].track_ids:
            return "MIDI 已就绪，所有启用轨道均已分配；副本将使用逐音符颤音"
        assigned = {track_id for assignment in assignments for track_id in assignment.track_ids}
        missing = sorted(enabled_ids - assigned)
        if missing:
            return f"{ '、'.join(missing) } 未分配 MIDI，将不参与偏移和逐音符颤音"
        return "所有启用轨道均已分配 MIDI；副本将使用逐音符颤音"

    def _sync_track_dependent_controls(self):
        has_tracks = bool(self.project.tracks)
        self.ai_button.setEnabled(has_tracks)
        self.analyze_action.setEnabled(has_tracks)
        self.ai_customize_button.setEnabled(has_tracks)
        tooltip = "请先导入至少一条音轨" if not has_tracks else "将全部启用音轨发送给 AI 分析"
        self.ai_button.setToolTip(tooltip)

    def choose_reference_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考 MIDI", str(self.project_dir or DEFAULT_PROJECT_DIR),
            "MIDI 文件 (*.mid *.midi)",
        )
        if path:
            self.add_reference_midi(path)

    def set_reference_midi_path(self, value: str | Path):
        """Legacy single-MIDI setter: replace the first assignment or add one."""
        if self.project.naturalization.assignments:
            self.replace_reference_midi(0, value)
        else:
            self.add_reference_midi(value)

    def add_reference_midi(self, value: str | Path):
        path = Path(value)
        if path.suffix.lower() not in {".mid", ".midi"} or not path.is_file():
            QMessageBox.warning(self, "MIDI 文件无效", "请选择可读取的 .mid 或 .midi 文件。")
            return
        try:
            selected = path.resolve()
            if self.project_dir:
                selected = copy_to_media(self.project_dir, selected)
                try:
                    stored_path = str(selected.relative_to(self.project_dir.resolve()))
                except ValueError:
                    stored_path = str(selected)
            else:
                stored_path = str(selected)
        except OSError as exc:
            QMessageBox.warning(self, "MIDI 导入失败", str(exc))
            return
        assignments = self.project.naturalization.assignments
        if len(assignments) == 1 and not assignments[0].track_ids:
            # Adding a second MIDI ends the old all-tracks default. Freeze the
            # existing ownership before asking the user to assign the new one.
            assignments[0].track_ids = [
                track.track_id for track in self.project.tracks if track.enabled
            ]
        assignments.append(MidiAssignment(stored_path, []))
        self.changed()
        self._status.showMessage(f"已添加 MIDI：{selected.name}", 8000)

    def replace_reference_midi(self, index: int, value: str | Path):
        path = Path(value)
        if path.suffix.lower() not in {".mid", ".midi"} or not path.is_file():
            QMessageBox.warning(self, "MIDI 文件无效", "请选择可读取的 .mid 或 .midi 文件。")
            return
        try:
            selected = path.resolve()
            if self.project_dir:
                selected = copy_to_media(self.project_dir, selected)
                try:
                    stored_path = str(selected.relative_to(self.project_dir.resolve()))
                except ValueError:
                    stored_path = str(selected)
            else:
                stored_path = str(selected)
        except OSError as exc:
            QMessageBox.warning(self, "MIDI 导入失败", str(exc))
            return
        try:
            self.project.naturalization.assignments[index].midi_path = stored_path
        except IndexError:
            return
        self.changed()
        self._status.showMessage(f"已替换 MIDI：{selected.name}", 8000)

    def assign_midi_tracks(self, index: int):
        try:
            assignment = self.project.naturalization.assignments[index]
        except IndexError:
            return
        try:
            midi_path = (
                resolve_midi_path(self.project_dir, assignment.midi_path)
                if self.project_dir else Path(assignment.midi_path)
            )
            midi_tracks = inspect_midi_tracks(midi_path)
        except ChoirError as exc:
            self.error(exc)
            return
        dialog = MidiAssignmentDialog(
            self.project, Path(assignment.midi_path).name, assignment.track_ids,
            midi_tracks, assignment.midi_track_index, self,
        )
        if not dialog.exec():
            return
        selected_ids = dialog.track_ids()
        # Track ownership is exclusive. Moving a selection here removes it
        # from all other MIDI assignments, which resolves conflicts directly.
        for other_index, other in enumerate(self.project.naturalization.assignments):
            if other_index != index:
                other.track_ids = [track_id for track_id in other.track_ids if track_id not in selected_ids]
        assignment.track_ids = selected_ids
        assignment.midi_track_index = dialog.midi_track_index()
        self.changed()

    def remove_reference_midi(self, index: int):
        assignments = self.project.naturalization.assignments
        if not 0 <= index < len(assignments):
            return
        assignments.pop(index)
        if len(assignments) == 1:
            assignments[0].track_ids = []
        if not assignments:
            self.project.naturalization.enabled = False
        self.changed()
        self._status.showMessage("已移除 MIDI", 8000)

    def clear_reference_midi(self):
        config = self.project.naturalization
        config.assignments.clear()
        config.enabled = False
        self.changed()
        self._status.showMessage("已移除参考 MIDI，随机偏移已关闭", 8000)

    def set_naturalization_enabled(self, enabled: bool):
        if self._updating:
            return
        if enabled and not self.project.naturalization.assignments:
            self.naturalization_enabled.blockSignals(True)
            self.naturalization_enabled.setChecked(False)
            self.naturalization_enabled.blockSignals(False)
            QMessageBox.information(self, "需要参考 MIDI", "请先选择或拖入参考 MIDI 文件。")
            return
        self.project.naturalization.enabled = enabled
        self.changed()
        self._status.showMessage("随机偏移已启用" if enabled else "随机偏移已关闭", 8000)

    def _sync_room_controls(self):
        values = {
            "length_m": self.project.room.length_m,
            "width_m": self.project.room.width_m,
            "height_m": self.project.room.height_m,
            "grid_step_m": self.project.room.grid_step_m,
            "rt60_s": self.project.room.rt60_s,
            "reverb_gain_db": self.project.room.reverb_gain_db,
            "bus_gain_db": self.project.room.bus_gain_db,
            "mic_count": self.project.microphone.count,
            "spacing_m": self.project.microphone.spacing_m,
            "mic_height_m": self.project.microphone.height_m,
        }
        for key, value in values.items():
            self.room_spins[key].setValue(value)

    def _on_room_singer_selected(self, track_id: str):
        self._selected_track_id = track_id
        self._track_panel.select_track(track_id)

    def _on_room_selection_cleared(self):
        self._selected_track_id = None
        self._track_panel.select_track(None)

    def _on_track_selected(self, track_id: str):
        self._selected_track_id = track_id
        self.room_view.select_singer(track_id)

    def _on_track_deselected(self):
        self._selected_track_id = None
        self.room_view.select_singer(None)

    def _on_track_gain_changed(self, track_id: str, gain_db: float):
        track = self._track(track_id)
        track.gain_db = gain_db
        self.changed(selection=True)

    def _on_track_position_changed(self, track_id: str, x: float, y: float, z: float):
        track = self._track(track_id)
        previous_position = replace(track.position)
        track.position = Position(x, y, z)
        try:
            validate_position(track.position, self.project.room)
        except ChoirError as exc:
            track.position = previous_position
            self._track_panel.update_track(track)
            self._status.showMessage(f"坐标未应用：{exc}", 8000)
            return
        self.changed(selection=True)

    def _locate_track(self, track_id: str):
        """Center RoomView on the given singer."""
        self.room_view.select_singer(track_id)
        self._selected_track_id = track_id
        self._track_panel.select_track(track_id)

    def apply_room_parameter(self, key: str):
        if self._updating:
            return
        try:
            value = self.room_spins[key].value()
            new_room = replace(self.project.room)
            new_mic = replace(self.project.microphone)
            if key == "mic_height_m":
                new_mic.height_m = value
            elif key == "mic_count":
                new_mic.count = int(value)
            elif key == "spacing_m":
                new_mic.spacing_m = value
            else:
                setattr(new_room, key, value)
            new_room.validate()
            new_mic.validate(new_room)
            for track in self.project.tracks:
                validate_position(track.position, new_room)
            self.project.room = new_room
            self.project.microphone = new_mic
            self.changed(room=True, tracks=True, selection=True)
        except ChoirError as exc:
            self._status.showMessage(f"参数未应用：{exc}", 8000)
            self.refresh(room=True, tracks=True, selection=True)

    def scene_moved(self, track_id: str, x: float, y: float):
        track = self._track(track_id)
        track.position.x_m, track.position.y_m = x, y
        self.changed()

    def scene_moved_batch(self, positions: list[tuple[str, float, float]]):
        for track_id, x, y in positions:
            track = self._track(track_id)
            track.position.x_m, track.position.y_m = x, y
        self.changed()

    def changed(
        self,
        *,
        room: bool = False,
        tracks: bool = False,
        selection: bool = False,
        preview_sources: bool = False,
        preserve_ai_undo: bool = False,
    ):
        if not preserve_ai_undo:
            self._ai_undo_project = None
            self.undo_ai_action.setEnabled(False)
        self.dirty = True
        self._project_revision += 1
        self.refresh(
            room=room, tracks=tracks, selection=selection,
            preview_sources=preview_sources,
        )

    def _sync_ai_recommendations(self):
        """Expose the project-persisted AI options through both UI entrypoints."""
        self._ai_recommendations = self.project.ai_recommendations
        has_recommendations = bool(self._ai_recommendations)
        self.choose_ai_action.setEnabled(has_recommendations)
        self.ai_solution_button.setEnabled(has_recommendations)
        if has_recommendations:
            self._ai_suggestion_base_project = deepcopy(self.project)
            self.ai_solution_status.setText(f"已保存 {len(self._ai_recommendations)} 套方案")
        else:
            self._ai_suggestion_base_project = None
            self.ai_solution_status.setText("尚未生成方案")

    def _track(self, track_id: str):
        return next(t for t in self.project.tracks if t.track_id == track_id)

    def toggle_track(self, track_id: str, enabled: bool):
        if not self._updating:
            self._track(track_id).enabled = enabled
            self.changed(selection=True)
