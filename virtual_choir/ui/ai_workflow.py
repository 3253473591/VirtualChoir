from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import QMessageBox

from ..errors import ChoirError
from ..models import Position, resolve_source_path
from ..project_io import load_project, save_ai_json
from ..settings import save_ai_config
from .dialogs import AIConversationDialog, AISettingsDialog, AISuggestionDialog
from .workers import AIAnalysisWorker

class AIWorkflowMixin:
    def ai_settings(self):
        dialog = AISettingsDialog(self)
        if self.ai_config:
            dialog.provider.setCurrentText(self.ai_config.provider)
            dialog.base_url.setText(self.ai_config.base_url)
            dialog.api_key.setText(self.ai_config.api_key)
            if self.ai_config.model:
                dialog.model.addItem(self.ai_config.model)
        if dialog.exec():
            self.ai_config = dialog.config()
            if save_ai_config(self.ai_config):
                self._status.showMessage("AI 设置与密钥已保存")
            else:
                self._status.showMessage("连接方式和模型已保存；当前系统无法持久保存 API 密钥", 10000)

    def customize_with_ai(self):
        if not self.project.tracks:
            QMessageBox.information(self, "没有音轨", "请先导入至少一个符合规范的 WAV 文件。")
            return
        if not self.ai_config or not self.ai_config.model:
            QMessageBox.information(
                self, "需要 AI 设置",
                "请先在\"AI 设置\"中填写密钥、拉取模型并选择模型。"
            )
            self.ai_settings()
            if not self.ai_config or not self.ai_config.model:
                return
        try:
            self.project.validate()
        except ChoirError as exc:
            self.error(exc)
            return
        previous_messages = deepcopy(self.project.ai_conversation)
        dialog = AIConversationDialog(
            self.ai_config, self.project, self.project.ai_conversation, self
        )
        accepted = dialog.exec()
        if dialog.messages != previous_messages:
            self.project.ai_conversation = deepcopy(dialog.messages)
            self.changed()
        if not accepted or not dialog.recommendations:
            return
        first_number = len(self.project.ai_recommendations) + 1
        for offset, recommendation in enumerate(dialog.recommendations):
            stored = deepcopy(recommendation)
            stored["name"] = f"定制方案 {first_number + offset}: {stored['name']}"
            self.project.ai_recommendations.append(stored)
        self._sync_ai_recommendations()
        self.changed()
        if self.project_dir:
            try:
                save_ai_json(
                    self.project_dir, "ai_suggestion.json",
                    {"recommendations": self.project.ai_recommendations},
                )
            except ChoirError as exc:
                self.error(exc)
                return
        self.choose_ai_suggestion()

    def analyze_with_ai(self):
        if not self.project.tracks:
            QMessageBox.information(self, "没有音轨", "请先导入至少一个符合规范的 WAV 文件。")
            return
        if not self.ai_config or not self.ai_config.model:
            QMessageBox.information(
                self, "需要 AI 设置",
                "请先在\"AI 设置\"中填写密钥、拉取模型并选择模型。"
            )
            self.ai_settings()
            if not self.ai_config or not self.ai_config.model:
                return
        enabled_tracks = [track for track in self.project.tracks if track.enabled]
        if not enabled_tracks:
            QMessageBox.information(self, "没有启用音轨", "请至少启用一条音轨后再发送给 AI。")
            return
        sources = []
        for track in enabled_tracks:
            source = (
                resolve_source_path(self.project_dir, track.source_path or track.file_name)
                if self.project_dir
                else Path(track.source_path or track.file_name)
            )
            if not source.is_file():
                self.error(ChoirError("AUDIO_NOT_FOUND", str(source)))
                return
            sources.append((track.track_id, track.file_name, source))

        # The room can still be edited while the request is in flight.  Freeze
        # the exact state used for the prompt so the response is validated
        # against the same boundaries that were sent to the model.
        analysis_project = deepcopy(self.project)
        try:
            analysis_project.validate()
        except ChoirError as exc:
            self.error(exc)
            return
        self._ai_start_revision = self._project_revision

        self.ai_button.setEnabled(False)
        self.analyze_action.setEnabled(False)
        self.ai_button.setText("◌ 分析中…")
        self._status.showMessage(f"正在转换并发送全部 {len(sources)} 条启用音轨…")
        self._overlay.show_overlay(f"AI 正在分析 {len(sources)} 条音频…")

        self.ai_thread = QThread(self)
        self.ai_worker = AIAnalysisWorker(
            self.ai_config, sources, analysis_project, self.project_dir
        )
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.completed.connect(
            self._task_callbacks.ai_completed, Qt.ConnectionType.QueuedConnection
        )
        self.ai_worker.failed.connect(
            self._task_callbacks.ai_failed, Qt.ConnectionType.QueuedConnection
        )
        self.ai_worker.completed.connect(self.ai_thread.quit)
        self.ai_worker.failed.connect(self.ai_thread.quit)
        self.ai_worker.completed.connect(self.ai_worker.deleteLater)
        self.ai_worker.failed.connect(self.ai_worker.deleteLater)
        self.ai_thread.finished.connect(self._ai_cleanup, Qt.ConnectionType.QueuedConnection)
        self.ai_thread.start()

    @Slot(object)
    def _ai_completed(self, result):
        response, metadata = result
        if self._ai_start_revision != self._project_revision:
            self._overlay.hide_overlay()
            self._status.showMessage("AI 分析期间工程已修改，建议未应用；请重新发送以基于当前参数分析。", 10000)
            self._toast.show_message("工程已修改，未应用 AI 建议", "info")
            return
        first_number = len(self.project.ai_recommendations) + 1
        recommendations = []
        for offset, recommendation in enumerate(response["recommendations"]):
            stored = deepcopy(recommendation)
            stored["name"] = f"方案 {first_number + offset}: {stored['name']}"
            recommendations.append(stored)
        self.project.ai_recommendations.extend(recommendations)
        if self.project_dir:
            try:
                save_ai_json(
                    self.project_dir, "ai_suggestion.json",
                    {"recommendations": self.project.ai_recommendations},
                )
            except ChoirError as exc:
                self._overlay.hide_overlay()
                self.error(exc)
                return
        self._ai_suggestion_base_project = deepcopy(self.project)
        self._sync_ai_recommendations()
        self.dirty = True
        self._project_revision += 1
        self.refresh(tracks=True, selection=True)
        self._overlay.hide_overlay()
        if not self.choose_ai_suggestion():
            self._status.showMessage("AI 建议未应用", 5000)
            return
        clip_note = "；输入已限幅量化" if metadata["clipped"] else ""
        self._status.showMessage(
            f"AI 方案已应用（输入：最多 10 秒有声干声 / 44.1kHz / 16-bit / mono{clip_note}）", 10000
        )
        self._toast.show_message("AI 方案已应用，可从 AI 菜单切换或撤销", "success")

    @Slot(object)
    def _ai_failed(self, error: ChoirError):
        self._overlay.hide_overlay()
        self.error(error)

    @Slot()
    def _ai_cleanup(self):
        thread = self.ai_thread
        self.ai_worker = None
        self.ai_thread = None
        self._ai_start_revision = None
        self.ai_button.setEnabled(True)
        self.analyze_action.setEnabled(True)
        if thread is not None:
            thread.deleteLater()
        self.ai_button.setText("✦ 发送给 AI")

    def _apply_ai_suggestion(self, suggestion: dict):
        for key in ("length_m", "width_m", "height_m"):
            if key in suggestion["room"]:
                setattr(self.project.room, key, suggestion["room"][key])
        self.project.room.rt60_s = suggestion["room"]["rt60_s"]
        self.project.room.reverb_gain_db = suggestion["room"]["reverb_gain_db"]
        self.project.microphone.count = suggestion["microphone"]["count"]
        self.project.microphone.spacing_m = suggestion["microphone"]["spacing_m"]
        self.project.microphone.height_m = suggestion["microphone"]["height_m"]
        by_id = {singer["track_id"]: singer for singer in suggestion["singers"]}
        for track in self.project.tracks:
            singer = by_id[track.track_id]
            track.position = Position(**singer["position"])
            track.gain_db = singer["gain_db"]
        self.changed(room=True, tracks=True, selection=True, preserve_ai_undo=True)

    def choose_ai_suggestion(self) -> bool:
        if not self._ai_recommendations or self._ai_suggestion_base_project is None:
            return False
        # Capture the latest user edits only when a solution is actually chosen.
        # Doing this in every high-frequency edit would negate incremental refresh.
        self._ai_suggestion_base_project = deepcopy(self.project)
        dialog = AISuggestionDialog(self._ai_recommendations, self)
        if not dialog.exec():
            return False
        recommendation = dialog.selected_recommendation()
        self.project = deepcopy(self._ai_suggestion_base_project)
        self._ai_undo_project = deepcopy(self._ai_suggestion_base_project)
        self._apply_ai_suggestion(recommendation)
        self.undo_ai_action.setEnabled(True)
        self.ai_solution_button.setEnabled(True)
        self.ai_solution_status.setText(f"当前方案：{recommendation['name']}")
        if self.project_dir:
            try:
                save_ai_json(self.project_dir, "approved_config.json", self.project.to_dict())
            except ChoirError as exc:
                self.error(exc)
        self._status.showMessage(f"AI 方案“{recommendation['name']}”已应用", 8000)
        return True

    def undo_ai_suggestion(self):
        if self._ai_undo_project is None:
            return
        self.project = deepcopy(self._ai_undo_project)
        self._ai_undo_project = None
        self.undo_ai_action.setEnabled(False)
        self.changed(room=True, tracks=True, selection=True, preserve_ai_undo=True)
        self.ai_solution_status.setText("已撤销当前方案，可重新切换")
        self._status.showMessage("已撤销最近应用的 AI 方案", 8000)
        self._toast.show_message("已撤销 AI 方案", "info")
