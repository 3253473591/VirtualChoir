from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox

from ..errors import ChoirError
from ..models import Position
from ..project_io import save_ai_json
from .dialogs import AISuggestionDialog
from .workers import RenderWorker

class RenderWorkflowMixin:
    def render(self, job: str):
        operation = {"preview": "预览", "stems": "导出分轨", "mix": "导出混音"}.get(job, "渲染")
        if not self._ensure_project_saved(operation):
            return
        output_names = {"preview": "preview", "stems": "Stems", "mix": "Mixdown"}
        output = self.project_dir / output_names[job]
        self._start_render(job, output, False)

    def _start_render(self, job: str, output: Path, accept_clip_risk: bool):
        self._pause_preview_playback()
        self.render_job = job
        self.render_output_dir = output
        self.progress.setValue(0)
        self._progress_label.setText("准备渲染…")
        self.stop_button.setEnabled(True)

        # Disable controls
        self.preview_button.setEnabled(False)
        self.stems_button.setEnabled(False)
        self.mix_button.setEnabled(False)
        self._track_panel.setEnabled(False)
        self.room_view.setEnabled(False)
        self.import_action.setEnabled(False)
        self.open_action.setEnabled(False)
        # The render worker holds an immutable project snapshot, so saving the
        # current project cannot invalidate its job.
        self.save_action.setEnabled(True)

        job_labels = {"preview": "正在生成预览…", "stems": "正在导出分轨…", "mix": "正在混音导出…"}
        self._overlay.show_overlay(job_labels.get(job, "正在渲染…"))

        self.render_thread = QThread(self)
        self.render_worker = RenderWorker(
            self.project, self.project_dir, output, job, accept_clip_risk
        )
        self.render_worker.moveToThread(self.render_thread)
        self.render_thread.started.connect(self.render_worker.run)
        self.render_worker.progress.connect(
            self._task_callbacks.render_progress, Qt.ConnectionType.QueuedConnection
        )
        self.render_worker.notice.connect(
            self._task_callbacks.render_notice, Qt.ConnectionType.QueuedConnection
        )
        self.render_worker.completed.connect(
            self._task_callbacks.render_completed, Qt.ConnectionType.QueuedConnection
        )
        self.render_worker.failed.connect(
            self._task_callbacks.render_failed, Qt.ConnectionType.QueuedConnection
        )
        self.render_worker.completed.connect(self.render_worker.deleteLater)
        self.render_worker.failed.connect(self.render_worker.deleteLater)
        self.render_thread.finished.connect(self._render_cleanup, Qt.ConnectionType.QueuedConnection)
        self.render_thread.start()

    @Slot(int, str)
    def _render_progress(self, percent: int, message: str):
        self.progress.setValue(percent)
        self._progress_label.setText(message)
        self._overlay.set_text(f"{message} ({percent}%)")
        self._status.showMessage(message)

    @Slot(str)
    def _render_notice(self, message: str):
        self._status.showMessage(message, 12000)
        self._toast.show_message(message, "info")

    @Slot(object)
    def _render_complete(self, result):
        self._overlay.hide_overlay()
        if self.render_job == "preview" and self.render_output_dir:
            self._refresh_preview_sources(self.project_dir, Path(result))
            self._start_preview_playback()
        elif self.render_job in {"stems", "mix"}:
            if self.render_output_dir:
                self._refresh_preview_sources(self.project_dir)
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.render_output_dir)))
                self._status.showMessage(f"导出完成，已打开目标文件夹：{self.render_output_dir}", 10000)
        self._toast.show_message(f"渲染完成：{Path(str(result)).name}", "success")
        self._finish_render_thread()

    @Slot(object, str, object)
    def _render_failed(self, error: ChoirError, render_job: str | None = None,
                       output_dir: Path | None = None):
        self._overlay.hide_overlay()
        if error.code == "AUDIO_CLIP_RISK" and QMessageBox.question(
            self, "削波风险",
            f"{error.message}\n是否仍继续写出？"
        ) == QMessageBox.StandardButton.Yes:
            if render_job and output_dir:
                self._rerun_after_cleanup = (render_job, output_dir)
            else:
                self.error(ChoirError("RENDER_FAILED", "渲染任务状态已丢失，无法安全重试"))
        else:
            self.error(error)
        self._finish_render_thread()

    def _finish_render_thread(self):
        """End the worker event loop only after the GUI handled its result."""
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.quit()

    @Slot()
    def _render_cleanup(self):
        thread = self.render_thread
        rerun = self._rerun_after_cleanup
        self._rerun_after_cleanup = None
        self.render_worker = None
        self.render_thread = None
        self.render_job = None
        self.render_output_dir = None
        self.stop_button.setEnabled(False)

        self.preview_button.setEnabled(True)
        self.stems_button.setEnabled(True)
        self.mix_button.setEnabled(True)
        self._track_panel.setEnabled(True)
        self.room_view.setEnabled(True)
        self.import_action.setEnabled(True)
        self.open_action.setEnabled(True)
        self.save_action.setEnabled(True)

        self._progress_label.setText("准备就绪")
        self._overlay.hide_overlay()

        if thread is not None:
            thread.deleteLater()

        if rerun:
            self._start_render(rerun[0], rerun[1], True)

    def cancel_render(self):
        self._pause_preview_playback()
        if self.render_worker is None or self.render_thread is None or not self.render_thread.isRunning():
            return
        self.stop_button.setEnabled(False)
        self.render_worker.cancel()
        self._status.showMessage("正在请求取消…")
        self._progress_label.setText("取消中…")

    def cancel_active_task(self):
        """Route the shared stop button to the task that owns it."""
        if self.render_worker is not None and self.render_thread is not None and self.render_thread.isRunning():
            self.cancel_render()
            return
        if self.dup_worker is not None and self.dup_thread is not None and self.dup_thread.isRunning():
            self.cancel_duplicate()

    def error(self, error: ChoirError):
        detail = f"{error.code}\n{error.message}"
        if error.detail:
            detail += f"\n\n详情：{error.detail}"
        self._status.showMessage(f"操作失败：{error.message}", 15000)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(error.code)
        box.setText(error.message)
        box.setDetailedText(detail)
        box.exec()

    def closeEvent(self, event):
        running_tasks = any(
            thread is not None and thread.isRunning()
            for thread in (self.render_thread, self.ai_thread, self.dup_thread)
        )
        if running_tasks:
            event.ignore()
            QMessageBox.information(
                self,
                "任务正在运行",
                "请等待当前 AI、渲染或副本任务完成后再关闭程序。",
            )
            return
        if not self.dirty:
            event.accept()
            return
        choice = QMessageBox.question(
            self, "未保存修改",
            "工程存在未保存修改。是否保存？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Save:
            self.save()
            event.accept() if not self.dirty else event.ignore()
        elif choice == QMessageBox.StandardButton.Discard:
            event.accept()
        else:
            event.ignore()
