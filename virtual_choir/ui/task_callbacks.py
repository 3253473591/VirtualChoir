"""Main-thread dispatch for background worker results."""

from __future__ import annotations

from PySide6.QtCore import QObject, Slot


class TaskCallbacks(QObject):
    """Receive worker signals on the GUI thread before touching widgets."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window

    @Slot(int, str)
    def duplicate_progress(self, percent: int, message: str) -> None:
        self._window._dup_progress(percent, message)

    @Slot(object)
    def duplicate_completed(self, paths) -> None:
        self._window._dup_completed(paths)

    @Slot(object)
    def duplicate_failed(self, error) -> None:
        self._window._dup_failed(error)

    @Slot(object)
    def ai_completed(self, result) -> None:
        self._window._ai_completed(result)

    @Slot(object)
    def ai_failed(self, error) -> None:
        self._window._ai_failed(error)

    @Slot(int, str)
    def render_progress(self, percent: int, message: str) -> None:
        self._window._render_progress(percent, message)

    @Slot(str)
    def render_notice(self, message: str) -> None:
        self._window._render_notice(message)

    @Slot(object)
    def render_completed(self, result) -> None:
        self._window._render_complete(result)

    @Slot(object, str, object)
    def render_failed(self, error, render_job: str, output_dir) -> None:
        self._window._render_failed(error, render_job, output_dir)
