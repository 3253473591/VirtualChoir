from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .theme import Colors


class Toast(QFrame):
    """Non-modal toast notification that slides in from bottom-right."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setVisible(False)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setStyleSheet(
            "color: #ffffff; font-size: 12px; padding: 10px 16px; border: none;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide_toast)

    def show_message(self, text: str, level: str = "info"):
        """Show a toast. level: 'success', 'error', 'info'."""
        colors = {
            "success": Colors.SUCCESS,
            "error": Colors.DANGER,
            "info": Colors.ACCENT,
        }
        bg = colors.get(level, Colors.ACCENT)
        self.setStyleSheet(
            f"QFrame {{ background: {bg}; border-radius: 6px; }}"
        )
        self._label.setText(text)
        self.adjustSize()
        self.setFixedWidth(min(self.sizeHint().width() + 20, 360))
        # Position at bottom-right of parent
        p = self.parent()
        self.move(p.width() - self.width() - 20, p.height() - self.height() - 50)
        self.setVisible(True)
        self.raise_()
        self._timer.start(3000)

    def hide_toast(self):
        self.setVisible(False)


class TaskOverlay(QWidget):
    """Semi-transparent overlay showing task progress."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(
            "background: rgba(255, 255, 255, 180);"
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._spinner = QLabel("◌")
        self._spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spinner.setStyleSheet(
            f"font-size: 32px; color: {Colors.ACCENT}; background: transparent;"
        )
        layout.addWidget(self._spinner)

        self._text = QLabel("处理中…")
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {Colors.TEXT_PRIMARY}; background: transparent;"
        )
        layout.addWidget(self._text)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_frames = ["◌", "◔", "◑", "◕", "●", "◕", "◑", "◔"]
        self._spinner_idx = 0

    def show_overlay(self, text: str = "处理中…"):
        self._text.setText(text)
        self.setGeometry(self.parent().rect())
        self.setVisible(True)
        self.raise_()
        self._spinner_idx = 0
        self._spinner_timer.start(120)

    def hide_overlay(self):
        self._spinner_timer.stop()
        self.setVisible(False)

    def set_text(self, text: str):
        self._text.setText(text)

    def _tick_spinner(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self._spinner.setText(self._spinner_frames[self._spinner_idx])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setGeometry(self.parent().rect())
