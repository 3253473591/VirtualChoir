from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QLineEdit, QPushButton, QSlider, QSpinBox, QStyle, QStyleOptionSlider,
    QStyleOptionSpinBox,
)


class ParameterSpinBox(QDoubleSpinBox):
    """A compact numeric input with visible, clickable stepper icons."""

    committed = Signal()
    stepped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setKeyboardTracking(False)

    def _over_step_button(self, position) -> bool:
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        style = self.style()
        for sub_control in (
            QStyle.SubControl.SC_SpinBoxUp,
            QStyle.SubControl.SC_SpinBoxDown,
        ):
            rect = style.subControlRect(
                QStyle.ComplexControl.CC_SpinBox, option, sub_control, self
            )
            if rect.contains(position):
                return True
        return False

    def mouseMoveEvent(self, event):
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if self._over_step_button(event.position().toPoint())
            else Qt.CursorShape.IBeamCursor
        )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.CursorShape.IBeamCursor)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        # The left parameter panel permits explicit typing or arrow clicks
        # only; a wheel over an input must never alter a value accidentally.
        event.ignore()

    def stepBy(self, steps: int):
        super().stepBy(steps)
        self.stepped.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.interpretText()
            self.committed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        # A click elsewhere does not commit a partially typed room value.
        value = self.value()
        super().focusOutEvent(event)
        self.blockSignals(True)
        self.setValue(value)
        self.blockSignals(False)


class ParameterIntSpinBox(QSpinBox):
    """Integer companion with the same explicit apply behavior."""

    committed = Signal()
    stepped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event):
        event.ignore()

    def stepBy(self, steps: int):
        super().stepBy(steps)
        self.stepped.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.interpretText()
            self.committed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        value = self.value()
        super().focusOutEvent(event)
        self.blockSignals(True)
        self.setValue(value)
        self.blockSignals(False)

class PreviewSlider(QSlider):
    """A slider that seeks on a track click while preserving handle dragging."""

    track_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._track_click_position: QPoint | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, option,
                QStyle.SubControl.SC_SliderHandle, self,
            )
            if not handle.contains(event.position().toPoint()):
                self._track_click_position = event.position().toPoint()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._track_click_position is not None:
            position = self._track_click_position.x()
            self._track_click_position = None
            available_width = max(1, self.width() - 1)
            value = self.minimum() + round(
                (self.maximum() - self.minimum()) * position / available_width
            )
            self.setValue(value)
            self.track_clicked.emit(value)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MidiPathEdit(QLineEdit):
    """Read-only MIDI target that accepts one or more local MIDI file drops."""

    midi_dropped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if any(
            url.isLocalFile() and url.toLocalFile().lower().endswith((".mid", ".midi"))
            for url in urls
        ):
            event.acceptProposedAction()
        else:
            event.ignore()


class MidiAddButton(QPushButton):
    """Add-MIDI button that also accepts multiple MIDI file drops."""

    midi_dropped = Signal(object)

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if any(
            url.isLocalFile() and url.toLocalFile().lower().endswith((".mid", ".midi"))
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [
            url.toLocalFile() for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile().lower().endswith((".mid", ".midi"))
        ]
        if paths:
            self.midi_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [
            url.toLocalFile() for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile().lower().endswith((".mid", ".midi"))
        ]
        if paths:
            self.midi_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
