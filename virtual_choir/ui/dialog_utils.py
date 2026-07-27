from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QScrollArea, QVBoxLayout, QWidget


def scrollable_dialog_layout(
    dialog: QDialog,
    *,
    margins: tuple[int, int, int, int] = (20, 20, 20, 20),
    spacing: int | None = None,
) -> QVBoxLayout:
    """Create a dialog-wide scroll surface for content that exceeds the viewport."""
    root = QVBoxLayout(dialog)
    root.setContentsMargins(0, 0, 0, 0)
    scroll_area = QScrollArea(dialog)
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    content = QWidget(scroll_area)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(*margins)
    if spacing is not None:
        layout.setSpacing(spacing)
    scroll_area.setWidget(content)
    root.addWidget(scroll_area)
    dialog._scroll_area = scroll_area
    return layout
