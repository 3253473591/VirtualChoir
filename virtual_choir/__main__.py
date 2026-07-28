from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QFile, QTextStream, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from .bootstrap import WorkspaceInitializationError, initialize_workspace
from .gui import MainWindow


def _load_stylesheet(app: QApplication) -> None:
    """Load the global QSS stylesheet from resources/style.qss."""
    qss_path = Path(__file__).parent / "resources" / "style.qss"
    if not qss_path.is_file():
        return
    file = QFile(str(qss_path))
    if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())
        file.close()


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    try:
        initialize_workspace()
    except WorkspaceInitializationError as exc:
        # The application is still usable for mixing existing tracks.  Keep the
        # failure visible instead of silently deferring it until a duplicate is
        # generated with the vocoder.
        QMessageBox.warning(None, "模型初始化未完成", str(exc))
    # Give Qt a concrete point size before applying QSS.  This avoids Qt trying
    # to derive a point size from a pixel-sized graphics font (-1).
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setApplicationName("虚拟合唱团")
    _load_stylesheet(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
