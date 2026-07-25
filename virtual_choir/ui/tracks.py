from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDialog, QFrame,
    QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QMenu, QMessageBox, QPushButton,
    QScrollArea, QStackedLayout, QTableWidget, QTableWidgetItem,
    QToolButton, QVBoxLayout, QWidget,
)

from ..models import ProjectConfig, TrackConfig
from .theme import Colors, Fonts
from .widgets import ParameterSpinBox


class CollapsibleCard(QFrame):
    """A foldable card with title bar and content area."""

    def __init__(self, title: str, help_text: str = "", parent=None):
        super().__init__(parent)
        self._expanded = True
        self._animation: QPropertyAnimation | None = None
        self.setObjectName("collapsibleCard")
        self.setAccessibleName(f"{title} 折叠面板")
        self.setStyleSheet(
            "QFrame#collapsibleCard {"
            "  background: #ffffff;"
            "  border: 1px solid #d1d9e0;"
            "  border-radius: 6px;"
            "}"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Title bar
        self._title_bar = QFrame()
        self._title_bar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title_bar.setFixedHeight(36)
        self._title_bar.setStyleSheet(
            "QFrame { background: transparent; border: none; }"
        )
        tb_layout = QHBoxLayout(self._title_bar)
        tb_layout.setContentsMargins(12, 0, 8, 0)

        self._arrow = QLabel("▼")
        self._arrow.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._arrow.setFixedWidth(16)
        self._arrow.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        tb_layout.addWidget(self._arrow)

        self._title_label = QLabel(title)
        self._title_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._title_label.setStyleSheet(
            f"font-weight: 600; font-size: 13px; color: {Colors.TEXT_PRIMARY}; border: none;"
        )
        tb_layout.addWidget(self._title_label, 1)

        self._title_click_targets = [self._title_bar, self._arrow, self._title_label]
        if help_text:
            help_btn = QLabel("?")
            help_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            help_btn.setToolTip(help_text)
            help_btn.setFixedSize(18, 18)
            help_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            help_btn.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY}; border: 1px solid {Colors.BORDER};"
                f" border-radius: 9px; font-size: 11px; font-weight: 600;"
            )
            tb_layout.addWidget(help_btn)
            self._title_click_targets.append(help_btn)

        self._layout.addWidget(self._title_bar)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 12, 12)
        self._content_layout.setSpacing(10)
        self._layout.addWidget(self._content)

        # Children in a title bar receive mouse presses themselves.  Filtering
        # all of them makes the whole visible header, not just its empty edge,
        # a reliable toggle target.
        for widget in self._title_click_targets:
            widget.installEventFilter(self)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def eventFilter(self, watched, event):
        if watched in self._title_click_targets and event.type() == QEvent.Type.MouseButtonPress:
            self.toggle()
            return True
        return super().eventFilter(watched, event)

    def toggle(self):
        self._expanded = not self._expanded
        self._arrow.setText("▼" if self._expanded else "▶")
        if self._animation:
            self._animation.stop()
        animation = QPropertyAnimation(self._content, b"maximumHeight", self)
        animation.setDuration(200)
        if self._expanded:
            self._content.setVisible(True)
            self._content.setMaximumHeight(0)
            animation.setStartValue(0)
            animation.setEndValue(self._content.sizeHint().height())
            animation.finished.connect(lambda: self._content.setMaximumHeight(16777215))
        else:
            animation.setStartValue(self._content.height())
            animation.setEndValue(0)
            animation.finished.connect(lambda: self._content.setVisible(False))
        self._animation = animation
        animation.finished.connect(lambda: setattr(self, "_animation", None))
        animation.start()


class TrackCard(QFrame):
    clicked = Signal(str)       # track_id
    locate_clicked = Signal(str)
    delete_clicked = Signal(str)
    duplicate_clicked = Signal(str)   # track_id
    rename_clicked = Signal(str)
    gain_changed = Signal(str, float)
    position_changed = Signal(str, float, float, float)
    enabled_toggled = Signal(str, bool)

    def __init__(self, track_id: str, file_name: str, x: float, y: float, z: float,
                 gain_db: float, enabled: bool, parent=None):
        super().__init__(parent)
        self.track_id = track_id
        self._selected = False
        self.setAccessibleName(f"轨道 {track_id}")

        self.setObjectName("trackCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame#trackCard {"
            "  background: #ffffff;"
            "  border: 1px solid #d1d9e0;"
            "  border-radius: 6px;"
            "  padding: 10px;"
            "}"
            "QFrame#trackCard:hover {"
            "  border-color: #0969da;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Keep the frequently scanned identity and state on the primary row.
        # Destructive and low-frequency commands are available from the menu.
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(14)
        self._dot.setStyleSheet(
            f"color: {'#0969da' if enabled else '#8c959f'}; font-size: 10px; border: none;"
        )
        row1.addWidget(self._dot)

        self._id_label = QLabel(track_id)
        self._id_label.setStyleSheet(
            f"font-weight: 600; font-size: 12px; color: {Colors.TEXT_PRIMARY}; border: none;"
        )
        row1.addWidget(self._id_label, 1)

        self._enable_switch = QCheckBox()
        self._enable_switch.setChecked(enabled)
        self._enable_switch.setToolTip("启用/禁用轨道")
        self._enable_switch.setAccessibleName(f"{track_id} 启用轨道")
        self._enable_switch.toggled.connect(lambda v: self.enabled_toggled.emit(self.track_id, v))
        self._enable_switch.setStyleSheet("border: none;")
        row1.addWidget(self._enable_switch)

        self._more_menu = QMenu(self)
        duplicate_action = self._more_menu.addAction("生成差异化副本…")
        duplicate_action.triggered.connect(lambda: self.duplicate_clicked.emit(self.track_id))
        rename_action = self._more_menu.addAction("重命名 Media 文件…")
        rename_action.triggered.connect(lambda: self.rename_clicked.emit(self.track_id))
        self._more_menu.addSeparator()
        delete_action = self._more_menu.addAction("移除轨道")
        delete_action.triggered.connect(lambda: self.delete_clicked.emit(self.track_id))

        more_button = QToolButton()
        more_button.setText("...")
        more_button.setObjectName("trackMoreButton")
        more_button.setToolTip("更多轨道操作")
        more_button.setAccessibleName(f"{track_id} 更多轨道操作")
        more_button.setMenu(self._more_menu)
        more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        row1.addWidget(more_button)

        layout.addLayout(row1)

        # Row 2: filename
        fn_label = QLabel(file_name)
        fn_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;"
        )
        fn_label.setWordWrap(True)
        layout.addWidget(fn_label)

        # Row 3: coordinates
        self._coord_container = QWidget()
        self._coord_stack = QStackedLayout(self._coord_container)
        self._coord_stack.setContentsMargins(0, 0, 0, 0)
        self._coord_label = QLabel(f"X {x:.3f}  Y {y:.3f}  Z {z:.3f}")
        self._coord_label.setStyleSheet(
            f"font-family: {Fonts.MONO}; font-size: 11px; color: {Colors.ACCENT}; border: none;"
        )
        self._coord_label.setToolTip("单击在房间中定位，双击编辑坐标")
        self._coord_label.setAccessibleName(f"{track_id} 坐标，单击定位，双击编辑")
        self._coord_label.installEventFilter(self)
        coordinate_display = QWidget()
        coordinate_layout = QHBoxLayout(coordinate_display)
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.setSpacing(4)
        coordinate_layout.addWidget(self._coord_label)
        coordinate_layout.addStretch()
        locate_button = QToolButton()
        locate_button.setText("定位")
        locate_button.setToolTip("在房间中定位")
        locate_button.setAccessibleName(f"在房间中定位 {track_id}")
        locate_button.clicked.connect(lambda: self.locate_clicked.emit(self.track_id))
        coordinate_layout.addWidget(locate_button)
        self._coord_stack.addWidget(coordinate_display)
        editor = QWidget()
        editor_layout = QHBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(4)
        self._coord_spins: list[QDoubleSpinBox] = []
        for label, value in (("X", x), ("Y", y), ("Z", z)):
            spin = QDoubleSpinBox()
            spin.setPrefix(f"{label} ")
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(value)
            spin.setMinimumWidth(72)
            spin.installEventFilter(self)
            spin.editingFinished.connect(self._schedule_coordinate_commit)
            self._coord_spins.append(spin)
            editor_layout.addWidget(spin)
        self._coord_stack.addWidget(editor)
        layout.addWidget(self._coord_container)

        # Row 4: gain slider
        gain_row = QHBoxLayout()
        gain_row.setSpacing(6)
        gain_lbl = QLabel("增益")
        gain_lbl.setStyleSheet(f"font-size: 11px; color: {Colors.TEXT_SECONDARY}; border: none;")
        gain_row.addWidget(gain_lbl)

        self._gain_spin = ParameterSpinBox()
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setDecimals(1)
        self._gain_spin.setSingleStep(0.1)
        self._gain_spin.setValue(gain_db)
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setAccessibleName(f"{track_id} 增益")
        self._gain_spin.stepped.connect(self._on_gain_finished)
        self._gain_spin.committed.connect(self._on_gain_finished)
        gain_row.addWidget(self._gain_spin, 1)

        self._gain_value = QLabel(f"{gain_db:+.1f}")
        self._gain_value.setFixedWidth(42)
        self._gain_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._gain_value.setStyleSheet(
            f"font-family: {Fonts.MONO}; font-size: 11px; color: {Colors.TEXT_PRIMARY}; border: none;"
        )
        gain_row.addWidget(self._gain_value)

        layout.addLayout(gain_row)

    def _on_gain_finished(self):
        value = self._gain_spin.value()
        self._gain_value.setText(f"{value:+.1f}")
        self.gain_changed.emit(self.track_id, value)

    def _begin_coordinate_editing(self):
        self._coord_stack.setCurrentIndex(1)
        self._coord_spins[0].setFocus()
        self._coord_spins[0].selectAll()

    def _cancel_coordinate_editing(self):
        self._coord_stack.setCurrentIndex(0)

    def _commit_coordinates(self):
        if self._coord_stack.currentIndex() != 1:
            return
        x, y, z = (spin.value() for spin in self._coord_spins)
        self._coord_stack.setCurrentIndex(0)
        self.position_changed.emit(self.track_id, x, y, z)

    def _schedule_coordinate_commit(self):
        QTimer.singleShot(0, self._commit_coordinates_if_focus_left)

    def _commit_coordinates_if_focus_left(self):
        if QApplication.focusWidget() not in self._coord_spins:
            self._commit_coordinates()

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self.setStyleSheet(
                "QFrame#trackCard {"
                f"  background: {Colors.BG_SECONDARY};"
                f"  border: 1px solid {Colors.ACCENT};"
                f"  border-left: 3px solid {Colors.ACCENT};"
                "  border-radius: 6px;"
                "  padding: 10px;"
                "}"
            )
        else:
            self.setStyleSheet(
                "QFrame#trackCard {"
                "  background: #ffffff;"
                "  border: 1px solid #d1d9e0;"
                "  border-radius: 6px;"
                "  padding: 10px;"
                "}"
                "QFrame#trackCard:hover {"
                "  border-color: #0969da;"
                "}"
            )

    def update_values(self, x: float, y: float, z: float, gain_db: float, enabled: bool):
        self._gain_spin.blockSignals(True)
        self._gain_spin.setValue(gain_db)
        self._gain_spin.blockSignals(False)
        self._gain_value.setText(f"{gain_db:+.1f}")
        self._enable_switch.blockSignals(True)
        self._enable_switch.setChecked(enabled)
        self._enable_switch.blockSignals(False)
        self._dot.setStyleSheet(
            f"color: {'#0969da' if enabled else '#8c959f'}; font-size: 10px; border: none;"
        )
        self._coord_label.setText(f"X {x:.3f}  Y {y:.3f}  Z {z:.3f}")
        for spin, value in zip(self._coord_spins, (x, y, z)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.track_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        self._more_menu.exec(event.globalPos())

    def eventFilter(self, watched, event):
        if watched is self._coord_label:
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self._begin_coordinate_editing()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self.locate_clicked.emit(self.track_id)
                return True
        if watched in getattr(self, "_coord_spins", ()) and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._cancel_coordinate_editing()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._commit_coordinates()
                return True
        return super().eventFilter(watched, event)


class TrackPanel(QWidget):
    track_selected = Signal(str)
    track_deselected = Signal()
    track_locate = Signal(str)
    track_delete = Signal(str)
    track_duplicate = Signal(str)
    track_rename = Signal(str)
    track_gain_changed = Signal(str, float)
    track_position_changed = Signal(str, float, float, float)
    track_enabled_toggled = Signal(str, bool)
    batch_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._card_mode = True
        self._cards: dict[str, TrackCard] = {}
        self._table_rows: dict[str, int] = {}
        self._selected_id: str | None = None
        self._updating = False

        self.setAccessibleName("音频轨道面板")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header row
        header = QHBoxLayout()
        title = QLabel("音频轨道")
        title.setStyleSheet(f"font-weight: 600; font-size: 13px; color: {Colors.TEXT_PRIMARY};")
        header.addWidget(title)
        header.addStretch()

        self._batch_button = QPushButton("批量操作")
        self._batch_button.setObjectName("small")
        self._batch_button.setToolTip("请先导入轨道")
        self._batch_button.setEnabled(False)
        self._batch_button.clicked.connect(self.batch_requested.emit)
        header.addWidget(self._batch_button)

        # View toggle button group
        self._btn_card = QPushButton("卡片")
        self._btn_card.setCheckable(True)
        self._btn_card.setChecked(True)
        self._btn_card.setObjectName("small")
        self._btn_card.clicked.connect(lambda: self._set_mode(True))

        self._btn_table = QPushButton("表格")
        self._btn_table.setCheckable(True)
        self._btn_table.setObjectName("small")
        self._btn_table.clicked.connect(lambda: self._set_mode(False))

        toggle_group = QHBoxLayout()
        toggle_group.setSpacing(0)
        toggle_group.addWidget(self._btn_card)
        toggle_group.addWidget(self._btn_table)
        header.addLayout(toggle_group)

        layout.addLayout(header)

        # Card view
        self._card_scroll = QScrollArea()
        self._card_scroll.setWidgetResizable(True)
        self._card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._card_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(4, 0, 4, 0)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch()
        self._card_scroll.setWidget(self._card_container)

        # Table view
        self._table = QTableWidget(0, 8)
        self._table.setAccessibleName("音频轨道列表")
        self._table.setHorizontalHeaderLabels(
            ["", "轨道", "文件", "X (m)", "Y (m)", "Z (m)", "增益 (dB)", "操作"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 36)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for col in (3, 4, 5, 6):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(col, 70)
        self._table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(7, 90)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._table_selection_changed)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._table_context_menu)
        self._table.setVisible(False)

        # Empty state
        self._empty = QLabel("🎤\n拖入 WAV 文件或点击导入按钮开始")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 13px; padding: 40px; border: none;"
        )
        self._empty.setWordWrap(True)

        layout.addWidget(self._card_scroll)
        layout.addWidget(self._table)
        layout.addWidget(self._empty, 1)

    def _set_mode(self, card_mode: bool):
        self._card_mode = card_mode
        self._btn_card.setChecked(card_mode)
        self._btn_table.setChecked(not card_mode)
        self._card_scroll.setVisible(card_mode)
        self._table.setVisible(not card_mode)
        self._empty.setVisible(False)

    def _table_selection_changed(self):
        if self._updating:
            return
        rows = self._table.selectionModel().selectedRows()
        if rows:
            item = self._table.item(rows[0].row(), 1)
            if item:
                tid = item.data(Qt.ItemDataRole.UserRole)
                self.select_track(tid)
                self.track_selected.emit(tid)
        else:
            self._selected_id = None
            self.track_deselected.emit()

    def _table_context_menu(self, pos: QPoint):
        """Right-click context menu for table rows."""
        row = self._table.rowAt(pos.y())
        if row < 0:
            return
        item = self._table.item(row, 1)
        if not item:
            return
        tid = item.data(Qt.ItemDataRole.UserRole)
        self.select_track(tid)
        menu = QMenu(self)
        dup_action = menu.addAction("🎤 生成差异化副本…")
        dup_action.triggered.connect(lambda checked=False, t=tid: self.track_duplicate.emit(t))
        rename_action = menu.addAction("重命名 Media 文件…")
        rename_action.triggered.connect(lambda checked=False, t=tid: self.track_rename.emit(t))
        menu.addSeparator()
        locate_action = menu.addAction("📍 在房间中定位")
        locate_action.triggered.connect(lambda checked=False, t=tid: self.track_locate.emit(t))
        del_action = menu.addAction("🗑 移除轨道")
        del_action.triggered.connect(lambda checked=False, t=tid: self.track_delete.emit(t))
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def refresh(self, project: ProjectConfig):
        """Rebuild views from project data."""
        self._updating = True

        self._batch_button.setEnabled(bool(project.tracks))
        self._batch_button.setToolTip(
            "批量启用、禁用或移除选定轨道" if project.tracks else "请先导入轨道"
        )

        if not project.tracks:
            for card in list(self._cards.values()):
                self._card_layout.removeWidget(card)
                card.deleteLater()
            self._cards.clear()
            self._table.setRowCount(0)
            self._table_rows.clear()
            self._empty.setVisible(True)
            self._card_scroll.setVisible(False)
            self._table.setVisible(False)
            self._updating = False
            return

        self._empty.setVisible(False)
        self._card_scroll.setVisible(self._card_mode)
        self._table.setVisible(not self._card_mode)

        # Card view
        # Remove old cards
        for card in list(self._cards.values()):
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._table_rows.clear()

        for track in project.tracks:
            self._insert_card(track)

        # Table view
        self._table.setRowCount(0)
        for track in project.tracks:
            self._insert_table_row(track)

        self._updating = False
        self.select_track(self._selected_id if self._selected_id in self._cards else None)

    def _insert_card(self, track: TrackConfig) -> None:
        card = TrackCard(
            track.track_id, track.file_name,
            track.position.x_m, track.position.y_m, track.position.z_m,
            track.gain_db, track.enabled,
        )
        card.clicked.connect(self._on_card_clicked)
        card.locate_clicked.connect(self.track_locate.emit)
        card.delete_clicked.connect(self.track_delete.emit)
        card.duplicate_clicked.connect(self.track_duplicate.emit)
        card.rename_clicked.connect(self.track_rename.emit)
        card.gain_changed.connect(self.track_gain_changed.emit)
        card.position_changed.connect(self.track_position_changed.emit)
        card.enabled_toggled.connect(self.track_enabled_toggled.emit)
        self._card_layout.insertWidget(self._card_layout.count() - 1, card)
        self._cards[track.track_id] = card

    def _insert_table_row(self, track: TrackConfig) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        check = QCheckBox()
        check.setChecked(track.enabled)
        check.setAccessibleName(f"{track.track_id} 启用")
        check.toggled.connect(lambda v, tid=track.track_id: self.track_enabled_toggled.emit(tid, v))
        self._table.setCellWidget(row, 0, check)
        for col, value in enumerate((
            track.track_id, track.file_name,
            f"{track.position.x_m:.3f}", f"{track.position.y_m:.3f}",
            f"{track.position.z_m:.3f}", f"{track.gain_db:.1f}",
        )):
            item = QTableWidgetItem(value)
            item.setData(Qt.ItemDataRole.UserRole, track.track_id)
            if col < 2:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, col + 1, item)
        ops_widget = QWidget()
        ops_layout = QHBoxLayout(ops_widget)
        ops_layout.setContentsMargins(2, 2, 2, 2)
        ops_layout.setSpacing(2)
        locate_btn = QPushButton("📍")
        locate_btn.setFixedSize(24, 24)
        locate_btn.setToolTip(f"定位 {track.track_id}")
        locate_btn.clicked.connect(lambda _, tid=track.track_id: self.track_locate.emit(tid))
        del_btn = QPushButton("🗑")
        del_btn.setFixedSize(24, 24)
        del_btn.setToolTip(f"移除 {track.track_id}")
        del_btn.clicked.connect(lambda _, tid=track.track_id: self.track_delete.emit(tid))
        ops_layout.addWidget(locate_btn)
        ops_layout.addWidget(del_btn)
        self._table.setCellWidget(row, 7, ops_widget)
        self._table_rows[track.track_id] = row

    def update_track(self, track: TrackConfig) -> None:
        """Update one track's card and table row in place."""
        if track.track_id not in self._cards:
            self.add_track(track)
            return
        card = self._cards[track.track_id]
        card.update_values(
            track.position.x_m, track.position.y_m, track.position.z_m,
            track.gain_db, track.enabled,
        )
        row = self._table_rows.get(track.track_id)
        if row is None:
            return
        check = self._table.cellWidget(row, 0)
        if isinstance(check, QCheckBox):
            check.blockSignals(True)
            check.setChecked(track.enabled)
            check.blockSignals(False)
        for col, value in enumerate((
            track.track_id, track.file_name,
            f"{track.position.x_m:.3f}", f"{track.position.y_m:.3f}",
            f"{track.position.z_m:.3f}", f"{track.gain_db:.1f}",
        )):
            item = self._table.item(row, col + 1)
            if item:
                item.setText(value)

    def add_track(self, track: TrackConfig) -> None:
        self._updating = True
        self._empty.setVisible(False)
        self._card_scroll.setVisible(self._card_mode)
        self._table.setVisible(not self._card_mode)
        self._insert_card(track)
        self._insert_table_row(track)
        self._batch_button.setEnabled(True)
        self._batch_button.setToolTip("批量启用、禁用或移除选定轨道")
        self._updating = False

    def remove_track(self, track_id: str) -> None:
        card = self._cards.pop(track_id, None)
        if card:
            self._card_layout.removeWidget(card)
            card.deleteLater()
        row = self._table_rows.pop(track_id, None)
        if row is not None:
            self._table.removeRow(row)
            self._table_rows = {
                self._table.item(index, 1).data(Qt.ItemDataRole.UserRole): index
                for index in range(self._table.rowCount())
                if self._table.item(index, 1)
            }
        if not self._cards:
            self._empty.setVisible(True)
            self._card_scroll.setVisible(False)
            self._table.setVisible(False)
            self._batch_button.setEnabled(False)
            self._batch_button.setToolTip("请先导入轨道")

    def _on_card_clicked(self, track_id: str):
        self.select_track(track_id)
        self.track_selected.emit(track_id)

    def select_track(self, track_id: str | None):
        """Highlight a track in both views."""
        self._selected_id = track_id
        for tid, card in self._cards.items():
            card.set_selected(tid == track_id)
        if not self._card_mode:
            self._updating = True
            self._table.clearSelection()
            if track_id:
                for row in range(self._table.rowCount()):
                    item = self._table.item(row, 1)
                    if item and item.data(Qt.ItemDataRole.UserRole) == track_id:
                        self._table.selectRow(row)
                        break
            self._updating = False


class BatchTrackDialog(QDialog):
    """Choose tracks, then choose one batch operation to apply to them."""

    def __init__(self, tracks: list[TrackConfig], parent=None):
        super().__init__(parent)
        self._operation: str | None = None
        self._checks: dict[str, QCheckBox] = {}
        self.setWindowTitle("批量处理轨道")
        self.setMinimumSize(380, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        hint = QLabel("勾选需要处理的轨道，再选择操作。移除只删除工程中的引用，不会删除音频文件。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(hint)

        selection_row = QHBoxLayout()
        select_all = QPushButton("全选")
        select_all.setObjectName("small")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        clear_all = QPushButton("清空")
        clear_all.setObjectName("small")
        clear_all.clicked.connect(lambda: self._set_all_checked(False))
        selection_row.addWidget(select_all)
        selection_row.addWidget(clear_all)
        selection_row.addStretch()
        layout.addLayout(selection_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        checks_layout = QVBoxLayout(container)
        checks_layout.setContentsMargins(8, 8, 8, 8)
        checks_layout.setSpacing(6)
        for track in tracks:
            check = QCheckBox(f"{track.track_id}  -  {track.file_name}")
            check.setChecked(False)
            check.setToolTip("已启用" if track.enabled else "已禁用")
            self._checks[track.track_id] = check
            checks_layout.addWidget(check)
        checks_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        actions = QHBoxLayout()
        for text, operation, object_name in (
            ("启用选中", "enable", "secondary"),
            ("禁用选中", "disable", "secondary"),
            ("移除选中", "delete", "danger"),
        ):
            button = QPushButton(text)
            button.setObjectName(object_name)
            button.clicked.connect(lambda checked=False, op=operation: self._choose_operation(op))
            actions.addWidget(button)
        layout.addLayout(actions)

        close_button = QPushButton("取消")
        close_button.clicked.connect(self.reject)
        layout.addWidget(close_button)

    def _set_all_checked(self, checked: bool):
        for check in self._checks.values():
            check.setChecked(checked)

    def _choose_operation(self, operation: str):
        if not self.selected_track_ids():
            QMessageBox.information(self, "未选择轨道", "请先勾选至少一条轨道。")
            return
        self._operation = operation
        self.accept()

    def selected_track_ids(self) -> list[str]:
        return [track_id for track_id, check in self._checks.items() if check.isChecked()]

    @property
    def operation(self) -> str | None:
        return self._operation
