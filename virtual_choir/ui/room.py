from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItemGroup, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QWidget,
)

from ..models import ProjectConfig, grid_nodes, snap_to_grid
from .theme import Colors


def _grid_nodes(length: float, step: float) -> list[float]:
    return grid_nodes(length, step)


def _snap(value: float, length: float, step: float) -> float:
    return snap_to_grid(value, length, step)


class SingerItem(QGraphicsEllipseItem):
    """Circle singer icon with hover/selection/disabled states."""

    def __init__(self, track_id: str, file_name: str, x: float, y: float,
                 gain_db: float, z_m: float):
        super().__init__(-0.13, -0.13, 0.26, 0.26)
        self.track_id = track_id
        self.file_name = file_name
        self._gain_db = gain_db
        self._z_m = z_m
        self._enabled = True
        self._selected_flag = False

        self.setPos(x, y)
        self.setBrush(QBrush(QColor(Colors.SINGER_FILL)))
        self.setPen(QPen(Qt.white, 0.025))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        # Label
        self.label = QGraphicsSimpleTextItem(track_id, self)
        self.label.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.label.setPos(0.14, -0.10)
        font = QFont()
        font.setPointSizeF(8.25)
        self.label.setFont(font)
        self.label.setBrush(QBrush(QColor(Colors.TEXT_PRIMARY)))

        # Selection halo (hidden by default)
        self.halo = QGraphicsEllipseItem(-0.19, -0.19, 0.38, 0.38, self)
        self.halo.setPen(QPen(QColor(Colors.ACCENT), 0.03))
        self.halo.setBrush(QBrush(Qt.NoBrush))
        self.halo.setVisible(False)
        self.halo.setZValue(-1)

        # Overlap badge (hidden by default)
        self.badge = QGraphicsRectItem(0.08, -0.22, 0.16, 0.10, self)
        self.badge.setBrush(QBrush(QColor(Colors.TEXT_PRIMARY)))
        self.badge.setPen(QPen(Qt.NoPen))
        self.badge.setVisible(False)
        self.badge.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.badge_text = QGraphicsSimpleTextItem("", self.badge)
        self.badge_text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.badge_text.setPos(0.02, -0.01)
        bf = QFont()
        bf.setPointSizeF(5.25)
        self.badge_text.setFont(bf)
        self.badge_text.setBrush(QBrush(Qt.white))

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self.setBrush(QBrush(QColor(Colors.SINGER_FILL)))
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setOpacity(1.0)
        else:
            self.setBrush(QBrush(QColor(Colors.SINGER_DISABLED)))
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            self.setOpacity(0.5)

    def set_selected_flag(self, selected: bool):
        self._selected_flag = selected
        self.halo.setVisible(selected)
        self.setZValue(100 if selected else 10)
        self.label.setZValue(1 if selected else 0)

    def set_overlap_count(self, count: int):
        if count > 1:
            self.badge.setVisible(True)
            self.badge_text.setText(f"+{count - 1}")
        else:
            self.badge.setVisible(False)

    def update_info(self, gain_db: float, z_m: float):
        self._gain_db = gain_db
        self._z_m = z_m

    def info_text(self) -> str:
        return (
            f"{self.track_id}\n{self.file_name}\n"
            f"X: {self.pos().x():.3f}  Y: {self.pos().y():.3f}  Z: {self._z_m:.3f}\n"
            f"增益: {self._gain_db:+.1f} dB"
        )

class MicItem(QGraphicsPathItem):
    """Simplified microphone-array icon."""

    def __init__(self, x: float, y: float, label: str):
        super().__init__()
        self.setPos(x, y)
        self.setZValue(5)

        # Build mic shape: rounded head + handle pointing toward singers (+Y)
        path = QPainterPath()
        # Mic body (rounded rect, wider at top)
        path.addRoundedRect(-0.08, -0.10, 0.16, 0.18, 0.03, 0.03)
        # Handle
        path.addRoundedRect(-0.025, 0.08, 0.05, 0.08, 0.01, 0.01)
        self.setPath(path)

        # Gradient fill
        gradient = QLinearGradient(0, -0.10, 0, 0.16)
        gradient.setColorAt(0.0, QColor("#fbbf24"))
        gradient.setColorAt(1.0, QColor(Colors.MIC_AMBER))
        self.setBrush(QBrush(gradient))
        self.setPen(QPen(QColor(Colors.MIC_AMBER_DARK), 0.02))

        # Label
        self.text = QGraphicsSimpleTextItem(label, self)
        self.text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.text.setPos(0.06, -0.04)
        tf = QFont()
        tf.setPointSizeF(7.5)
        tf.setBold(True)
        self.text.setFont(tf)
        self.text.setBrush(QBrush(QColor(Colors.MIC_AMBER_DARK)))


class RoomView(QGraphicsView):
    singer_moved = Signal(str, float, float)
    singers_moved = Signal(object)
    singer_selected = Signal(str)
    selection_cleared = Signal()

    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setAccessibleName("房间俯视图")

        self.project: ProjectConfig | None = None
        self.room = ProjectConfig().room
        self._singer_items: dict[str, SingerItem] = {}
        self._show_pickup = True
        self._pickup_mode = "outline"
        self._show_labels = True
        self._label_mode = "context"
        self._show_grid = True
        self._view_mode = "top"  # "top" | "side"
        self._drag_guide: QGraphicsLineItem | None = None
        self._hovered_singer: SingerItem | None = None
        self._drag_start = None
        self._drag_origins: dict[str, tuple[float, float]] = {}
        self._selection_start = None
        self._selection_rect: QGraphicsRectItem | None = None
        self._selection_additive = False
        self._zoom_factor = 1.0
        self._min_zoom_factor = 0.5
        self._max_zoom_factor = 4.0

        # Keep hover information in viewport pixels instead of scene units.  The
        # room is only a few scene units wide, so a scene-sized text background
        # becomes disproportionately large.
        self._hover_tooltip = QLabel(self.viewport())
        self._hover_tooltip.setStyleSheet(
            "background: #1f2328; color: white; border-radius: 4px; padding: 4px 6px;"
        )
        tooltip_font = QFont()
        tooltip_font.setPointSizeF(8.25)
        self._hover_tooltip.setFont(tooltip_font)
        self._hover_tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hover_tooltip.hide()

        self._empty_hint = QLabel("拖入 WAV 文件，或从“文件”菜单导入 WAV", self.viewport())
        self._empty_hint.setObjectName("roomEmptyHint")
        self._empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setStyleSheet(
            "background: rgba(255,255,255,224); color: #656d76; "
            "border: 1px solid #d1d9e0; border-radius: 6px; padding: 10px 14px;"
        )
        self._empty_hint.adjustSize()

        self._create_scene_layers()

        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Build view toolbar
        self._build_view_toolbar()

    def _build_view_toolbar(self):
        """Floating toolbar in top-left corner of the view."""
        self.view_toolbar = QWidget(self)
        self.view_toolbar.setObjectName("viewToolbar")
        self.view_toolbar.setStyleSheet(
            "QWidget#viewToolbar { background: rgba(255,255,255,230);"
            " border: 1px solid #d1d9e0; border-radius: 6px; padding: 4px; }"
        )
        layout = QHBoxLayout(self.view_toolbar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self._btn_fit = QPushButton("⊞ 自适应")
        self._btn_fit.setObjectName("small")
        self._btn_fit.clicked.connect(self.fit_room)
        layout.addWidget(self._btn_fit)

        self._btn_grid = QPushButton("▦ 网格")
        self._btn_grid.setObjectName("small")
        self._btn_grid.setCheckable(True)
        self._btn_grid.setChecked(True)
        self._btn_grid.toggled.connect(self._toggle_grid)
        layout.addWidget(self._btn_grid)

        self._pickup_mode_box = QComboBox()
        self._pickup_mode_box.setAccessibleName("拾音区显示模式")
        self._pickup_mode_box.addItem("拾音：线框", "outline")
        self._pickup_mode_box.addItem("拾音：填充", "fill")
        self._pickup_mode_box.addItem("拾音：隐藏", "hidden")
        self._pickup_mode_box.currentIndexChanged.connect(self._set_pickup_mode)
        layout.addWidget(self._pickup_mode_box)

        self._label_mode_box = QComboBox()
        self._label_mode_box.setAccessibleName("歌手标签显示模式")
        self._label_mode_box.addItem("标签：选中/悬停", "context")
        self._label_mode_box.addItem("标签：始终", "always")
        self._label_mode_box.addItem("标签：悬停", "hover")
        self._label_mode_box.currentIndexChanged.connect(self._set_label_mode)
        layout.addWidget(self._label_mode_box)

        self.view_toolbar.move(8, 8)

    def _toggle_grid(self, checked: bool):
        self._show_grid = checked
        self._grid_group.setVisible(checked)

    def _toggle_pickup(self, checked: bool):
        self._show_pickup = checked
        self._pickup_group.setVisible(checked)

    def _set_pickup_mode(self, index: int):
        mode = self._pickup_mode_box.itemData(index)
        self._pickup_mode = mode
        self._show_pickup = mode != "hidden"
        self._pickup_group.setVisible(self._show_pickup)
        for item in self._pickup_group.childItems():
            if not isinstance(item, QGraphicsPathItem):
                continue
            color = QColor(254, 158, 11)
            if mode == "fill":
                fill = QColor(color)
                fill.setAlpha(30)
                item.setBrush(QBrush(fill))
                item.setPen(QPen(color, 0.018))
            else:
                item.setBrush(QBrush(Qt.NoBrush))
                item.setPen(QPen(color, 0.022))

    def _toggle_labels(self, checked: bool):
        self._show_labels = checked
        self._label_mode = "always" if checked else "hover"
        self._apply_label_visibility()

    def _set_label_mode(self, index: int):
        self._label_mode = self._label_mode_box.itemData(index)
        self._show_labels = self._label_mode != "hover"
        self._apply_label_visibility()

    def _apply_label_visibility(self):
        for item in self._singer_items.values():
            visible = (
                self._label_mode == "always"
                or (self._label_mode == "context" and (
                    item._selected_flag or item is self._hovered_singer
                ))
                or (self._label_mode == "hover" and item is self._hovered_singer)
            )
            item.label.setVisible(visible)

    def set_project(self, project: ProjectConfig) -> None:
        """Rebuild static room geometry, then populate the dynamic singer layer."""
        self.project = project
        self.room = project.room
        first_load = self.scene_.itemsBoundingRect().isEmpty()
        self.scene_.clear()
        self._singer_items.clear()
        self._create_scene_layers()

        width, length, step = self.room.width_m, self.room.length_m, self.room.grid_step_m

        # Grid layer
        grid_major = QPen(QColor(Colors.GRID_MAJOR), 0.008)
        for value in _grid_nodes(width, step):
            self._grid_group.addToGroup(self.scene_.addLine(value, 0, value, length, grid_major))
        for value in _grid_nodes(length, step):
            self._grid_group.addToGroup(self.scene_.addLine(0, value, width, value, grid_major))

        # Room border
        self._background_group.addToGroup(
            self.scene_.addRect(0, 0, width, length, QPen(QColor(Colors.ROOM_BORDER), 0.04))
        )

        # Coordinate rulers (Section 7.2)
        self._draw_rulers(width, length, step, self._background_group)

        # Mic items
        for index, mic in enumerate(project.microphone_positions(), start=1):
            label_text = f"M{index}"
            mic_item = MicItem(mic["x_m"], mic["y_m"], label_text)
            self.scene_.addItem(mic_item)
            self._mic_group.addToGroup(mic_item)

            # Pickup zone (semi-transparent fan)
            pickup = self._make_pickup_zone(mic["x_m"], mic["y_m"])
            self.scene_.addItem(pickup)
            self._pickup_group.addToGroup(pickup)

        self.setSceneRect(-0.5, -0.5, width + 1.0, length + 1.0)
        self._hovered_singer = None
        self._hover_tooltip.hide()
        self.update_tracks(project)
        self._update_empty_hint()
        if first_load:
            self.fit_room()

    def _create_scene_layers(self):
        self._background_group = QGraphicsItemGroup()
        self._grid_group = QGraphicsItemGroup()
        self._pickup_group = QGraphicsItemGroup()
        self._mic_group = QGraphicsItemGroup()
        self._singer_group = QGraphicsItemGroup()
        for group in (
            self._background_group, self._grid_group, self._pickup_group,
            self._mic_group, self._singer_group,
        ):
            self.scene_.addItem(group)
        self._grid_group.setVisible(self._show_grid)
        self._pickup_group.setVisible(self._show_pickup)

    def _draw_rulers(self, width: float, length: float, step: float, group: QGraphicsItemGroup):
        """Draw X-axis ruler (bottom) and Y-axis ruler (left)."""
        ruler_font = QFont()
        ruler_font.setPointSizeF(7.5)
        ruler_pen = QPen(QColor(Colors.TEXT_SECONDARY), 0.01)
        # Y-axis ruler (left side)
        for value in _grid_nodes(length, step):
            # Tick mark
            line = self.scene_.addLine(-0.12, value, -0.04, value, ruler_pen)
            line.setZValue(0)
            group.addToGroup(line)
            # Label
            text = self.scene_.addSimpleText(f"{value:.1f}" if value > 0 else "0m")
            text.setPos(-0.55, value - 0.06)
            text.setFont(ruler_font)
            text.setBrush(QBrush(QColor(Colors.TEXT_SECONDARY)))
            text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
            group.addToGroup(text)

        # X-axis ruler (bottom)
        for value in _grid_nodes(width, step):
            line = self.scene_.addLine(value, length + 0.04, value, length + 0.12, ruler_pen)
            line.setZValue(0)
            group.addToGroup(line)
            text = self.scene_.addSimpleText(f"{value:.1f}" if value > 0 else "0m")
            text.setPos(value - 0.15, length + 0.14)
            text.setFont(ruler_font)
            text.setBrush(QBrush(QColor(Colors.TEXT_SECONDARY)))
            text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
            group.addToGroup(text)

    def update_tracks(self, project: ProjectConfig) -> None:
        """Incrementally synchronize singer items without rebuilding room geometry."""
        self.project = project
        current_ids = set(self._singer_items)
        new_ids = {track.track_id for track in project.tracks}
        for track_id in current_ids - new_ids:
            item = self._singer_items.pop(track_id)
            self.scene_.removeItem(item)

        for track in project.tracks:
            item = self._singer_items.get(track.track_id)
            if item is None:
                item = SingerItem(
                    track.track_id, track.file_name,
                    track.position.x_m, track.position.y_m,
                    track.gain_db, track.position.z_m,
                )
                item.label.setVisible(False)
                # QGraphicsItemGroup handles child mouse events itself. Keep
                # interactive singers directly in the scene so dragging reaches
                # their ItemIsMovable handler.
                self.scene_.addItem(item)
                self._singer_items[track.track_id] = item
            item.setPos(track.position.x_m, track.position.y_m)
            item.update_info(track.gain_db, track.position.z_m)
            item.set_enabled(track.enabled)
        self._update_overlaps()
        self._apply_label_visibility()
        self._update_empty_hint()

    def _update_empty_hint(self):
        self._empty_hint.setVisible(not self._singer_items)
        if not self._empty_hint.isVisible():
            return
        self._empty_hint.adjustSize()
        self._empty_hint.move(
            max(8, (self.viewport().width() - self._empty_hint.width()) // 2),
            max(8, (self.viewport().height() - self._empty_hint.height()) // 2),
        )

    def _make_pickup_zone(self, x: float, y: float) -> QGraphicsPathItem:
        """Create a sector (fan) pickup zone in front of a mic."""
        import math
        path = QPainterPath()
        radius = 2.0
        angle = 120  # degrees total
        start_angle = -60  # degrees from center (which points to +Y)
        steps = 20
        path.moveTo(x, y)
        for i in range(steps + 1):
            a = math.radians(start_angle + angle * i / steps)
            px = x + radius * math.sin(a)
            py = y + radius * math.cos(a)
            path.lineTo(px, py)
        path.closeSubpath()
        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setPen(QPen(QColor(254, 158, 11), 0.022))
        item.setZValue(1)
        return item

    def _update_overlaps(self):
        """Detect overlapping singers and update badges."""
        positions: dict[tuple[float, float], list[str]] = {}
        for tid, item in self._singer_items.items():
            key = (round(item.pos().x(), 2), round(item.pos().y(), 2))
            positions.setdefault(key, []).append(tid)
        for tids in positions.values():
            count = len(tids)
            for tid in tids:
                self._singer_items[tid].set_overlap_count(count)

    def select_singer(self, track_id: str | None):
        """Programmatically select a singer (called from TrackPanel)."""
        self._set_selected_ids({track_id} if track_id else set())
        if track_id and track_id in self._singer_items:
            item = self._singer_items[track_id]
            self.centerOn(item)

    def _set_selected_ids(self, track_ids: set[str]) -> None:
        for track_id, item in self._singer_items.items():
            item.set_selected_flag(track_id in track_ids)
        self._apply_label_visibility()

    def _selected_singers(self) -> list[SingerItem]:
        return [item for item in self._singer_items.values() if item._selected_flag and item._enabled]

    def _notify_selection(self) -> None:
        selected = self._selected_singers()
        if len(selected) == 1:
            self.singer_selected.emit(selected[0].track_id)
        elif not selected:
            self.selection_cleared.emit()
        else:
            # The track panel is intentionally single-select; keep room's
            # multi-selection independent while clearing its single highlight.
            self.selection_cleared.emit()

    @staticmethod
    def _additive_selection(event) -> bool:
        return bool(event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ))

    def mousePressEvent(self, event):
        """Select singers, begin a group drag, or begin a marquee selection."""
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        scene_pos = self.mapToScene(event.position().toPoint())
        singer = self._singer_at(scene_pos)
        additive = self._additive_selection(event)
        if singer is not None and singer._enabled:
            selected_ids = {item.track_id for item in self._selected_singers()}
            if additive:
                if singer.track_id in selected_ids:
                    selected_ids.remove(singer.track_id)
                else:
                    selected_ids.add(singer.track_id)
                self._set_selected_ids(selected_ids)
                self._notify_selection()
                event.accept()
                return
            if singer.track_id not in selected_ids:
                self._set_selected_ids({singer.track_id})
            selected = self._selected_singers()
            self._drag_start = scene_pos
            self._drag_origins = {
                item.track_id: (item.pos().x(), item.pos().y()) for item in selected
            }
            for item in selected:
                item.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._notify_selection()
            event.accept()
            return

        if not additive:
            self._set_selected_ids(set())
            self._notify_selection()
        self._selection_start = scene_pos
        self._selection_additive = additive
        self._selection_rect = QGraphicsRectItem()
        self._selection_rect.setPen(QPen(QColor(Colors.ACCENT), 0.02))
        selection_color = QColor(Colors.ACCENT)
        selection_color.setAlpha(35)
        self._selection_rect.setBrush(QBrush(selection_color))
        self._selection_rect.setZValue(1000)
        self.scene_.addItem(self._selection_rect)
        event.accept()

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._drag_start is not None:
            dx = scene_pos.x() - self._drag_start.x()
            dy = scene_pos.y() - self._drag_start.y()
            for track_id, (x, y) in self._drag_origins.items():
                item = self._singer_items[track_id]
                item.setPos(
                    max(0.0, min(self.room.width_m, x + dx)),
                    max(0.0, min(self.room.length_m, y + dy)),
                )
            event.accept()
            return
        if self._selection_start is not None and self._selection_rect is not None:
            self._selection_rect.setRect(QRectF(self._selection_start, scene_pos).normalized())
            event.accept()
            return
        self._update_hover(self._singer_at(scene_pos), event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        if self._drag_start is not None:
            moved = []
            for track_id, (origin_x, origin_y) in self._drag_origins.items():
                item = self._singer_items[track_id]
                x = _snap(item.pos().x(), self.room.width_m, self.room.grid_step_m)
                y = _snap(item.pos().y(), self.room.length_m, self.room.grid_step_m)
                item.setPos(x, y)
                item.setCursor(Qt.CursorShape.OpenHandCursor)
                if (x, y) != (origin_x, origin_y):
                    moved.append((track_id, x, y))
            self._drag_start = None
            self._drag_origins = {}
            self._update_overlaps()
            if moved:
                self.singers_moved.emit(moved)
            event.accept()
            return
        if self._selection_start is not None:
            rect = QRectF(self._selection_start, self.mapToScene(event.position().toPoint())).normalized()
            selected_ids = {item.track_id for item in self._selected_singers()} if self._selection_additive else set()
            selected_ids.update(
                item.track_id for item in self._singer_items.values()
                if item._enabled and item.sceneBoundingRect().intersects(rect)
            )
            self._set_selected_ids(selected_ids)
            self._selection_start = None
            self._selection_additive = False
            if self._selection_rect is not None:
                self.scene_.removeItem(self._selection_rect)
                self._selection_rect = None
            self._notify_selection()
            event.accept()
            return
        event.accept()

    def _singer_at(self, scene_pos) -> SingerItem | None:
        """Resolve the actual singer circle instead of its grid or label items."""
        hits = [
            singer for singer in self._singer_items.values()
            if singer.isVisible() and singer.contains(singer.mapFromScene(scene_pos))
        ]
        return max(hits, key=lambda singer: singer.zValue(), default=None)

    def _update_hover(self, singer: SingerItem | None, viewport_pos=None):
        if singer == self._hovered_singer:
            if singer:
                self._position_hover_tooltip(viewport_pos)
            return
        self._hovered_singer = singer
        self._apply_label_visibility()
        if singer is None:
            self._hover_tooltip.hide()
        else:
            info = singer.info_text()
            self._hover_tooltip.setText(info)
            self._hover_tooltip.adjustSize()
            self._position_hover_tooltip(viewport_pos)
            self._hover_tooltip.show()

    def _position_hover_tooltip(self, viewport_pos=None):
        if viewport_pos is None and self._hovered_singer:
            viewport_pos = self.mapFromScene(self._hovered_singer.scenePos())
        if viewport_pos is None:
            return
        size = self._hover_tooltip.size()
        x = min(viewport_pos.x() + 12, self.viewport().width() - size.width() - 4)
        y = viewport_pos.y() - size.height() - 12
        if y < 4:
            y = min(viewport_pos.y() + 12, self.viewport().height() - size.height() - 4)
        self._hover_tooltip.move(max(4, x), max(4, y))

    def leaveEvent(self, event):
        self._update_hover(None)
        super().leaveEvent(event)

    def fit_room(self):
        if self.sceneRect().isEmpty():
            return
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = 1.0

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            event.ignore()
            return
        step = 1.15 if delta > 0 else 1 / 1.15
        target = max(self._min_zoom_factor, min(self._max_zoom_factor, self._zoom_factor * step))
        if target != self._zoom_factor:
            factor = target / self._zoom_factor
            self.scale(factor, factor)
            self._zoom_factor = target
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep toolbar in top-left
        self.view_toolbar.move(8, 8)
        self._update_empty_hint()
