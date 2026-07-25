from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QToolButton,
    QScrollArea, QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget,
    QDoubleSpinBox,
)

from ..ai import AIConfig
from ..audio import load_playback_audio, play_audio_segment, stop_playback
from ..errors import ChoirError
from ..models import ProjectConfig, TrackConfig
from ..render import Renderer
from ..settings import load_ai_config
from .feedback import TaskOverlay, Toast
from .room import RoomView
from .task_callbacks import TaskCallbacks
from .theme import Colors, Fonts, Spacing
from .tracks import CollapsibleCard, TrackPanel
from .widgets import MidiPathEdit, ParameterIntSpinBox, ParameterSpinBox, PreviewSlider
from .workers import DuplicateWorker, RenderWorker


from .ai_workflow import AIWorkflowMixin
from .project_workflow import ProjectWorkflowMixin
from .render_workflow import RenderWorkflowMixin
from .window_state import WindowStateMixin

class MainWindow(
    WindowStateMixin, ProjectWorkflowMixin, AIWorkflowMixin,
    RenderWorkflowMixin, QMainWindow,
):
    def __init__(self):
        super().__init__()
        self.project = ProjectConfig()
        self.project_dir: Path | None = None
        self.dirty = False
        self._updating = False
        self.ai_config: AIConfig | None = load_ai_config()
        self._ai_undo_project: ProjectConfig | None = None
        self._ai_suggestion_base_project: ProjectConfig | None = None
        self._ai_recommendations: list[dict] = []
        self._ai_start_revision: int | None = None
        self._project_revision = 0
        self._selected_track_id: str | None = None
        self._task_callbacks = TaskCallbacks(self)

        self.setWindowTitle("虚拟合唱团")
        self.setMinimumSize(1100, 700)
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self._build_actions()
        self._build_menubar()
        self._build_ui()
        self.refresh(room=True, tracks=True, selection=True)
        self.statusBar().showMessage("就绪：可拖入符合规范的 WAV")

    def _build_actions(self):
        self.import_action = QAction("导入 WAV…", self, shortcut="Ctrl+I", triggered=self.import_files)
        self.import_action.setToolTip("导入 48kHz/32-bit/单声道 WAV")

        self.open_action = QAction("打开工程…", self, shortcut="Ctrl+O", triggered=self.open_project)

        self.save_action = QAction("保存工程", self, shortcut="Ctrl+S", triggered=self.save)

        self.save_as_action = QAction("另存为…", self, triggered=self.save_as)

        self.settings_action = QAction("AI 设置…", self, triggered=self.ai_settings)

        self.analyze_action = QAction("发送给 AI", self, shortcut="Ctrl+Return", triggered=self.analyze_with_ai)
        self.analyze_action.setToolTip("将全部启用音轨发送给 AI 分析")

        self.customize_ai_action = QAction("AI 定制方案对话…", self, triggered=self.customize_with_ai)

        self.undo_ai_action = QAction("撤销 AI 方案", self, triggered=self.undo_ai_suggestion)
        self.undo_ai_action.setEnabled(False)

        self.choose_ai_action = QAction("切换 AI 方案…", self, triggered=self.choose_ai_suggestion)
        self.choose_ai_action.setEnabled(False)

        self.reset_layout_action = QAction("重置布局", self, triggered=self._reset_layout)
        self.toggle_left_action = QAction("显示/隐藏左侧面板", self, triggered=self._toggle_left_panel)
        self.toggle_left_action.setCheckable(True)
        self.toggle_left_action.setChecked(True)
        self.toggle_right_action = QAction("显示/隐藏右侧面板", self, triggered=self._toggle_right_panel)
        self.toggle_right_action.setCheckable(True)
        self.toggle_right_action.setChecked(True)

    def keyPressEvent(self, event):
        """Use Space for the preview transport when no text field is being edited."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            focused = self.focusWidget()
            if not isinstance(focused, (QLineEdit, QDoubleSpinBox)) and self._preview_data is not None:
                self._toggle_preview_playback()
                event.accept()
                return
        super().keyPressEvent(event)

    def _build_menubar(self):
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addActions([
            self.import_action, self.open_action, self.save_action, self.save_as_action
        ])
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(self.reset_layout_action)
        view_menu.addAction(self.toggle_left_action)
        view_menu.addAction(self.toggle_right_action)

        ai_menu = self.menuBar().addMenu("AI")
        ai_menu.addActions([self.settings_action, self.analyze_action, self.customize_ai_action, self.choose_ai_action, self.undo_ai_action])

        help_menu = self.menuBar().addMenu("帮助")
        help_menu.addAction("快捷键", self._show_shortcuts)
        help_menu.addAction("关于", lambda: QMessageBox.about(self, "关于", "虚拟合唱团 v1.0"))

    def _build_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setAccessibleName("主工具栏")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(toolbar.iconSize())
        toolbar.addActions([
            self.import_action, self.open_action, self.save_action
        ])
        toolbar.addSeparator()
        toolbar.addActions([self.settings_action, self.analyze_action, self.customize_ai_action])
        self.addToolBar(toolbar)

    def _build_ui(self):
        self._build_toolbar()

        # Central widget: vertical layout with splitter + bottom bar
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Three-column splitter (Section 4)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (Section 6)
        self._left_panel = self._build_left_panel()
        self._splitter.addWidget(self._left_panel)

        # Center: RoomView (Section 7)
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.room_view = RoomView()
        self.room_view.singer_moved.connect(self.scene_moved)
        self.room_view.singers_moved.connect(self.scene_moved_batch)
        self.room_view.singer_selected.connect(self._on_room_singer_selected)
        self.room_view.selection_cleared.connect(self._on_room_selection_cleared)
        center_layout.addWidget(self.room_view)
        self._splitter.addWidget(center_widget)

        # Right panel (Section 8)
        self._track_panel = TrackPanel()
        self._track_panel.track_selected.connect(self._on_track_selected)
        self._track_panel.track_deselected.connect(self._on_track_deselected)
        self._track_panel.track_locate.connect(self._locate_track)
        self._track_panel.track_delete.connect(self.remove_track)
        self._track_panel.track_duplicate.connect(self.duplicate_track)
        self._track_panel.track_rename.connect(self.rename_track_source)
        self._track_panel.track_gain_changed.connect(self._on_track_gain_changed)
        self._track_panel.track_position_changed.connect(self._on_track_position_changed)
        self._track_panel.track_enabled_toggled.connect(self.toggle_track)
        self._track_panel.batch_requested.connect(self.batch_tracks)
        self._splitter.addWidget(self._track_panel)

        # Splitter proportions: 22 : 56 : 22
        self._splitter.setSizes([320, 800, 320])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._left_panel.setMinimumWidth(260)
        self._left_panel.setMaximumWidth(480)
        self._track_panel.setMinimumWidth(280)
        self._track_panel.setMaximumWidth(480)

        central_layout.addWidget(self._splitter, 1)

        # Transport remains available after a preview render and is kept
        # separate from render progress so playback does not obscure the room.
        self._build_preview_transport(central_layout)

        # Task overlay
        self._overlay = TaskOverlay(self.room_view)

        # Bottom operation bar (Section 9)
        self._build_bottom_bar(central_layout)

        self.setCentralWidget(central)

        # Toast
        self._toast = Toast(self)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Render state
        self.renderer: Renderer | None = None
        self.render_thread: QThread | None = None
        self.render_worker: RenderWorker | None = None
        self.render_job: str | None = None
        self.render_output_dir: Path | None = None
        self._rerun_after_cleanup: tuple | None = None
        self.ai_thread: QThread | None = None
        self.ai_worker: AIAnalysisWorker | None = None
        self.dup_thread: QThread | None = None
        self.dup_worker: DuplicateWorker | None = None
        self._dup_source_track: TrackConfig | None = None

    def _build_left_panel(self) -> QWidget:
        """Build left parameter panel with collapsible cards."""
        # Wrap content in a scroll area so cards don't get squished when the
        # window is too short — prevents QDoubleSpinBox up/down buttons from
        # being clipped or overlapped.
        scroll = QScrollArea()
        self._left_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {Colors.BG_SECONDARY}; }}"
        )

        panel = QWidget()
        panel.setStyleSheet(f"background: {Colors.BG_SECONDARY};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)

        self._add_panel_group(layout, "空间", "空间声学参数")

        # Room card
        self._room_card = CollapsibleCard("房间参数", "设置虚拟房间的物理尺寸与网格")
        room_form = QFormLayout()
        room_form.setSpacing(10)
        self.room_spins: dict[str, QDoubleSpinBox | QSpinBox] = {}

        definitions = (
            ("length_m", "房间长度 (m)", 0.001, 100, 3),
            ("width_m", "房间宽度 (m)", 0.001, 100, 3),
            ("height_m", "房间高度 (m)", 0.001, 30, 3),
            ("grid_step_m", "网格步进 (m)", 0.01, 100, 3),
        )
        for key, label_text, minimum, maximum, decimals in definitions:
            spin = self._make_spin(key, label_text, minimum, maximum, decimals)
            room_form.addRow(label_text, spin)

        self._room_card.content_layout().addLayout(room_form)
        layout.addWidget(self._room_card)

        self._reverb_card = CollapsibleCard("混响参数", "RT60 与混响增益控制")
        reverb_form = QFormLayout()
        reverb_form.setSpacing(10)
        for key, label_text, minimum, maximum, decimals in (
            ("rt60_s", "RT60 (s)", 0.2, 2, 3),
            ("reverb_gain_db", "混响增益 (dB)", -30, 0, 1),
        ):
            spin = self._make_spin(key, label_text, minimum, maximum, decimals)
            reverb_form.addRow(label_text, spin)
        self._reverb_card.content_layout().addLayout(reverb_form)
        layout.addWidget(self._reverb_card)

        self._mic_card = CollapsibleCard("麦克风参数", "2～6 个等距麦克风阵列，保持立体声输出")
        mic_form = QFormLayout()
        mic_form.setSpacing(10)
        mic_count = ParameterIntSpinBox()
        mic_count.setMinimumWidth(140)
        mic_count.setMaximumWidth(190)
        mic_count.setRange(2, 6)
        mic_count.setAccessibleName("麦克风数量")
        mic_count.stepped.connect(lambda: self.apply_room_parameter("mic_count"))
        mic_count.committed.connect(lambda: self.apply_room_parameter("mic_count"))
        self.room_spins["mic_count"] = mic_count
        mic_form.addRow("麦克风数量", mic_count)
        for key, label_text, minimum, maximum, decimals in (
            ("spacing_m", "麦克风间距 (m)", 0.2, 3, 3),
            ("mic_height_m", "麦克风高度 (m)", 0.001, 30, 3),
        ):
            spin = self._make_spin(key, label_text, minimum, maximum, decimals)
            mic_form.addRow(label_text, spin)
        self._mic_card.content_layout().addLayout(mic_form)
        layout.addWidget(self._mic_card)

        self._add_panel_group(layout, "输入", "输入与时间线")
        self._midi_card = CollapsibleCard("MIDI 时间线", "按轨道分配颤音与音符起音偏移的参考时间线")
        self._midi_card.setAccessibleName("参考 MIDI 折叠面板")
        self.naturalization_enabled = QCheckBox("启用随机偏移")
        self.naturalization_enabled.setToolTip("基于分配给各轨道的 MIDI 进行每音符起音随机微偏移")
        self.naturalization_enabled.toggled.connect(self.set_naturalization_enabled)
        self._midi_card.content_layout().addWidget(self.naturalization_enabled)

        self.midi_list_layout = QVBoxLayout()
        self.midi_list_layout.setSpacing(Spacing.SM)
        self._midi_card.content_layout().addLayout(self.midi_list_layout)

        self.midi_empty_frame = QFrame()
        self.midi_empty_frame.setObjectName("midiDropTarget")
        empty_layout = QVBoxLayout(self.midi_empty_frame)
        self.reference_midi_input = MidiPathEdit()
        self.reference_midi_input.setPlaceholderText("拖入 MIDI 或选择文件")
        self.reference_midi_input.setAccessibleName("参考 MIDI 文件")
        self.reference_midi_input.midi_dropped.connect(self.add_reference_midi)
        empty_layout.addWidget(self.reference_midi_input)
        self.reference_midi_browse_button = QPushButton("选择")
        self.reference_midi_browse_button.setObjectName("secondary")
        self.reference_midi_browse_button.clicked.connect(self.choose_reference_midi)
        empty_layout.addWidget(self.reference_midi_browse_button)
        self.midi_list_layout.addWidget(self.midi_empty_frame)

        self.add_midi_button = QPushButton("+ 添加 MIDI")
        self.add_midi_button.setObjectName("secondary")
        self.add_midi_button.clicked.connect(self.choose_reference_midi)
        self._midi_card.content_layout().addWidget(self.add_midi_button)

        # Kept as a compatibility hook for extensions written against the
        # former single-MIDI widget. Per-item remove buttons are rendered below.
        self.reference_midi_clear_button = QToolButton(self)
        self.reference_midi_clear_button.hide()
        self.reference_midi_clear_button.clicked.connect(self.clear_reference_midi)

        self.naturalization_status = QLabel("未选择 MIDI")
        self.naturalization_status.setWordWrap(True)
        self.naturalization_status.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;"
        )
        self._midi_card.content_layout().addWidget(self.naturalization_status)
        params = QLabel("参数：σ=2 ms · ±5 ms · 截断正态分布")
        params.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;")
        self._midi_card.content_layout().addWidget(params)
        layout.addWidget(self._midi_card)

        self._add_panel_group(layout, "AI", "AI 方案")
        self._ai_card = CollapsibleCard("AI 方案", "分析全部启用音轨并应用空间与混响建议")
        self._ai_card.setStyleSheet(
            "QFrame#collapsibleCard { background: #f0f7ff; border: 1px solid #0969da; border-radius: 6px; }"
        )
        self.ai_button = QPushButton("发送给 AI")
        self.ai_button.setObjectName("primary")
        self.ai_button.setStyleSheet("color: #ffffff;")
        self.ai_button.setAccessibleName("发送给 AI 分析")
        self.ai_button.clicked.connect(self.analyze_with_ai)
        self._ai_card.content_layout().addWidget(self.ai_button)
        ai_secondary_row = QHBoxLayout()
        ai_secondary_row.setSpacing(Spacing.SM)
        self.ai_settings_button = QPushButton("AI 设置")
        self.ai_settings_button.setObjectName("secondary")
        self.ai_settings_button.clicked.connect(self.ai_settings)
        ai_secondary_row.addWidget(self.ai_settings_button)
        self.ai_customize_button = QPushButton("定制对话")
        self.ai_customize_button.setObjectName("secondary")
        self.ai_customize_button.clicked.connect(self.customize_with_ai)
        ai_secondary_row.addWidget(self.ai_customize_button)
        self.ai_solution_button = QPushButton("切换 AI 方案")
        self.ai_solution_button.setObjectName("secondary")
        self.ai_solution_button.setAccessibleName("切换 AI 方案")
        self.ai_solution_button.setEnabled(False)
        self.ai_solution_button.clicked.connect(self.choose_ai_suggestion)
        ai_secondary_row.addWidget(self.ai_solution_button)
        self._ai_card.content_layout().addLayout(ai_secondary_row)
        self.ai_solution_status = QLabel("尚未生成方案")
        self.ai_solution_status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;")
        self._ai_card.content_layout().addWidget(self.ai_solution_status)
        layout.addWidget(self._ai_card)

        self._add_panel_group(layout, "输出", "输出参数")
        self._bus_card = CollapsibleCard("总线响度", "仅控制预览和混音总输出，不会发送给 AI，也不影响分轨导出")
        bus_form = QFormLayout()
        bus_form.setSpacing(10)
        bus_spin = self._make_spin("bus_gain_db", "总线增益 (dB)", -24, 12, 1)
        bus_form.addRow("总线增益 (dB)", bus_spin)
        self._bus_card.content_layout().addLayout(bus_form)
        layout.addWidget(self._bus_card)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    @staticmethod
    def _add_panel_group(layout: QVBoxLayout, title: str, accessible_name: str) -> None:
        label = QLabel(title)
        label.setAccessibleName(accessible_name)
        label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; font-weight: 600; "
            f"border: none; padding: 6px 2px 0 2px;"
        )
        layout.addWidget(label)

    def _make_spin(self, key: str, label: str, minimum: float, maximum: float,
                   decimals: int) -> QDoubleSpinBox:
        spin = ParameterSpinBox()
        # Keep the editable field compact so the stepper buttons retain a
        # generous hit target in wide left panels.
        spin.setMinimumWidth(140)
        spin.setMaximumWidth(190)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1 if decimals else 1)
        spin.setAccessibleName(label)
        spin.stepped.connect(lambda k=key: self.apply_room_parameter(k))
        spin.committed.connect(lambda k=key: self.apply_room_parameter(k))
        self.room_spins[key] = spin
        return spin

    def _build_preview_transport(self, parent_layout: QVBoxLayout):
        """Build a persistent transport for the most recently rendered audio."""
        transport = QFrame()
        transport.setObjectName("previewTransport")
        transport.setStyleSheet(
            "QFrame#previewTransport { background: #ffffff; border-top: 1px solid #d1d9e0; }"
        )
        layout = QHBoxLayout(transport)
        layout.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)
        layout.setSpacing(Spacing.SM)

        self.preview_source_button = QPushButton("选择试听音源")
        self.preview_source_button.setObjectName("secondary")
        self.preview_source_button.setAccessibleName("选择混音或分轨试听")
        self.preview_source_button.clicked.connect(self._open_preview_source_menu)
        layout.addWidget(self.preview_source_button)

        self.preview_back_button = QPushButton("<< 5s")
        self.preview_back_button.setObjectName("small")
        self.preview_back_button.setAccessibleName("后退五秒")
        self.preview_back_button.clicked.connect(lambda: self._seek_preview_seconds(-5))
        layout.addWidget(self.preview_back_button)

        self.preview_play_button = QPushButton("Play")
        self.preview_play_button.setObjectName("primary")
        self.preview_play_button.setAccessibleName("播放或暂停试听")
        self.preview_play_button.clicked.connect(self._toggle_preview_playback)
        layout.addWidget(self.preview_play_button)

        self.preview_forward_button = QPushButton("5s >>")
        self.preview_forward_button.setObjectName("small")
        self.preview_forward_button.setAccessibleName("前进五秒")
        self.preview_forward_button.clicked.connect(lambda: self._seek_preview_seconds(5))
        layout.addWidget(self.preview_forward_button)

        self.preview_time_label = QLabel("00:00 / 00:00")
        self.preview_time_label.setMinimumWidth(92)
        self.preview_time_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-family: {Fonts.MONO};")
        layout.addWidget(self.preview_time_label)

        self.preview_slider = PreviewSlider()
        self.preview_slider.setAccessibleName("试听进度")
        self.preview_slider.sliderPressed.connect(self._preview_scrub_started)
        self.preview_slider.sliderMoved.connect(self._preview_scrub_moved)
        self.preview_slider.sliderReleased.connect(self._preview_scrub_finished)
        self.preview_slider.track_clicked.connect(self._preview_seek_to_milliseconds)
        layout.addWidget(self.preview_slider, 1)

        self.preview_time_input = QLineEdit()
        self.preview_time_input.setPlaceholderText("mm:ss")
        self.preview_time_input.setMaximumWidth(72)
        self.preview_time_input.setAccessibleName("跳转时间，格式为 分钟:秒")
        self.preview_time_input.returnPressed.connect(self._jump_preview_time)
        layout.addWidget(self.preview_time_input)

        self.preview_jump_button = QPushButton("跳转")
        self.preview_jump_button.setObjectName("small")
        self.preview_jump_button.clicked.connect(self._jump_preview_time)
        layout.addWidget(self.preview_jump_button)

        self._preview_data = None
        self._preview_rate = 0
        self._preview_frame = 0
        self._preview_started_frame = 0
        self._preview_started_at = 0.0
        self._preview_playing = False
        self._preview_was_playing = False
        self._preview_sources: list[tuple[str, Path]] = []
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._update_preview_progress)
        self._set_preview_controls_enabled(False)

        parent_layout.addWidget(transport)

    def _set_preview_controls_enabled(self, enabled: bool):
        for widget in (
            self.preview_source_button, self.preview_back_button, self.preview_play_button,
            self.preview_forward_button, self.preview_slider, self.preview_time_input,
            self.preview_jump_button,
        ):
            widget.setEnabled(enabled)

    @staticmethod
    def _format_preview_time(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

    def _sync_preview_time(self, frame: int | None = None):
        if self._preview_data is None or not self._preview_rate:
            self.preview_time_label.setText("00:00 / 00:00")
            return
        if frame is not None:
            self._preview_frame = max(0, min(int(frame), len(self._preview_data)))
        self.preview_time_label.setText(
            f"{self._format_preview_time(self._preview_frame / self._preview_rate)} / "
            f"{self._format_preview_time(len(self._preview_data) / self._preview_rate)}"
        )
        milliseconds = round(self._preview_frame * 1000 / self._preview_rate)
        self.preview_slider.blockSignals(True)
        self.preview_slider.setValue(milliseconds)
        self.preview_slider.blockSignals(False)

    def _refresh_preview_sources(self, project_dir: Path, selected: Path | None = None):
        sources: list[tuple[str, Path]] = []
        preview_path = project_dir / "preview" / "preview.wav"
        mix_path = project_dir / "Mixdown" / "mix.wav"
        if preview_path.is_file():
            sources.append(("1 混音预览", preview_path))
        preview_stems_dir = project_dir / "preview" / "stems"
        if preview_stems_dir.is_dir():
            # Only surface stems belonging to the current project; a previous
            # preview may have files for tracks that were since removed.
            for track in self.project.tracks:
                if not track.enabled:
                    continue
                path = preview_stems_dir / f"{track.track_id}.wav"
                if path.is_file():
                    sources.append((f"2 分轨预览 - {track.track_id}", path))
        if mix_path.is_file():
            sources.append(("混音导出", mix_path))
        stems_dir = project_dir / "Stems"
        if stems_dir.is_dir():
            sources.extend((f"分轨 - {path.stem}", path) for path in sorted(stems_dir.glob("*.wav")))
        self._preview_sources = sources
        if selected is not None and selected.is_file():
            self._set_preview_source(selected)

    def _open_preview_source_menu(self):
        if not self._preview_sources:
            return
        menu = QMenu(self)
        for label, path in self._preview_sources:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.preview_source_button.property("previewPath") == str(path))
            action.triggered.connect(lambda checked=False, selected_path=path: self._set_preview_source(selected_path))
        menu.exec(self.preview_source_button.mapToGlobal(QPoint(0, -menu.sizeHint().height())))

    def _set_preview_source(self, path: Path):
        try:
            data, rate = load_playback_audio(path)
        except ChoirError as exc:
            self.error(exc)
            return
        stop_playback()
        self._preview_timer.stop()
        self._preview_data = data
        self._preview_rate = rate
        self._preview_frame = 0
        self._preview_playing = False
        self.preview_play_button.setText("Play")
        label = next((item_label for item_label, item_path in self._preview_sources if item_path == path), path.stem)
        self.preview_source_button.setText(label)
        self.preview_source_button.setProperty("previewPath", str(path))
        self.preview_slider.setRange(0, max(1, round(len(data) * 1000 / rate)))
        self._set_preview_controls_enabled(True)
        self._sync_preview_time(0)
        self._status.showMessage(f"已载入试听：{label}", 5000)

    def _start_preview_playback(self):
        if self._preview_data is None or not self._preview_rate:
            return
        if self._preview_frame >= len(self._preview_data):
            self._preview_frame = 0
        try:
            play_audio_segment(self._preview_data, self._preview_rate, self._preview_frame)
        except ChoirError as exc:
            self.error(exc)
            return
        self._preview_started_frame = self._preview_frame
        self._preview_started_at = time.monotonic()
        self._preview_playing = True
        self.preview_play_button.setText("Pause")
        self._preview_timer.start()

    def _pause_preview_playback(self):
        if not self._preview_playing:
            return
        self._update_preview_progress()
        stop_playback()
        self._preview_timer.stop()
        self._preview_playing = False
        self.preview_play_button.setText("Play")

    def _toggle_preview_playback(self):
        if self._preview_playing:
            self._pause_preview_playback()
        else:
            self._start_preview_playback()

    def _update_preview_progress(self):
        if not self._preview_playing or self._preview_data is None:
            return
        frame = self._preview_started_frame + round(
            (time.monotonic() - self._preview_started_at) * self._preview_rate
        )
        if frame >= len(self._preview_data):
            self._sync_preview_time(len(self._preview_data))
            stop_playback()
            self._preview_timer.stop()
            self._preview_playing = False
            self.preview_play_button.setText("Play")
            return
        self._sync_preview_time(frame)

    def _seek_preview_seconds(self, seconds: int):
        if self._preview_data is None:
            return
        self._update_preview_progress()
        resume = self._preview_playing
        if resume:
            self._pause_preview_playback()
        self._sync_preview_time(self._preview_frame + seconds * self._preview_rate)
        if resume:
            self._start_preview_playback()

    def _preview_scrub_started(self):
        self._preview_was_playing = self._preview_playing
        if self._preview_was_playing:
            self._pause_preview_playback()

    def _preview_scrub_moved(self, milliseconds: int):
        if self._preview_rate:
            self.preview_time_label.setText(
                f"{self._format_preview_time(milliseconds / 1000)} / "
                f"{self._format_preview_time(len(self._preview_data) / self._preview_rate)}"
            )

    def _preview_scrub_finished(self):
        self._preview_seek_to_milliseconds(self.preview_slider.value())
        self._preview_was_playing = False

    def _preview_seek_to_milliseconds(self, milliseconds: int):
        if self._preview_data is None or not self._preview_rate:
            return
        self._update_preview_progress()
        resume = self._preview_playing or self._preview_was_playing
        if self._preview_playing:
            self._pause_preview_playback()
        self._sync_preview_time(round(milliseconds * self._preview_rate / 1000))
        if resume:
            self._start_preview_playback()

    def _jump_preview_time(self):
        value = self.preview_time_input.text().strip()
        try:
            parts = value.split(":")
            if len(parts) == 2:
                minutes, remaining_seconds = int(parts[0]), int(parts[1])
                if minutes < 0 or not 0 <= remaining_seconds < 60:
                    raise ValueError
                seconds = minutes * 60 + remaining_seconds
            elif len(parts) == 1:
                seconds = int(parts[0])
            else:
                raise ValueError
            if seconds < 0:
                raise ValueError
        except ValueError:
            self._status.showMessage("请输入秒数或 mm:ss，例如 01:30", 5000)
            return
        if self._preview_data is None:
            return
        self._update_preview_progress()
        resume = self._preview_playing
        if resume:
            self._pause_preview_playback()
        self._sync_preview_time(seconds * self._preview_rate)
        if resume:
            self._start_preview_playback()

    def _build_bottom_bar(self, parent_layout: QVBoxLayout):
        """Bottom operation bar (Section 9)."""
        bar = QFrame()
        bar.setObjectName("bottomBar")
        bar.setFixedHeight(56)
        bar.setStyleSheet(
            f"QFrame#bottomBar {{"
            f"  background: {Colors.BG_SECONDARY};"
            f"  border-top: 1px solid {Colors.BORDER};"
            f"}}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)
        layout.setSpacing(Spacing.MD)

        # Render actions
        self.preview_button = QPushButton("▶ 预览")
        self.preview_button.setObjectName("secondary")
        self.preview_button.clicked.connect(lambda: self.render("preview"))
        self.preview_button.setAccessibleName("预览")
        layout.addWidget(self.preview_button)

        self.stems_button = QPushButton("分轨导出")
        self.stems_button.setObjectName("secondary")
        self.stems_button.clicked.connect(lambda: self.render("stems"))
        self.stems_button.setAccessibleName("分轨导出")
        layout.addWidget(self.stems_button)

        self.mix_button = QPushButton("混音导出")
        self.mix_button.setObjectName("primary")
        self.mix_button.clicked.connect(lambda: self.render("mix"))
        self.mix_button.setAccessibleName("混音导出")
        layout.addWidget(self.mix_button)

        layout.addStretch()

        # Right: stop + progress + status
        self.stop_button = QPushButton("■ 停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.cancel_active_task)
        self.stop_button.setAccessibleName("停止/取消")
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(180)
        self.progress.setAccessibleName("任务进度")
        layout.addWidget(self.progress)

        self._progress_label = QLabel("● 准备就绪")
        self._progress_label.setStyleSheet(
            f"color: {Colors.SUCCESS}; font-size: 11px; font-weight: 400;"
        )
        layout.addWidget(self._progress_label)

        parent_layout.addWidget(bar)

    # PySide only guarantees queued delivery when a receiver is registered in
    # the concrete QObject's meta-object. Workflow mixins are plain Python
    # classes, so their callbacks need these concrete Qt slot entry points.
    @Slot(int, str)
    def _dup_progress(self, percent: int, message: str):
        ProjectWorkflowMixin._dup_progress(self, percent, message)

    @Slot(object)
    def _dup_completed(self, paths):
        ProjectWorkflowMixin._dup_completed(self, paths)

    @Slot(object)
    def _dup_failed(self, error):
        ProjectWorkflowMixin._dup_failed(self, error)

    @Slot()
    def _dup_cleanup(self):
        ProjectWorkflowMixin._dup_cleanup(self)

    @Slot(object)
    def _ai_completed(self, result):
        AIWorkflowMixin._ai_completed(self, result)

    @Slot(object)
    def _ai_failed(self, error):
        AIWorkflowMixin._ai_failed(self, error)

    @Slot()
    def _ai_cleanup(self):
        AIWorkflowMixin._ai_cleanup(self)

    @Slot(int, str)
    def _render_progress(self, percent: int, message: str):
        RenderWorkflowMixin._render_progress(self, percent, message)

    @Slot(str)
    def _render_notice(self, message: str):
        RenderWorkflowMixin._render_notice(self, message)

    @Slot(object)
    def _render_complete(self, result):
        RenderWorkflowMixin._render_complete(self, result)

    @Slot(object, str, object)
    def _render_failed(self, error, render_job, output_dir):
        RenderWorkflowMixin._render_failed(self, error, render_job, output_dir)

    @Slot()
    def _render_cleanup(self):
        RenderWorkflowMixin._render_cleanup(self)
