from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget, QVBoxLayout,
    QWidget,
)

from ..ai import AIClient, AIConfig
from ..errors import ChoirError
from ..models import MidiAssignment, NaturalizationConfig, ProjectConfig
from ..project_io import MEDIA_DIR, copy_to_media
from .theme import Colors, DEFAULT_PROJECT_DIR, Fonts, Spacing
from .workers import AIChatWorker


class AISettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 设置")
        self.setAccessibleName("AI 设置")
        self.setMinimumSize(420, 380)
        self.resize(520, 440)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Info banner
        info = QLabel("🔑 API 密钥仅保存在当前会话或 Windows 凭据管理器，不会写入工程 JSON。")
        info.setWordWrap(True)
        info.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding: 8px 12px;"
            f" background: {Colors.BG_SECONDARY}; border-radius: 4px;"
        )
        main_layout.addWidget(info)

        # Connection group
        conn_group = QGroupBox("连接设置")
        conn_layout = QFormLayout(conn_group)
        conn_layout.setSpacing(10)

        self.provider = QComboBox()
        self.provider.addItems(["gemini_native_api", "aggregator_openai_compatible"])
        self.provider.setAccessibleName("接入方式")
        conn_layout.addRow("接入方式", self.provider)

        self.base_url = QLineEdit("https://generativelanguage.googleapis.com/v1beta")
        self.base_url.setAccessibleName("API 地址")
        conn_layout.addRow("API 地址", self.base_url)

        main_layout.addWidget(conn_group)

        # Auth group
        auth_group = QGroupBox("认证设置")
        auth_layout = QFormLayout(auth_group)
        auth_layout.setSpacing(10)

        key_row = QHBoxLayout()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setAccessibleName("API 密钥")
        key_row.addWidget(self.api_key)

        self._show_key_btn = QPushButton("👁")
        self._show_key_btn.setFixedWidth(32)
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self._show_key_btn)

        key_widget = QWidget()
        key_widget.setLayout(key_row)
        key_widget.layout().setContentsMargins(0, 0, 0, 0)
        auth_layout.addRow("API 密钥", key_widget)

        self._auth_hint = QLabel("密钥仅保存在当前会话，不会写入工程。")
        self._auth_hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;")
        auth_layout.addRow("", self._auth_hint)

        main_layout.addWidget(auth_group)

        # Model group
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(10)

        self.model = QComboBox()
        self.model.setAccessibleName("模型")
        self.model.setPlaceholderText("请先拉取模型")
        model_layout.addRow("模型", self.model)

        self._model_status = QLabel("")
        self._model_status.setStyleSheet(f"font-size: 11px; border: none;")
        self._model_status.setVisible(False)

        model_btn_row = QHBoxLayout()
        self.fetch_btn = QPushButton("拉取模型")
        self.fetch_btn.setObjectName("primary")
        self.fetch_btn.setFixedHeight(32)
        self.fetch_btn.clicked.connect(self.fetch_models)
        model_btn_row.addWidget(self.fetch_btn)

        self._test_btn = QPushButton("测试连接")
        self._test_btn.setObjectName("secondary")
        self._test_btn.setFixedHeight(32)
        self._test_btn.clicked.connect(self.test_connection)
        model_btn_row.addWidget(self._test_btn)
        model_btn_row.addStretch()

        model_layout.addRow("", model_btn_row)
        model_layout.addRow("", self._model_status)

        main_layout.addWidget(model_group)

        # Error label for field-level validation
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(
            f"color: {Colors.DANGER}; font-size: 11px; font-weight: 600;"
        )
        self._error_label.setVisible(False)
        main_layout.addWidget(self._error_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        # Signal connections
        self.provider.currentTextChanged.connect(self._provider_changed)

    def _toggle_key_visibility(self, checked: bool):
        self.api_key.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )

    def _provider_changed(self, value: str):
        if value == "gemini_native_api":
            self.base_url.setText("https://generativelanguage.googleapis.com/v1beta")
        else:
            self.base_url.setText("https://api.example.com/v1")

    def _validate(self) -> bool:
        """Field-level validation. Returns True if valid."""
        errors = []
        if not self.base_url.text().strip():
            errors.append("API 地址不能为空")
        if not self.api_key.text():
            errors.append("API 密钥不能为空")
        if errors:
            self._error_label.setText("\n".join(errors))
            self._error_label.setVisible(True)
            return False
        self._error_label.setVisible(False)
        return True

    def _on_accept(self):
        if not self._validate():
            return
        if not self.model.currentText():
            # Allow but warn
            pass
        self.accept()

    def config(self) -> AIConfig:
        return AIConfig(
            self.provider.currentText(),
            self.base_url.text().strip(),
            self.api_key.text(),
            self.model.currentText(),
        )

    def fetch_models(self):
        if not self._validate():
            return
        self._model_status.setVisible(True)
        self._model_status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;")
        self._model_status.setText("正在拉取模型列表…")
        self.fetch_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            models, fetched_at = AIClient(self.config()).fetch_models()
            self.model.clear()
            self.model.addItems(models)
            self._model_status.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 11px; border: none;")
            self._model_status.setText(f"✓ 已获取 {len(models)} 个模型 — {fetched_at}")
        except ChoirError as exc:
            self._model_status.setStyleSheet(f"color: {Colors.DANGER}; font-size: 11px; border: none;")
            self._model_status.setText(f"✗ {exc}")
        finally:
            self.fetch_btn.setEnabled(True)

    def test_connection(self):
        if not self._validate():
            return
        self._model_status.setVisible(True)
        self._model_status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;")
        self._model_status.setText("正在测试连接…")
        self._test_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            models, _ = AIClient(self.config()).fetch_models()
            self._model_status.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 11px; border: none;")
            self._model_status.setText(f"✓ 连接成功，{len(models)} 个模型可用")
        except ChoirError as exc:
            self._model_status.setStyleSheet(f"color: {Colors.DANGER}; font-size: 11px; border: none;")
            self._model_status.setText(f"✗ 连接失败：{exc}")
        finally:
            self._test_btn.setEnabled(True)


class AISuggestionDialog(QDialog):
    """Review an AI response and explicitly choose one option to apply."""

    def __init__(self, recommendations: list[dict], parent=None):
        super().__init__(parent)
        self._recommendations = recommendations
        self.setWindowTitle("选择 AI 方案")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel("AI 返回了多套方案。选择一套后才会修改工程；应用后可从 AI 菜单撤销。"))

        self.selector = QComboBox()
        self.selector.addItems([recommendation["name"] for recommendation in recommendations])
        self.selector.currentIndexChanged.connect(self._show_current)
        layout.addWidget(self.selector)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setStyleSheet(
            f"background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER};"
            " border-radius: 4px; padding: 10px;"
        )
        layout.addWidget(self.details)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        # ApplyRole does not emit QDialogButtonBox.accepted(), unlike OK.
        # Connect it explicitly so selecting an option actually returns
        # QDialog.Accepted to the caller.
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._show_current(0)

    def _show_current(self, index: int):
        recommendation = self._recommendations[index]
        room = recommendation["room"]
        microphone = recommendation["microphone"]
        self.details.setText(
            f"{recommendation['description']}\n\n"
            f"RT60：{room['rt60_s']:.2f}s    混响增益：{room['reverb_gain_db']:.1f} dB\n"
            f"麦克风：{microphone['count']} 个，间距 {microphone['spacing_m']:.2f}m，高度 {microphone['height_m']:.2f}m\n"
            f"歌手位置建议：{len(recommendation['singers'])} 条"
        )

    def selected_recommendation(self) -> dict:
        return self._recommendations[self.selector.currentIndex()]


class NaturalizationDialog(QDialog):
    def __init__(self, config: NaturalizationConfig, project_dir: Path | None, parent=None):
        super().__init__(parent)
        self.project_dir = project_dir
        self._assignments = deepcopy(config.assignments)
        self.setWindowTitle("随机偏移设置")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self.enabled = QCheckBox("启用随机偏移")
        self.enabled.setChecked(config.enabled)
        layout.addWidget(self.enabled)

        form = QFormLayout()
        midi_row = QHBoxLayout()
        self.midi_path = QLineEdit(config.midi_path or "")
        self.midi_path.setReadOnly(True)
        midi_row.addWidget(self.midi_path, 1)
        browse = QPushButton("选择 MIDI")
        browse.clicked.connect(self._browse_midi)
        midi_row.addWidget(browse)
        form.addRow("歌词 MIDI", midi_row)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(max(0, min(config.random_seed, 2_147_483_647)))
        form.addRow("随机种子", self.seed)
        layout.addLayout(form)

        rule = QLabel("每个音符独立使用截断正态分布：均值 0 ms，标准差 2 ms，范围 -5～5 ms。仅读取 MIDI 第一轨。")
        rule.setWordWrap(True)
        rule.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(rule)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_config)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择带歌词的 MIDI", str(self.project_dir or DEFAULT_PROJECT_DIR),
            "MIDI 文件 (*.mid *.midi)"
        )
        if path:
            selected = Path(path).resolve()
            # Copy MIDI into the project Media directory so the project is
            # self-contained and the reference won't break.
            if self.project_dir:
                media_copy = copy_to_media(self.project_dir, selected)
                try:
                    selected_text = str(media_copy.relative_to(self.project_dir.resolve()))
                except ValueError:
                    selected_text = str(media_copy)
            else:
                selected_text = str(selected)
            self.midi_path.setText(selected_text)
            self._assignments = [MidiAssignment(selected_text, [])]

    def config(self) -> NaturalizationConfig:
        result = NaturalizationConfig(
            enabled=self.enabled.isChecked(),
            assignments=deepcopy(self._assignments),
            random_seed=self.seed.value(),
        )
        result.validate()
        return result

    def _accept_config(self):
        try:
            self._result = self.config()
        except ChoirError as exc:
            QMessageBox.warning(self, exc.code, f"{exc.message}\n\n{exc.detail or ''}")
            return
        self.accept()

    def result_config(self) -> NaturalizationConfig:
        return self._result


class MidiAssignmentDialog(QDialog):
    """Choose the enabled tracks owned by one MIDI timeline."""

    def __init__(
        self,
        project: ProjectConfig,
        midi_name: str,
        selected_ids: list[str],
        midi_tracks,
        selected_midi_track_index: int | None,
        parent=None,
    ):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle(f"为 {midi_name} 分配轨道")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel("请选择此 MIDI 负责哪些启用轨道。已归属其他 MIDI 的轨道会自动移入这里。"))

        midi_track_form = QFormLayout()
        self.midi_track = QComboBox()
        self.midi_track.addItem("自动选择首条有音符音轨", None)
        for track in midi_tracks:
            if not track.note_count:
                continue
            pitch_range = _midi_pitch_range(track.lowest_pitch, track.highest_pitch)
            self.midi_track.addItem(
                f"音轨 {track.index + 1}（{track.note_count} 个音符，{pitch_range}）",
                track.index,
            )
        selected_index = self.midi_track.findData(selected_midi_track_index)
        self.midi_track.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        midi_track_form.addRow("MIDI 内部音轨：", self.midi_track)
        layout.addLayout(midi_track_form)

        self._checks: dict[str, QCheckBox] = {}
        for track in project.tracks:
            if not track.enabled:
                continue
            check = QCheckBox(f"{track.track_id}   {track.file_name}")
            check.setChecked(track.track_id in selected_ids)
            self._checks[track.track_id] = check
            track_row = QVBoxLayout()
            track_row.setSpacing(2)
            track_row.addWidget(check)
            owner = self._track_owner_name(track.track_id, midi_name)
            owner_text = (
                f"{track.track_id} 已分配到 {owner} MIDI"
                if owner else f"{track.track_id} 尚未分配 MIDI"
            )
            owner_label = QLabel(owner_text)
            owner_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
            track_row.addWidget(owner_label)
            layout.addLayout(track_row)

        actions = QHBoxLayout()
        select_all = QPushButton("全选")
        select_all.clicked.connect(lambda: [check.setChecked(True) for check in self._checks.values()])
        clear_all = QPushButton("清空")
        clear_all.clicked.connect(lambda: [check.setChecked(False) for check in self._checks.values()])
        actions.addWidget(select_all)
        actions.addWidget(clear_all)
        actions.addStretch(1)
        layout.addLayout(actions)
        hint = QLabel("未被任何 MIDI 选中的轨道将不参与音符偏移和逐音符颤音。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def track_ids(self) -> list[str]:
        return [track_id for track_id, check in self._checks.items() if check.isChecked()]

    def midi_track_index(self) -> int | None:
        return self.midi_track.currentData()

    def _track_owner_name(self, track_id: str, current_midi_name: str) -> str | None:
        assignments = self.project.naturalization.assignments
        if len(assignments) == 1 and not assignments[0].track_ids:
            return Path(current_midi_name).stem
        for assignment in assignments:
            if track_id in assignment.track_ids:
                return Path(assignment.midi_path).stem
        return None


def _midi_pitch_range(lowest: int | None, highest: int | None) -> str:
    if lowest is None or highest is None:
        return "无音符"
    def name(value: int) -> str:
        return f"{('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')[value % 12]}{value // 12 - 1}"
    return name(lowest) if lowest == highest else f"{name(lowest)} - {name(highest)}"


class AIConversationDialog(QDialog):
    """A small asynchronous conversation surface for feedback-driven plans."""

    def __init__(self, config: AIConfig, project: ProjectConfig, messages: list[dict[str, str]], parent=None):
        super().__init__(parent)
        self.config = config
        self.project = deepcopy(project)
        self.messages = deepcopy(messages)
        self.recommendations: list[dict] = []
        self.thread: QThread | None = None
        self.worker: AIChatWorker | None = None
        self.setWindowTitle("AI 定制方案对话")
        self.setMinimumSize(620, 500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        layout.addWidget(self.transcript, 1)
        for message in self.messages:
            self._append_message(message["role"], message["content"])
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("描述听感或希望调整的内容")
        self.input.returnPressed.connect(self.send)
        input_row.addWidget(self.input, 1)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self.send)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)
        self.status = QLabel("可针对声道能量、房间尺寸、混响和歌手位置提出调整")
        self.status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_button = buttons.addButton("保存最新方案", QDialogButtonBox.ButtonRole.AcceptRole)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_message(self, role: str, content: str):
        label = "你" if role == "user" else "AI"
        self.transcript.appendPlainText(f"{label}：{content}\n")
        cursor = self.transcript.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.transcript.setTextCursor(cursor)

    def send(self):
        content = self.input.text().strip()
        if not content or self.thread:
            return
        message = {"role": "user", "content": content}
        self.messages.append(message)
        self._append_message(**message)
        self.input.clear()
        self.input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status.setText("AI 正在生成定制方案…")
        self.thread = QThread(self)
        self.worker = AIChatWorker(self.config, self.project, deepcopy(self.messages))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._completed, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.completed.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._cleanup, Qt.ConnectionType.QueuedConnection)
        self.thread.start()

    @Slot(object)
    def _completed(self, result: dict):
        message = {"role": "assistant", "content": result["message"]}
        self.messages.append(message)
        self._append_message(**message)
        self.recommendations = result["recommendations"]
        self.save_button.setEnabled(True)
        self.status.setText(f"已生成 {len(self.recommendations)} 套定制方案")

    @Slot(object)
    def _failed(self, error: ChoirError):
        self.status.setText(f"生成失败：{error.message}")
        QMessageBox.warning(self, error.code, f"{error.message}\n\n{error.detail or ''}")

    @Slot()
    def _cleanup(self):
        thread = self.thread
        self.worker = None
        self.thread = None
        self.input.setEnabled(True)
        self.send_button.setEnabled(True)
        if thread is not None:
            thread.deleteLater()

    def reject(self):
        if self.thread:
            QMessageBox.information(self, "AI 正在回复", "请等待本轮回复完成后再关闭对话。")
            return
        super().reject()


class DuplicateTrackDialog(QDialog):
    """Let the user choose how many differentiated copies to generate."""

    def __init__(
        self, source_track, project_dir: Path | None, midi_path: str | None = None, parent=None,
    ):
        super().__init__(parent)
        self.source_track = source_track
        self.project_dir = project_dir
        self.midi_path = midi_path
        self.setWindowTitle("生成差异化副本")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)

        # ── Source info ──
        info_group = QGroupBox("源音频")
        info_layout = QFormLayout(info_group)
        info_layout.addRow("轨道：", QLabel(source_track.track_id))
        info_layout.addRow("文件：", QLabel(source_track.file_name))
        layout.addWidget(info_group)

        # ── Copy count ──
        count_group = QGroupBox("副本设置")
        count_layout = QFormLayout(count_group)
        self.copy_count = QSpinBox()
        self.copy_count.setRange(1, 64)
        self.copy_count.setValue(3)
        self.copy_count.setToolTip("生成 1-64 份音色差异化的副本")
        self.copy_count.valueChanged.connect(self._update_preview)
        count_layout.addRow("副本数量：", self.copy_count)

        self.preset = QComboBox()
        self.preset.addItem("一档 - 极轻微", 1)
        self.preset.addItem("二档 - 轻微", 2)
        self.preset.addItem("三档 - 自然合唱（默认）", 3)
        self.preset.addItem("四档 - 明显差异", 4)
        self.preset.addItem("五档 - 强差异", 5)
        self.preset.setCurrentIndex(2)
        self.preset.setToolTip("同一批副本共享此档位，但各自使用独立随机参数")
        self.preset.currentIndexChanged.connect(self._update_preview)
        count_layout.addRow("差异化预设：", self.preset)
        if midi_path:
            count_layout.addRow("颤音边界：", QLabel("使用工程 MIDI 的逐音符颤音"))
        else:
            count_layout.addRow("颤音边界：", QLabel("无 MIDI 时使用有声段备用模式"))

        layout.addWidget(count_group)

        # ── Output preview ──
        preview_group = QGroupBox("输出预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet(
            f"font-family: {Fonts.MONO}; font-size: {Fonts.SMALL_SIZE}px; "
            f"color: {Colors.TEXT_SECONDARY}; padding: 4px;"
        )
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        # ── Buttons ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("生成")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_preview()

    def _update_preview(self):
        n = self.copy_count.value()
        if self.project_dir:
            media = self.project_dir / MEDIA_DIR
            stem = Path(self.source_track.file_name).stem
        else:
            media = Path("<项目目录>/Media")
            stem = Path(self.source_track.file_name).stem
        preset_text = self.preset.currentText()
        lines = [f"预设：{preset_text}", "每份副本使用独立随机参数。", f"将在 {media} 中生成以下文件："]
        for i in range(1, n + 1):
            lines.append(f"  {stem}_副本{i}.wav")
        if n > 8:
            lines.append(f"  … 共 {n} 个文件")
        self.preview_label.setText("\n".join(lines))

    def result(self) -> dict:
        return {
            "copy_count": self.copy_count.value(),
            "preset_level": int(self.preset.currentData()),
        }
