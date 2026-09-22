"""Qt desktop client for OpenAI-compatible streaming chat-completion APIs.

PySide6 is Qt for Python and is API-compatible with PyQt for the widgets used
here.  It is already the project's Qt binding, so no second Qt binding is
required.
"""
from __future__ import annotations

import sys
import threading
from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QSplitter, QTextEdit, QVBoxLayout,
    QWidget,
)

from mama.model.desktop_model_config import DesktopModelConfig, DesktopModelStore
from mama.model.foreign_llm_model import OpenAICompatibleModel


class LLMWorker(QThread):
    reasoning_delta = Signal(str)
    content_delta = Signal(str)
    completed = Signal(bool, str)

    def __init__(self, config: DesktopModelConfig, prompt: str) -> None:
        super().__init__()
        self.config = config
        self.prompt = prompt
        self.stop_event = threading.Event()

    def cancel(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        model = OpenAICompatibleModel(
            self.config.url, self.config.api_key, self.config.model,
            self.config.temperature, self.config.max_tokens, self.config.timeout,
        )
        result = model.call_llm(
            self.prompt, self.stop_event,
            on_reasoning=self.reasoning_delta.emit,
            on_content=self.content_delta.emit,
        )
        message = result.error or "调用完成"
        self.completed.emit(result.success, message)


class ModelSettingsDialog(QDialog):
    """Edit one named endpoint configuration without exposing it in the main UI."""

    def __init__(self, config: DesktopModelConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("模型配置")
        self.setMinimumWidth(540)
        form = QFormLayout(self)
        self.name = QLineEdit(config.name)
        self.url = QLineEdit(config.url)
        self.api_key = QLineEdit(config.api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.model = QLineEdit(config.model)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(config.temperature)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 1_000_000)
        self.max_tokens.setValue(config.max_tokens)
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(1.0, 3600.0)
        self.timeout.setValue(config.timeout)
        form.addRow("配置名称", self.name)
        form.addRow("完整 API URL", self.url)
        form.addRow("API Key", self.api_key)
        form.addRow("模型名称", self.model)
        form.addRow("Temperature", self.temperature)
        form.addRow("Max tokens", self.max_tokens)
        form.addRow("超时（秒）", self.timeout)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def value(self) -> DesktopModelConfig:
        return DesktopModelConfig(
            name=self.name.text().strip(), url=self.url.text().strip(),
            api_key=self.api_key.text().strip(), model=self.model.text().strip(),
            temperature=self.temperature.value(), max_tokens=self.max_tokens.value(),
            timeout=self.timeout.value(),
        )

    def accept(self) -> None:
        value = self.value()
        if not value.name or not value.url or not value.model:
            QMessageBox.warning(self, "配置不完整", "配置名称、API URL 和模型名称不能为空。")
            return
        super().accept()


class ChatWindow(QMainWindow):
    def __init__(self, store: Optional[DesktopModelStore] = None) -> None:
        super().__init__()
        self.store = store or DesktopModelStore()
        self.active_name, self.models = self.store.load()
        self.worker: Optional[LLMWorker] = None
        self.setWindowTitle("Mama AI 模型调用工具")
        self.resize(1200, 720)
        self._build_ui()
        self._reload_model_selector()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("模型配置："))
        self.model_selector = QComboBox()
        self.model_selector.currentTextChanged.connect(self._model_changed)
        controls.addWidget(self.model_selector, 1)
        self.settings_button = QPushButton("编辑配置")
        self.settings_button.clicked.connect(self._edit_model)
        controls.addWidget(self.settings_button)
        self.add_button = QPushButton("新增配置")
        self.add_button.clicked.connect(self._add_model)
        controls.addWidget(self.add_button)
        self.delete_button = QPushButton("删除配置")
        self.delete_button.clicked.connect(self._delete_model)
        controls.addWidget(self.delete_button)
        self.start_button = QPushButton("开始调用")
        self.start_button.clicked.connect(self.start_api_call)
        controls.addWidget(self.start_button)
        self.stop_button = QPushButton("终止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_api_call)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        top = QSplitter()
        self.input_text = self._text_panel("输入 (Input)", editable=True)
        self.thinking_text = self._text_panel("思考过程 / 日志 (Log)")
        top.addWidget(self.input_text)
        top.addWidget(self.thinking_text)
        top.setSizes([700, 400])
        all_panels = QSplitter()
        all_panels.setOrientation(Qt.Orientation.Vertical)
        all_panels.addWidget(top)
        self.output_text = self._text_panel("输出 (Output)")
        all_panels.addWidget(self.output_text)
        all_panels.setSizes([300, 350])
        layout.addWidget(all_panels, 1)
        self.statusBar().showMessage(f"配置文件：{self.store.path}")

    @staticmethod
    def _text_panel(title: str, editable: bool = False) -> QFrame:
        panel = QFrame()
        box = QVBoxLayout(panel)
        box.setContentsMargins(4, 4, 4, 4)
        box.addWidget(QLabel(title))
        text = QTextEdit()
        text.setReadOnly(not editable)
        box.addWidget(text)
        panel.text_edit = text  # type: ignore[attr-defined]
        return panel

    @property
    def input_editor(self) -> QTextEdit:
        return self.input_text.text_edit  # type: ignore[attr-defined]

    @property
    def thinking_editor(self) -> QTextEdit:
        return self.thinking_text.text_edit  # type: ignore[attr-defined]

    @property
    def output_editor(self) -> QTextEdit:
        return self.output_text.text_edit  # type: ignore[attr-defined]

    def _reload_model_selector(self) -> None:
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems([config.name for config in self.models])
        self.model_selector.setCurrentText(self.active_name)
        self.model_selector.blockSignals(False)
        self.delete_button.setEnabled(len(self.models) > 1)

    def _model_changed(self, name: str) -> None:
        if name and name != self.active_name:
            self.active_name = name
            self._save_models()

    def _current_model(self) -> DesktopModelConfig:
        return next(config for config in self.models if config.name == self.active_name)

    def _save_models(self) -> bool:
        try:
            self.store.save(self.active_name, self.models)
            self.statusBar().showMessage(f"已保存到 {self.store.path}", 3000)
            return True
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return False

    def _edit_model(self) -> None:
        old = self._current_model()
        dialog = ModelSettingsDialog(old, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = dialog.value()
        if updated.name != old.name and any(item.name == updated.name for item in self.models):
            QMessageBox.warning(self, "保存失败", "配置名称已存在。")
            return
        self.models[self.models.index(old)] = updated
        self.active_name = updated.name
        if self._save_models():
            self._reload_model_selector()

    def _add_model(self) -> None:
        dialog = ModelSettingsDialog(DesktopModelConfig(name="新配置"), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_model = dialog.value()
        if any(item.name == new_model.name for item in self.models):
            QMessageBox.warning(self, "新增失败", "配置名称已存在。")
            return
        self.models.append(new_model)
        self.active_name = new_model.name
        if self._save_models():
            self._reload_model_selector()

    def _delete_model(self) -> None:
        if len(self.models) == 1:
            return
        current = self._current_model()
        if QMessageBox.question(self, "删除配置", f"确定删除“{current.name}”吗？") != QMessageBox.StandardButton.Yes:
            return
        self.models.remove(current)
        self.active_name = self.models[0].name
        if self._save_models():
            self._reload_model_selector()

    def start_api_call(self) -> None:
        prompt = self.input_editor.toPlainText().strip()
        config = self._current_model()
        if not prompt:
            QMessageBox.warning(self, "无法调用", "输入内容不能为空。")
            return
        if not config.api_key:
            QMessageBox.warning(self, "无法调用", "请先在“编辑配置”中填写 API Key。")
            return
        self.output_editor.clear()
        self.thinking_editor.clear()
        self._append_log(f"准备调用模型：{config.model}")
        self.worker = LLMWorker(config, prompt)
        self.worker.reasoning_delta.connect(self._append_reasoning)
        self.worker.content_delta.connect(self._append_content)
        self.worker.completed.connect(self._call_completed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.settings_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.worker.start()

    def stop_api_call(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._append_log("正在发送终止信号…")
            self.stop_button.setEnabled(False)

    def _append_log(self, message: str) -> None:
        self.thinking_editor.append(message)

    def _append_reasoning(self, text: str) -> None:
        self.thinking_editor.moveCursor(QTextCursor.MoveOperation.End)
        self.thinking_editor.insertPlainText(text)
        self.thinking_editor.ensureCursorVisible()

    def _append_content(self, text: str) -> None:
        self.output_editor.moveCursor(QTextCursor.MoveOperation.End)
        self.output_editor.insertPlainText(text)
        self.output_editor.ensureCursorVisible()

    def _call_completed(self, success: bool, message: str) -> None:
        self._append_log("\n" + ("调用完成。" if success else f"调用结束：{message}"))
        if not success and message != "用户已终止请求":
            QMessageBox.critical(self, "API 调用失败", message)
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.settings_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.delete_button.setEnabled(len(self.models) > 1)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
