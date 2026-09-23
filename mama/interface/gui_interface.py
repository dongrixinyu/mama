"""Modern Qt desktop interface for Mama.

The application opens directly on a clean, empty conversation.  Configuration is
kept behind the Settings item in the left navigation and is persisted by
DesktopModelStore.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QIcon, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from mama.model.desktop_model_config import (
    MAX_TOKENS_MAX, MAX_TOKENS_MIN, TEMPERATURE_MAX, TEMPERATURE_MIN,
    TIMEOUT_MAX, TIMEOUT_MIN, DesktopModelConfig, DesktopModelStore,
)
from mama.model.foreign_llm_model import OpenAICompatibleModel


class LLMWorker(QThread):
    reasoning_delta = Signal(str)
    content_delta = Signal(str)
    completed = Signal(bool, str)

    def __init__(self, config: DesktopModelConfig, prompt: str) -> None:
        super().__init__()
        self.config, self.prompt = config, prompt
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
        self.completed.emit(result.success, result.error or "Call completed")


class ChatWindow(QMainWindow):
    def __init__(self, store: Optional[DesktopModelStore] = None) -> None:
        super().__init__()
        self.store = store or DesktopModelStore()
        self.active_name, self.models = self.store.load()
        self.worker: Optional[LLMWorker] = None
        self._pending_model: Optional[DesktopModelConfig] = None
        self._settings_open = False
        self.setWindowTitle("Mama · AI Copilot")
        icon_dir = Path(__file__).resolve().parents[2] / "image" / "logo"
        # Provide both resolutions: Qt/Windows/macOS can select the 64px
        # variant for the title bar and the 128px variant for the taskbar/Dock.
        window_icon = QIcon()
        window_icon.addFile(str(icon_dir / "logo-64.png"))
        window_icon.addFile(str(icon_dir / "logo-128.png"))
        self.setWindowIcon(window_icon)
        self.resize(1280, 780)
        self._build_ui()
        self._load_form(self._current_model())

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #eeeeee; color: #172033; font-size: 14px; }
            #sidebar { background: #eeeeee; }
            #brand { color: #172033; font-size: 21px; font-weight: 700; padding: 18px 16px; }
            QPushButton.nav { color: #4b5563; background: transparent; border: 0; text-align: left; padding: 12px 16px; border-radius: 7px; }
            QPushButton.nav:hover, QPushButton.nav[active="true"] { color: #172033; background: #d9d9d9; }
            QPushButton.primary { background: #2563eb; color: white; border: 0; border-radius: 7px; padding: 10px 18px; font-weight: 600; }
            QPushButton#saveButton { background: rgb(132, 175, 35); color: white; border: 0; border-radius: 7px; padding: 10px 18px; font-weight: 600; }
            QPushButton#saveButton:hover { background: rgb(105, 151, 30); }
            QPushButton.secondary { background: white; border: 1px solid #d8dee9; border-radius: 7px; padding: 9px 16px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox { background: white; border: 1px solid #d8dee9; border-radius: 6px; padding: 8px; }
            QComboBox QLineEdit { border: 0; padding: 0; background: transparent; }
            QFormLayout QLabel { min-width: 125px; }
            QListWidget { background: white; border: 1px solid #e1e5ec; border-radius: 8px; padding: 5px; }
            QListWidget::item { padding: 11px 9px; border-radius: 5px; }
            QListWidget::item:selected { background: rgba(132, 175, 35, 0.1); color: #172033; }
            QLabel.title { font-size: 24px; font-weight: 700; }
            QLabel.muted { color: #718096; }
        """)
        root = QWidget(); self.setCentralWidget(root)
        outer = QHBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        side = QFrame(); side.setObjectName("sidebar"); side.setFixedWidth(220)
        sl = QVBoxLayout(side); sl.setContentsMargins(12, 12, 12, 12); sl.setSpacing(5)
        brand = QLabel("✦  MAMA"); brand.setObjectName("brand"); sl.addWidget(brand)
        self.chat_nav = self._nav("New Chat"); self.chat_nav.clicked.connect(self._return_to_chat); sl.addWidget(self.chat_nav)
        sl.addStretch()
        self.settings_nav = self._nav("Settings"); self.settings_nav.clicked.connect(self._open_settings); sl.addWidget(self.settings_nav)
        self.models_nav = self._nav("    Models"); self.models_nav.setVisible(False); self.models_nav.clicked.connect(self._open_models); sl.addWidget(self.models_nav)
        outer.addWidget(side)
        self.pages = QStackedWidget(); outer.addWidget(self.pages, 1)
        self._build_chat_page(); self._build_models_page()

    def _nav(self, text: str) -> QPushButton:
        button = QPushButton(text); button.setProperty("class", "nav"); button.setProperty("active", False); return button

    def _build_chat_page(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(38, 30, 38, 26); layout.setSpacing(15)
        head = QHBoxLayout(); title = QLabel("New Chat"); title.setProperty("class", "title"); head.addWidget(title); head.addStretch()
        self.chat_model = QLabel(); self.chat_model.setProperty("class", "muted"); head.addWidget(self.chat_model); layout.addLayout(head)
        self.context = QTextEdit(); self.context.setReadOnly(True); self.context.setPlaceholderText("Conversation content will appear here"); layout.addWidget(self.context, 1)
        prompt_row = QHBoxLayout(); self.input_editor = QTextEdit(); self.input_editor.setPlaceholderText("Type a message to start a new conversation…"); self.input_editor.setFixedHeight(88)
        prompt_row.addWidget(self.input_editor, 1)
        actions = QVBoxLayout(); self.send_button = QPushButton("Send"); self.send_button.setProperty("class", "primary"); self.send_button.clicked.connect(self.start_api_call); actions.addWidget(self.send_button)
        self.stop_button = QPushButton("Stop"); self.stop_button.setProperty("class", "secondary"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop_api_call); actions.addWidget(self.stop_button); actions.addStretch(); prompt_row.addLayout(actions); layout.addLayout(prompt_row)
        self.pages.addWidget(page)

    def _build_models_page(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(38, 30, 38, 26)
        header = QHBoxLayout(); title = QLabel("Models"); title.setProperty("class", "title"); header.addWidget(title); header.addStretch(); back = QPushButton("Back to Chat"); back.setProperty("class", "secondary"); back.clicked.connect(self._return_to_chat); header.addWidget(back); layout.addLayout(header)
        sub = QLabel("Manage URL, API keys, and model parameters. Changes are saved to models.json automatically"); sub.setProperty("class", "muted"); layout.addWidget(sub)
        body = QHBoxLayout();
        left_models = QVBoxLayout()
        add = QPushButton("+ Add"); add.setProperty("class", "secondary"); add.clicked.connect(self._add_model); left_models.addWidget(add)
        self.model_list = QListWidget(); self.model_list.setFixedWidth(245); self.model_list.currentRowChanged.connect(self._model_selected); left_models.addWidget(self.model_list, 1)
        body.addLayout(left_models)
        form_box = QFrame(); form = QFormLayout(form_box); form.setContentsMargins(24, 8, 24, 8); form.setVerticalSpacing(14)
        self.name_field = QLineEdit()
        self.url_field = self._option_field(self._current_model().url_options)
        self.url_field.setPlaceholderText("https://example.com/v1/chat/completions")
        self.key_field = self._option_field(self._current_model().api_key_options)
        self.key_field.lineEdit().setEchoMode(QLineEdit.EchoMode.Normal)
        self.model_field = self._option_field(self._current_model().model_options)
        self.temp_field = self._numeric_field([i / 10 for i in range(11)])
        self.tokens_field = self._numeric_field([10_000, 20_000, 30_000, 50_000, 80_000, 100_000, 150_000, 200_000, 500_000, 1_000_000])
        self.timeout_field = self._numeric_field([(10, "10 sec"), (20, "20 sec"), (30, "30 sec"), (60, "60 sec"), (120, "120 sec"), (300, "5 min"), (600, "10 min")])
        for field in (self.name_field, self.url_field, self.key_field, self.model_field, self.temp_field, self.tokens_field, self.timeout_field):
            field.setFixedWidth(460)
        for label, widget in (("Configuration Name", self.name_field), ("URL", self.url_field), ("API Key", self.key_field), ("Model Name", self.model_field), ("Temperature", self.temp_field), ("Max Tokens", self.tokens_field), ("Timeout", self.timeout_field)): form.addRow(label, widget)
        buttons = QHBoxLayout(); save = QPushButton("Save"); save.setObjectName("saveButton"); save.clicked.connect(self._save_form); delete = QPushButton("Delete Configuration"); delete.setProperty("class", "secondary"); delete.clicked.connect(self._delete_model); buttons.addWidget(save); buttons.addWidget(delete); buttons.addStretch(); form.addRow(buttons); body.addWidget(form_box, 1); layout.addLayout(body, 1)
        self.pages.addWidget(page); self._reload_model_list()

    @staticmethod
    def _option_field(options: list[str]) -> QComboBox:
        field = QComboBox(); field.setEditable(True); field.addItems(options)
        field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        return field

    @staticmethod
    def _numeric_field(values: list[object]) -> QComboBox:
        field = QComboBox(); field.setEditable(True)
        field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for value in values:
            if isinstance(value, tuple):
                numeric, label = value
            else:
                numeric, label = value, str(value)
            field.addItem(label, numeric)
        return field

    @staticmethod
    def _set_numeric_value(field: QComboBox, value: float | int | None) -> None:
        field.blockSignals(True)
        if value is None:
            field.setCurrentIndex(-1)
            field.lineEdit().clear()
            field.blockSignals(False)
            return
        for index in range(field.count()):
            if float(field.itemData(index)) == float(value):
                field.setCurrentIndex(index)
                field.blockSignals(False)
                return
        field.setCurrentIndex(-1)
        field.lineEdit().setText(str(value))
        field.blockSignals(False)

    @staticmethod
    def _numeric_value(field: QComboBox, field_name: str) -> float:
        text = field.currentText().strip().lower().replace(",", "")
        if text.endswith("min"):
            return float(text[:-3].strip()) * 60
        return float(text.replace("seconds", "").replace("sec", "").strip())


    @staticmethod
    def _set_options(field: QComboBox, options: list[str], current: str, keep_empty: bool = False) -> None:
        values: list[str] = []
        for value in [*options, current]:
            value = str(value).strip()
            if (value or keep_empty) and value not in values:
                values.append(value)
        field.blockSignals(True)
        field.clear()
        field.addItems(values)
        field.setCurrentIndex(-1)
        field.lineEdit().clear()
        field.lineEdit().setText(current)
        field.blockSignals(False)

    def _show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index); self.chat_nav.setProperty("active", index == 0); self.models_nav.setProperty("active", index == 1)
        self.chat_nav.style().unpolish(self.chat_nav); self.chat_nav.style().polish(self.chat_nav); self.models_nav.style().unpolish(self.models_nav); self.models_nav.style().polish(self.models_nav)

    def _return_to_chat(self) -> None:
        """Return to the initial chat navigation state."""
        self._settings_open = False
        self.models_nav.setVisible(False)
        self._show_page(0)
        self.settings_nav.setProperty("active", False)
        self.models_nav.setProperty("active", False)
        for button in (self.settings_nav, self.models_nav):
            button.style().unpolish(button)
            button.style().polish(button)

    def _open_settings(self) -> None:
        self._settings_open = not self._settings_open
        self.models_nav.setVisible(self._settings_open)
        # Opening Settings only expands the navigation. The conversation stays
        # visible until the user explicitly chooses one of its sub-pages.
        if self._settings_open:
            self._show_page(0)
            self.settings_nav.setProperty("active", True)
        else:
            self._show_page(0)
            self.settings_nav.setProperty("active", False)
        self.settings_nav.style().unpolish(self.settings_nav)
        self.settings_nav.style().polish(self.settings_nav)

    def _open_models(self) -> None:
        self._show_page(1)
        self.settings_nav.setProperty("active", True)
        self.models_nav.setProperty("active", True)
        for button in (self.settings_nav, self.models_nav):
            button.style().unpolish(button)
            button.style().polish(button)


    def _current_model(self) -> DesktopModelConfig:
        return next(item for item in self.models if item.name == self.active_name)

    def _reload_model_list(self) -> None:
        self.model_list.blockSignals(True); self.model_list.clear()
        for model in self.models: self.model_list.addItem(QListWidgetItem(model.name))
        row = next((i for i, m in enumerate(self.models) if m.name == self.active_name), 0); self.model_list.setCurrentRow(row); self.model_list.blockSignals(False); self.chat_model.setText(f"模型：{self.active_name}")

    def _model_selected(self, row: int) -> None:
        if not 0 <= row < len(self.models):
            return
        selected = self.models[row]
        if self._pending_model is not None and selected is not self._pending_model:
            self.models.remove(self._pending_model)
            self._pending_model = None
            self.active_name = selected.name
            self._reload_model_list()
            self._load_form(selected)
            self._save_models()
            return
        self.active_name = selected.name
        self._load_form(selected)
        if selected is not self._pending_model:
            self._save_models()

    def _load_form(self, model: DesktopModelConfig) -> None:
        if not hasattr(self, "name_field"): return
        self.name_field.setText(model.name); self._set_options(self.url_field, model.url_options, model.url); self._set_options(self.key_field, model.api_key_options, model.api_key, keep_empty=True); self._set_options(self.model_field, model.model_options, model.model); self._set_numeric_value(self.temp_field, model.temperature); self._set_numeric_value(self.tokens_field, model.max_tokens); self._set_numeric_value(self.timeout_field, model.timeout); self.chat_model.setText(f"模型：{model.name}")

    def _save_models(self) -> bool:
        try: self.store.save(self.active_name, self.models); return True
        except (OSError, ValueError) as exc: QMessageBox.critical(self, "Save Failed", str(exc)); return False

    def _save_form(self) -> None:
        name = self.name_field.text().strip(); old = self._current_model()
        url = self.url_field.currentText().strip(); api_key = self.key_field.currentText().strip(); model_name = self.model_field.currentText().strip()
        if not name or not url or not model_name: QMessageBox.warning(self, "Incomplete Configuration", "Configuration name, URL, and model name are required."); return
        if any(m is not old and m.name == name for m in self.models): QMessageBox.warning(self, "Save Failed", "Configuration name already exists."); return
        try:
            temperature = self._numeric_value(self.temp_field, "temperature")
            max_tokens_value = self._numeric_value(self.tokens_field, "max_tokens")
            if not max_tokens_value.is_integer():
                raise ValueError("max tokens must be an integer")
            max_tokens = int(max_tokens_value)
            timeout = self._numeric_value(self.timeout_field, "timeout")
        except (TypeError, ValueError):
            QMessageBox.warning(self, "Invalid Configuration", "Temperature, max tokens, and timeout must be valid numbers."); return
        if not TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX:
            QMessageBox.warning(self, "Invalid Configuration", f"Temperature must be between {TEMPERATURE_MIN:g} and {TEMPERATURE_MAX:g}."); return
        if not MAX_TOKENS_MIN <= max_tokens <= MAX_TOKENS_MAX:
            QMessageBox.warning(self, "Invalid Configuration", f"Max tokens must be between {MAX_TOKENS_MIN:,} and {MAX_TOKENS_MAX:,}."); return
        if not TIMEOUT_MIN <= timeout <= TIMEOUT_MAX:
            QMessageBox.warning(self, "Invalid Configuration", f"Timeout must be between {TIMEOUT_MIN:g} and {TIMEOUT_MAX:g} seconds."); return
        updated = DesktopModelConfig(name, url, api_key, model_name, temperature, max_tokens, timeout, old.url_options, old.api_key_options, old.model_options)
        updated.url_options = self._add_option(updated.url_options, url)
        updated.api_key_options = self._add_option(updated.api_key_options, api_key, keep_empty=True)
        updated.model_options = self._add_option(updated.model_options, model_name)
        self.models[self.models.index(old)] = updated; self.active_name = name
        if old is self._pending_model:
            self._pending_model = None
        if self._save_models(): self._reload_model_list()

    @staticmethod
    def _add_option(options: list[str], value: str, keep_empty: bool = False) -> list[str]:
        result = list(options)
        if (value or keep_empty) and value not in result: result.append(value)
        return result

    def _add_model(self) -> None:
        base = DesktopModelConfig(
            name="New Configuration", url="", api_key="", model="",
            temperature=None, max_tokens=None, timeout=None,
        ); names = {m.name for m in self.models}; i = 2
        while base.name in names: base.name = f"New Configuration {i}"; i += 1
        self.models.append(base); self.active_name = base.name; self._pending_model = base; self._reload_model_list(); self._load_form(base)

    def _delete_model(self) -> None:
        if len(self.models) <= 1: QMessageBox.information(self, "Cannot Delete", "At least one model configuration must remain."); return
        current = self._current_model()
        if QMessageBox.question(self, "Delete Configuration", f"Delete “{current.name}”?") != QMessageBox.StandardButton.Yes: return
        self.models.remove(current)
        if current is self._pending_model:
            self._pending_model = None
            self.active_name = self.models[0].name
            self._reload_model_list()
            self._load_form(self.models[0])
            return
        self.active_name = self.models[0].name; self._save_models(); self._reload_model_list()

    def start_api_call(self) -> None:
        prompt = self.input_editor.toPlainText().strip(); config = self._current_model()
        if not prompt: QMessageBox.warning(self, "Cannot Send", "Please enter a message."); return
        if not config.api_key: QMessageBox.warning(self, "Cannot Send", "Enter an API Key in Settings → Models first."); return
        self.context.append(f"You\n{prompt}\n"); self.input_editor.clear(); self.worker = LLMWorker(config, prompt); self.worker.content_delta.connect(self._append_content); self.worker.reasoning_delta.connect(lambda text: None); self.worker.completed.connect(self._call_completed); self.send_button.setEnabled(False); self.stop_button.setEnabled(True); self.worker.start()

    def stop_api_call(self) -> None:
        if self.worker and self.worker.isRunning(): self.worker.cancel(); self.stop_button.setEnabled(False)

    def _append_content(self, text: str) -> None:
        self.context.moveCursor(QTextCursor.MoveOperation.End); self.context.insertPlainText(text); self.context.ensureCursorVisible()

    def _call_completed(self, success: bool, message: str) -> None:
        self.context.append("\n"); self.worker = None; self.send_button.setEnabled(True); self.stop_button.setEnabled(False)
        if not success and message != "User cancelled the request": QMessageBox.critical(self, "Call Failed", message)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    icon_dir = Path(__file__).resolve().parents[2] / "image" / "logo"
    app_icon = QIcon(str(icon_dir / "logo-128.png"))
    app.setWindowIcon(app_icon)
    window = ChatWindow(); window.show(); return app.exec()


if __name__ == "__main__": raise SystemExit(main())
