import sys
import asyncio
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel
)
from PySide6.QtCore import Qt

from mama.core.chat_service import ChatService
from mama.model.mock_ai_backend import MockAIBackend


class ChatWindow(QWidget):
    def __init__(self, chat_service: ChatService):
        super().__init__()
        self.chat_service = chat_service
        self.setWindowTitle("AI Chat GUI Demo")
        self.resize(500, 400)

        self.layout = QVBoxLayout()

        self.title_label = QLabel("简易 AI 聊天窗口")
        self.layout.addWidget(self.title_label)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.layout.addWidget(self.chat_area)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("请输入你的消息...")
        self.input_box.returnPressed.connect(self.on_send)
        self.layout.addWidget(self.input_box)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.on_send)
        self.layout.addWidget(self.send_button)

        self.setLayout(self.layout)

    def append_message(self, role: str, content: str):
        if role == "user":
            self.chat_area.append(f"<b>你:</b> {content}")
        else:
            self.chat_area.append(f"<b>AI:</b> {content}")

    def on_send(self):
        user_text = self.input_box.text().strip()
        if not user_text:
            return

        self.append_message("user", user_text)
        self.input_box.clear()

        try:
            reply = asyncio.run(self.chat_service.send_user_message(user_text))
            self.append_message("assistant", reply)
        except Exception as e:
            self.chat_area.append(f"<span style='color:red;'>错误: {e}</span>")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    chat_service = ChatService(MockAIBackend())
    window = ChatWindow(chat_service)
    window.show()

    sys.exit(app.exec())

