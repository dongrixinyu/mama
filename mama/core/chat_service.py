from mama.core.chat_interface import ChatSession, Message
from mama.model.mock_ai_backend import AIModel


class ChatService:
    def __init__(self, ai_backend: AIModel):
        self.ai_backend = ai_backend
        self.session = ChatSession()

    def send_user_message(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("用户输入不能为空")

        self.session.messages.append(Message(role="user", content=user_input))
        reply = self.ai_backend.generate_response_complete(self.session)
        self.session.messages.append(Message(role="assistant", content=reply))

        return reply

    def get_history(self) -> list[Message]:
        return self.session.messages

    def clear_history(self) -> None:
        self.session.messages.clear()