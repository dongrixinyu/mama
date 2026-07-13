from typing import AsyncIterator
from mama.core.chat_interface import AIModel, ChatSession


class MockAIModel(AIModel):
    """模拟AI后端，用于测试"""

    def generate_response(self, session: ChatSession) -> str:
        """流式生成回复"""
        response = f"你说了: {session.messages[-1].content}"
        # for char in response:
            # yield char
        return response

    def generate_response_complete(self, session: ChatSession) -> str:
        """生成完整回复"""
        last_user_msg = session.messages[-1].content
        return f"收到你的消息：「{last_user_msg}」。这是一个模拟回复。"

