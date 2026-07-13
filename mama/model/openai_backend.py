from typing import AsyncIterator, Optional
from mama.core.chat_interface import AIModel, ChatSession
import os


class OpenAIBackend(AIModel):
    """OpenAI API后端"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("需要提供OpenAI API密钥")

    async def generate_response(self, session: ChatSession) -> AsyncIterator[str]:
        """流式生成回复"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in session.messages
            ]

            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            raise RuntimeError("需要安装 openai 库: pip install openai")

    async def generate_response_complete(self, session: ChatSession) -> str:
        """生成完整回复"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in session.messages
            ]

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            return response.choices[0].message.content

        except ImportError:
            raise RuntimeError("需要安装 openai 库: pip install openai")

