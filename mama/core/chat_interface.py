from __future__ import annotations

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    AsyncIterator, Optional, Any, Union, Literal, Sequence, Callable,
    Iterator
)

logger = logging.getLogger(__name__)


# ---------- 数据结构 ----------

@dataclass
class TextPart:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImagePart:
    """OpenAI vision 格式: data URL 或 URL"""
    image_url: dict   # {"url": "...", "detail": "auto|low|high"}
    type: Literal["image_url"] = "image_url"


@dataclass
class AudioPart:
    """OpenAI audio input 格式 (gpt-4o-audio)"""
    input_audio: dict  # {"data": "<base64>", "format": "wav|mp3"}
    type: Literal["input_audio"] = "input_audio"


ContentPart = Union[TextPart, ImagePart, AudioPart]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_litellm(self) -> dict:
        # 序列化 content
        if isinstance(self.content, str):
            content = self.content
        else:
            content = []
            for p in self.content:
                if isinstance(p, TextPart):
                    content.append({"type": "text", "text": p.text})
                elif isinstance(p, ImagePart):
                    content.append({"type": "image_url", "image_url": p.image_url})
                elif isinstance(p, AudioPart):
                    content.append({"type": "input_audio", "input_audio": p.input_audio})

        msg: dict = {"role": self.role, "content": content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class ChatSession:
    messages: list[Message]
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[list[dict]] = None          # OpenAI tools schema
    tool_choice: Optional[Union[str, dict]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerateResult:
    """非流式返回的完整结果"""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""
    raw: Any = None


# ---------- 抽象基类 ----------

class AIModel(ABC):
    @abstractmethod
    def generate_response(self, session: "ChatSession") -> Iterator[str]:
        """流式返回文本分片"""
        ...

    @abstractmethod
    def generate_response_complete(self, session: "ChatSession") -> str:
        """一次性返回完整文本"""
        ...

    def generate(self, session: "ChatSession") -> "GenerateResult":
        """可选：返回结构化结果（含 tool_calls / usage）"""
        raise NotImplementedError


class ChatInterface(ABC):
    """聊天界面抽象基类"""

    @abstractmethod
    def display_message(self, message: Message) -> None:
        """显示一条消息"""
        pass

    @abstractmethod
    def get_user_input(self) -> Optional[str]:
        """获取用户输入，返回None表示退出"""
        pass

    @abstractmethod
    def display_error(self, error: str) -> None:
        """显示错误信息"""
        pass

    @abstractmethod
    def display_system_message(self, message: str) -> None:
        """显示系统消息"""
        pass

    @abstractmethod
    def clear_screen(self) -> None:
        """清屏（可选实现）"""
        pass

