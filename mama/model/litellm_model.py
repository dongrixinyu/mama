from __future__ import annotations

import os
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any, Union, Iterator

import litellm
from litellm import acompletion
from litellm.exceptions import (
    Timeout, RateLimitError, APIConnectionError,
    APIError, InternalServerError, ServiceUnavailableError,
)

from mama.core.chat_interface import (
    AIModel, ChatSession, Message, ToolCall, GenerateResult, TextPart
)

logger = logging.getLogger(__name__)

# 关掉 litellm 自身的冗余日志
litellm.set_verbose = False
# litellm 内置兜底（也可在 kwargs 中传 fallbacks）
litellm.suppress_debug_info = True


@dataclass
class LiteLLMConfig:
    """统一模型配置。model 字符串遵循 litellm 约定，例如：
       - "gpt-4o", "gpt-4o-mini"
       - "claude-3-5-sonnet-20240620"
       - "gemini/gemini-1.5-pro"
       - "deepseek/deepseek-chat"
       - "ollama/llama3"
       - "azure/gpt-4o"
    """
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    custom_llm_provider: Optional[str] = None

    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[list[str]] = None
    seed: Optional[int] = None

    timeout: float = 60.0
    max_retries: int = 3      # litellm 内置重试次数
    retry_delay: float = 1.0

    # fallback 模型列表（任一主模型失败时按序尝试）
    fallbacks: Optional[list[str]] = None

    # 结构化输出 / JSON 模式
    response_format: Optional[dict] = None

    # 透传给 litellm 的额外参数
    extra: dict = field(default_factory=dict)


class LiteLLMModel(AIModel):
    """基于 litellm 的统一 AI 后端"""

    def __init__(self, config: LiteLLMConfig):
        self.config = config
        # 全局兜底配置
        litellm.request_timeout = config.timeout
        litellm.num_retries = config.max_retries
        litellm.retry_after = config.retry_delay
        litellm.allowed_fails = 3
        # 环境变量兼容（让用户可在 .env 设置 OPENAI_API_KEY 等）
        if config.api_key and not os.getenv(_provider_env(config.model)):
            os.environ[_provider_env(config.model)] = config.api_key

    # ---------- 内部 ----------

    def _build_kwargs(self, session: ChatSession, *, stream: bool = False) -> dict:
        messages: list[dict] = []
        if session.system_prompt:
            messages.append({"role": "system", "content": session.system_prompt})
        messages.extend(m.to_litellm() for m in session.messages)

        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "timeout": self.config.timeout,
            "stream": stream,
            **self.config.extra,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.custom_llm_provider:
            kwargs["custom_llm_provider"] = self.config.custom_llm_provider
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.stop:
            kwargs["stop"] = self.config.stop
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if self.config.response_format:
            kwargs["response_format"] = self.config.response_format
        if session.tools:
            kwargs["tools"] = session.tools
        if session.tool_choice is not None:
            kwargs["tool_choice"] = session.tool_choice
        if self.config.fallbacks:
            kwargs["fallbacks"] = [
                {"model": m} for m in self.config.fallbacks
            ]
        return kwargs

    @staticmethod
    def _handle_error(e: Exception) -> str:
        if isinstance(e, Timeout):
            msg = f"请求超时: {e}"
        elif isinstance(e, RateLimitError):
            msg = f"触发限流: {e}"
        elif isinstance(e, APIConnectionError):
            msg = f"网络连接错误: {e}"
        elif isinstance(e, (InternalServerError, ServiceUnavailableError)):
            msg = f"服务端错误(5xx): {e}"
        elif isinstance(e, APIError):
            msg = f"API 错误: {e}"
        else:
            msg = f"未知错误: {e}"
        logger.exception("LLM 调用失败: %s", msg)
        return msg

    # ---------- 接口实现 ----------

    def generate_response(self, session: ChatSession) -> Iterator[str]:
        """
        流式返回文本片段
        """
        params = self._build_kwargs(session, stream=True)
        response = litellm.completion(**params)

        for chunk in response:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except Exception:
                continue

    def generate_response_complete(self, session: ChatSession) -> str:
        """
        非流式一次性返回完整文本
        """
        params = self._build_kwargs(session, stream=False)
        response = litellm.completion(**params)

        try:
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def generate(self, session: ChatSession) -> GenerateResult:
        """
        返回结构化结果
        """
        params = self._build_kwargs(session, stream=False)
        response = litellm.completion(**params)

        content = ""
        tool_calls = []
        usage = None
        raw_response = None

        try:
            message = response.choices[0].message
            content = getattr(message, "content", "") or ""
            tool_calls = getattr(message, "tool_calls", []) or []
        except Exception:
            pass

        try:
            usage_obj = getattr(response, "usage", None)
            if usage_obj is not None:
                if hasattr(usage_obj, "dict"):
                    usage = usage_obj.dict()
                elif hasattr(usage_obj, "model_dump"):
                    usage = usage_obj.model_dump()
                else:
                    usage = dict(usage_obj)
        except Exception:
            usage = None

        try:
            if hasattr(response, "model_dump"):
                raw_response = response.model_dump()
            elif hasattr(response, "dict"):
                raw_response = response.dict()
        except Exception:
            raw_response = None

        return GenerateResult(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw=raw_response,
        )


def _provider_env(model: str) -> str:
    """根据 litellm model 前缀推断对应 provider 的环境变量名"""
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        return {
            "openai": "OPENAI_API_KEY",
            "azure": "AZURE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "vertex_ai": "VERTEXAI_PROJECT",
            "huggingface": "HUGGINGFACE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "ollama": "OLLAMA_API_BASE",
            "together_ai": "TOGETHERAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "anyscale": "ANYSCALE_API_KEY",
        }.get(prefix, "OPENAI_API_KEY")

    return "OPENAI_API_KEY"

