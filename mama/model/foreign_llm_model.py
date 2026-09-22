"""OpenAI-compatible LLM model used by the desktop interface.

The adapter deliberately lives in the model package: the UI only receives
text/progress callbacks and does not know how an SSE response is parsed.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests


@dataclass
class LLMCallResult:
    success: bool
    reasoning: str = ""
    content: str = ""
    error: str = ""


class LLMCallCancelled(Exception):
    """Raised internally when the caller requests cancellation."""


class OpenAICompatibleModel:
    """Call any endpoint implementing ``/chat/completions`` and SSE."""

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 50000,
        timeout: float = 800.0,
    ) -> None:
        self.url = url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def call_llm(
        self,
        input_text: str,
        stop_event: Optional[threading.Event] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        on_content: Optional[Callable[[str], None]] = None,
    ) -> LLMCallResult:
        """Execute one streaming request and report deltas as they arrive."""
        if not self.api_key:
            return LLMCallResult(False, error="API Key 不能为空")
        if not self.model:
            return LLMCallResult(False, error="模型名称不能为空")
        if not input_text.strip():
            return LLMCallResult(False, error="输入内容不能为空")

        stop_event = stop_event or threading.Event()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": input_text}],
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        reasoning: list[str] = []
        content: list[str] = []

        try:
            with requests.post(
                self.url, headers=headers, json=payload, stream=True,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    detail = response.text[:2000]
                    return LLMCallResult(
                        False, error=f"API 请求失败 (HTTP {response.status_code}): {detail}"
                    )

                for raw_line in response.iter_lines(decode_unicode=True):
                    if stop_event.is_set():
                        raise LLMCallCancelled
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(thought, str) and thought:
                        reasoning.append(thought)
                        if on_reasoning:
                            on_reasoning(thought)
                    if isinstance(text, str) and text:
                        content.append(text)
                        if on_content:
                            on_content(text)

            result = LLMCallResult(True, "".join(reasoning), "".join(content))
            self._save_response(input_text, result)
            return result
        except LLMCallCancelled:
            return LLMCallResult(False, "".join(reasoning), "".join(content), "用户已终止请求")
        except requests.RequestException as exc:
            return LLMCallResult(False, "".join(reasoning), "".join(content), f"网络请求异常: {exc}")
        except Exception as exc:
            return LLMCallResult(False, "".join(reasoning), "".join(content), f"请求失败: {exc}")

    def _save_response(self, input_text: str, result: LLMCallResult) -> None:
        response_dir = Path(os.path.expanduser("~/.mama/responses"))
        try:
            response_dir.mkdir(parents=True, exist_ok=True)
            path = response_dir / f"{self.model.replace('/', '_')}-{uuid.uuid4()}.json"
            path.write_text(json.dumps({
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": self.model,
                "input": input_text,
                "reasoning": result.reasoning,
                "output": result.content,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            # Saving a transcript must never turn a successful API call into a failure.
            pass


def call_llm(url, api_key, model_name, temperature, max_tokens, input_text, stop_event,
             on_reasoning=None, on_content=None) -> LLMCallResult:
    """Compatibility function for callers migrating from ``foreign_llm_gui``."""
    return OpenAICompatibleModel(
        url, api_key, model_name, float(temperature), int(max_tokens)
    ).call_llm(input_text, stop_event, on_reasoning, on_content)
