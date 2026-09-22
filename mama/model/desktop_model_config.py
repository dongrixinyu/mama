"""Persistent desktop-model configuration stored in ``~/.mama/models.json``."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class DesktopModelConfig:
    name: str = "默认模型"
    url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 50000
    timeout: float = 800.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesktopModelConfig":
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in data.items() if key in fields}
        return cls(**values)


class DesktopModelStore:
    """Load and atomically save named OpenAI-compatible endpoint settings."""

    DEFAULT_PATH = Path.home() / ".mama" / "models.json"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path else self.DEFAULT_PATH

    def load(self) -> tuple[str, list[DesktopModelConfig]]:
        if not self.path.exists():
            default = DesktopModelConfig()
            self.save(default.name, [default])
            return default.name, [default]
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取模型配置 {self.path}: {exc}") from exc

        # The documented format is an object. Also accept a bare list so early
        # hand-written configuration files do not break the application.
        if isinstance(data, list):
            active_name, raw_models = "", data
        elif isinstance(data, dict):
            active_name = str(data.get("active_model", ""))
            raw_models = data.get("models", [])
        else:
            raise RuntimeError("models.json 必须是对象或模型配置数组")
        if not isinstance(raw_models, list):
            raise RuntimeError("models.json 的 models 字段必须是数组")

        models = [DesktopModelConfig.from_dict(item) for item in raw_models if isinstance(item, dict)]
        if not models:
            default = DesktopModelConfig()
            self.save(default.name, [default])
            return default.name, [default]
        names = {item.name for item in models}
        return (active_name if active_name in names else models[0].name), models

    def save(self, active_name: str, models: list[DesktopModelConfig]) -> None:
        if not models:
            raise ValueError("至少要保留一个模型配置")
        names = [model.name.strip() for model in models]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("配置名称不能为空且不能重复")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = {"version": 1, "active_model": active_name, "models": [asdict(model) for model in models]}
        fd, temporary_path = tempfile.mkstemp(prefix="models-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
