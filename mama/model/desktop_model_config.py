"""Persistent desktop-model configuration stored in ``~/.mama/models.json``."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 1.0
MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = 1_000_000
TIMEOUT_MIN = 1.0
TIMEOUT_MAX = 3_600.0


@dataclass
class DesktopModelConfig:
    name: str = "Default Model"
    url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    model: str = ""
    temperature: float | None = 0.2
    max_tokens: int | None = 50000
    timeout: float | None = 800.0
    # Each named configuration owns its own selectable values. These lists are
    # deliberately stored on the model entry rather than globally.
    url_options: list[str] = field(default_factory=list)
    api_key_options: list[str] = field(default_factory=list)
    model_options: list[str] = field(default_factory=list)

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
            raise RuntimeError(f"Unable to read model configuration {self.path}: {exc}") from exc

        # The documented format is an object. Also accept a bare list so early
        # hand-written configuration files do not break the application.
        if isinstance(data, list):
            active_name, raw_models = "", data
            saved_options: dict[str, Any] = {}
        elif isinstance(data, dict):
            active_name = str(data.get("active_model", ""))
            raw_models = data.get("models", [])
            saved_options = data.get("options", {}) if isinstance(data.get("options", {}), dict) else {}
        else:
            raise RuntimeError("models.json must be an object or model configuration array")
        if not isinstance(raw_models, list):
            raise RuntimeError("models.json models field must be an array")

        models = [DesktopModelConfig.from_dict(item) for item in raw_models if isinstance(item, dict)]
        # Older files had one global options block. Do not share it between
        # configurations; each entry starts with its own current value.
        for item in models:
            item.url_options = self._merge_options(item.url_options, [item.url])
            item.api_key_options = self._merge_options(item.api_key_options, [item.api_key], keep_empty=True)
            item.model_options = self._merge_options(item.model_options, [item.model])
        if not models:
            default = DesktopModelConfig()
            self.save(default.name, [default])
            return default.name, [default]
        names = {item.name for item in models}
        return (active_name if active_name in names else models[0].name), models

    @staticmethod
    def _merge_options(saved: Any, values: Any, keep_empty: bool = False) -> list[str]:
        result: list[str] = []
        candidates = list(saved) if isinstance(saved, list) else []
        candidates.extend(values)
        for value in candidates:
            value = str(value).strip()
            if (value or keep_empty) and value not in result:
                result.append(value)
        return result

    def save(self, active_name: str, models: list[DesktopModelConfig]) -> None:
        if not models:
            raise ValueError("At least one model configuration must be kept")
        for model in models:
            if model.temperature is not None and not TEMPERATURE_MIN <= model.temperature <= TEMPERATURE_MAX:
                raise ValueError(f"Temperature must be between {TEMPERATURE_MIN:g} and {TEMPERATURE_MAX:g}")
            if model.max_tokens is not None and not MAX_TOKENS_MIN <= model.max_tokens <= MAX_TOKENS_MAX:
                raise ValueError(f"Max tokens must be between {MAX_TOKENS_MIN:,} and {MAX_TOKENS_MAX:,}")
            if model.timeout is not None and not TIMEOUT_MIN <= model.timeout <= TIMEOUT_MAX:
                raise ValueError(f"Timeout must be between {TIMEOUT_MIN:g} and {TIMEOUT_MAX:g} seconds")
        names = [model.name.strip() for model in models]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Configuration names cannot be empty or duplicated")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        for model in models:
            model.url_options = self._merge_options(model.url_options, [model.url])
            model.api_key_options = self._merge_options(model.api_key_options, [model.api_key], keep_empty=True)
            model.model_options = self._merge_options(model.model_options, [model.model])
        payload = {
            "version": 2, "active_model": active_name,
            "models": [asdict(model) for model in models],
        }
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
