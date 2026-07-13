from dataclasses import dataclass, asdict
from typing import Any

from mama.model.litellm_model import LiteLLMConfig


@dataclass
class ConfigIssue:
    field_name: str
    message: str
    current_value: Any
    required: bool = True


import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None


class ConfigManager:
    DEFAULT_CONFIG_FILE = ".mama.config.yml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 可选的配置文件路径，如果未提供则使用默认路径
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.cwd() / self.DEFAULT_CONFIG_FILE

    def load(self, cli_args: Optional[argparse.Namespace] = None) -> LiteLLMConfig:
        """
        按优先级加载配置：配置文件 < 环境变量 < 命令行参数
        
        Args:
            cli_args: 命令行参数对象
            
        Returns:
            合并后的配置对象
        """
        # 1. 首先加载配置文件（优先级最低）
        config = self._load_from_file()
        
        # 2. 然后加载环境变量（优先级中等）
        config = self._load_from_env(config)
        
        # 3. 最后加载命令行参数（优先级最高）
        if cli_args:
            config = self._load_from_cli(config, cli_args)
        
        return config

    def _load_from_file(self) -> LiteLLMConfig:
        """从配置文件加载，不存在则返回默认配置"""
        if not self.config_path.exists():
            return LiteLLMConfig()

        suffix = self.config_path.suffix.lower()

        with open(self.config_path, "r", encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                if yaml is None:
                    raise RuntimeError("未安装 PyYAML，无法读取 YAML 配置，请先 pip install pyyaml")
                data = yaml.safe_load(f) or {}
            elif suffix == ".json":
                data = json.load(f) or {}
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")

        return self._dict_to_config(data)

    def _load_from_env(self, config: LiteLLMConfig) -> LiteLLMConfig:
        """从环境变量加载配置，覆盖已有配置"""
        env_mapping = {
            "MAMA_MODEL": "model",
            "MAMA_API_KEY": "api_key",
            "LITELLM_API_KEY": "api_key",
            "OPENAI_API_KEY": "api_key",
            "MAMA_API_BASE": "api_base",
            "LITELLM_API_BASE": "api_base",
            "MAMA_CUSTOM_LLM_PROVIDER": "custom_llm_provider",
            "MAMA_TEMPERATURE": "temperature",
            "MAMA_TOP_P": "top_p",
            "MAMA_MAX_TOKENS": "max_tokens",
            "MAMA_PRESENCE_PENALTY": "presence_penalty",
            "MAMA_FREQUENCY_PENALTY": "frequency_penalty",
            "MAMA_TIMEOUT": "timeout",
            "MAMA_MAX_RETRIES": "max_retries",
            "MAMA_RETRY_DELAY": "retry_delay",
            "MAMA_SEED": "seed",
        }
        
        for env_key, field_name in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    parsed_value = self._parse_env_value(field_name, env_value)
                    setattr(config, field_name, parsed_value)
                except (ValueError, TypeError):
                    # 环境变量格式错误时跳过
                    pass
        
        return config

    def _load_from_cli(self, config: LiteLLMConfig, cli_args: argparse.Namespace) -> LiteLLMConfig:
        """从命令行参数加载配置，覆盖已有配置"""
        cli_mapping = {
            "model": "model",
            "api_key": "api_key",
            "api_base": "api_base",
            "custom_llm_provider": "custom_llm_provider",
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "max_tokens",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "timeout": "timeout",
            "max_retries": "max_retries",
            "retry_delay": "retry_delay",
            "seed": "seed",
        }
        
        for cli_arg, field_name in cli_mapping.items():
            if hasattr(cli_args, cli_arg):
                cli_value = getattr(cli_args, cli_arg)
                if cli_value is not None:
                    setattr(config, field_name, cli_value)
        
        return config

    def _parse_env_value(self, field_name: str, value: str) -> Any:
        """将环境变量字符串解析为合适的类型"""
        if field_name in ("temperature", "top_p", "presence_penalty", 
                         "frequency_penalty", "timeout", "retry_delay"):
            return float(value)
        elif field_name in ("max_tokens", "seed", "max_retries"):
            return int(value)
        elif field_name in ("model", "api_key", "api_base", "custom_llm_provider"):
            return value
        else:
            return value

    def save(self, config: LiteLLMConfig) -> None:
        """保存配置到文件"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(config)
        suffix = self.config_path.suffix.lower()

        with open(self.config_path, "w", encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                if yaml is None:
                    raise RuntimeError("未安装 PyYAML，无法写入 YAML 配置，请先 pip install pyyaml")
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            elif suffix == ".json":
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")
            f.flush()
            os.fsync(f.fileno())

    def _dict_to_config(self, data: dict) -> LiteLLMConfig:
        """dict -> LiteLLMConfig，只接受已定义字段"""
        valid_fields = LiteLLMConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return LiteLLMConfig(**filtered)

    def validate(self, config: LiteLLMConfig) -> list[ConfigIssue]:
        """校验配置"""
        issues: list[ConfigIssue] = []

        # model 必填
        if not isinstance(config.model, str) or not config.model.strip():
            issues.append(ConfigIssue(
                field_name="model",
                message="model 不能为空，且必须是字符串",
                current_value=config.model,
                required=True
            ))

        # 数值范围校验
        if not isinstance(config.temperature, (int, float)) or not (0 <= config.temperature <= 2):
            issues.append(ConfigIssue(
                field_name="temperature",
                message="temperature 必须在 0~2 之间",
                current_value=config.temperature
            ))

        if not isinstance(config.top_p, (int, float)) or not (0 <= config.top_p <= 1):
            issues.append(ConfigIssue(
                field_name="top_p",
                message="top_p 必须在 0~1 之间",
                current_value=config.top_p
            ))

        if config.max_tokens is not None:
            if not isinstance(config.max_tokens, int) or config.max_tokens <= 0:
                issues.append(ConfigIssue(
                    field_name="max_tokens",
                    message="max_tokens 必须是正整数或 null",
                    current_value=config.max_tokens
                ))

        if not isinstance(config.presence_penalty, (int, float)) or not (-2 <= config.presence_penalty <= 2):
            issues.append(ConfigIssue(
                field_name="presence_penalty",
                message="presence_penalty 必须在 -2~2 之间",
                current_value=config.presence_penalty
            ))

        if not isinstance(config.frequency_penalty, (int, float)) or not (-2 <= config.frequency_penalty <= 2):
            issues.append(ConfigIssue(
                field_name="frequency_penalty",
                message="frequency_penalty 必须在 -2~2 之间",
                current_value=config.frequency_penalty
            ))

        if config.stop is not None:
            if not isinstance(config.stop, list) or not all(isinstance(x, str) for x in config.stop):
                issues.append(ConfigIssue(
                    field_name="stop",
                    message="stop 必须是字符串列表或 null",
                    current_value=config.stop
                ))

        if config.seed is not None and not isinstance(config.seed, int):
            issues.append(ConfigIssue(
                field_name="seed",
                message="seed 必须是整数或 null",
                current_value=config.seed
            ))

        if not isinstance(config.timeout, (int, float)) or config.timeout <= 0:
            issues.append(ConfigIssue(
                field_name="timeout",
                message="timeout 必须是正数",
                current_value=config.timeout
            ))

        if not isinstance(config.max_retries, int) or config.max_retries < 0:
            issues.append(ConfigIssue(
                field_name="max_retries",
                message="max_retries 必须是 >= 0 的整数",
                current_value=config.max_retries
            ))

        if not isinstance(config.retry_delay, (int, float)) or config.retry_delay < 0:
            issues.append(ConfigIssue(
                field_name="retry_delay",
                message="retry_delay 必须是 >= 0 的数值",
                current_value=config.retry_delay
            ))

        if config.fallbacks is not None:
            if not isinstance(config.fallbacks, list) or not all(isinstance(x, str) and x.strip() for x in config.fallbacks):
                issues.append(ConfigIssue(
                    field_name="fallbacks",
                    message="fallbacks 必须是非空字符串列表或 null",
                    current_value=config.fallbacks
                ))

        if config.response_format is not None and not isinstance(config.response_format, dict):
            issues.append(ConfigIssue(
                field_name="response_format",
                message="response_format 必须是 dict 或 null",
                current_value=config.response_format
            ))

        if not isinstance(config.extra, dict):
            issues.append(ConfigIssue(
                field_name="extra",
                message="extra 必须是 dict",
                current_value=config.extra
            ))

        # 可选：根据 model 判断 api_key 是否必需
        if self._model_probably_needs_api_key(config.model):
            if not config.api_key or not str(config.api_key).strip():
                issues.append(ConfigIssue(
                    field_name="api_key",
                    message="当前 model 看起来需要 api_key，但 api_key 为空",
                    current_value=config.api_key,
                    required=True
                ))

        return issues

    def get_missing_required_issues(self, config: LiteLLMConfig) -> list[ConfigIssue]:
        """仅返回开始对话前必须补齐的关键配置项问题"""
        return [issue for issue in self.validate(config) if issue.required]

    def _model_probably_needs_api_key(self, model: str) -> bool:
        """简单规则：本地 ollama 通常可不需要 key，其他默认需要"""
        if not model:
            return False
        model = model.lower()
        if model.startswith("ollama/"):
            return False
        return True



class ConfigInteractiveEditor:
    def __init__(self, interface=None):
        self.interface = interface

    def _print(self, text: str) -> None:
        if self.interface and hasattr(self.interface, "display_system_message"):
            self.interface.display_system_message(text)
        else:
            print(text)

    def _error(self, text: str) -> None:
        if self.interface and hasattr(self.interface, "display_error"):
            self.interface.display_error(text)
        else:
            print(f"[错误] {text}")

    def _input(self, prompt: str) -> str:
        if self.interface and hasattr(self.interface, "get_user_input"):
            # 如果你的 interface.get_user_input() 不支持 prompt，
            # 可以先 display_system_message(prompt) 再 get_user_input()
            self._print(prompt)
            result = self.interface.get_user_input()
            return "" if result is None else result
        return input(prompt)

    def repair(self, config: LiteLLMConfig, issues: list[ConfigIssue],
               validator: ConfigManager) -> LiteLLMConfig:
        """逐项修复配置，直到通过校验或用户放弃"""
        self._print("检测到配置文件存在问题，下面进入交互式修复。")
        self._print("输入空字符串表示保留当前值；输入 null 表示设为 None。")

        while issues:
            for issue in issues:
                self._print(f"\n字段: {issue.field_name}")
                self._print(f"问题: {issue.message}")
                self._print(f"当前值: {issue.current_value!r}")

                while True:
                    raw = self._input(f"请输入新的 {issue.field_name} 值: ").strip()

                    if raw == "":
                        # 用户选择保留原值，但如果原值本来就不合法，继续提示
                        new_value = getattr(config, issue.field_name)
                    else:
                        try:
                            new_value = self._parse_field_value(issue.field_name, raw)
                        except ValueError as e:
                            self._error(str(e))
                            continue

                    setattr(config, issue.field_name, new_value)

                    # 仅重新校验一次整个配置，看该问题是否被修复
                    new_issues = validator.validate(config)
                    same_issue = [x for x in new_issues if x.field_name == issue.field_name]
                    if same_issue:
                        self._error(f"{issue.field_name} 仍不合法：{same_issue[0].message}")
                        continue
                    break

            issues = validator.validate(config)

            if issues:
                self._print("\n仍有以下问题未解决：")
                for i in issues:
                    self._print(f"- {i.field_name}: {i.message}")
                answer = self._input("是否继续修复？(y/n): ").strip().lower()
                if answer not in ("y", "yes", "是"):
                    raise RuntimeError("用户取消配置修复，程序退出。")

        self._print("配置修复完成。")
        return config

    def repair_required_only(self, config: LiteLLMConfig,
                             validator: ConfigManager) -> LiteLLMConfig:
        """只修复开始对话前必须补齐的关键配置项"""
        issues = validator.get_missing_required_issues(config)
        if not issues:
            return config

        self._print("开始对话前，必须先补齐以下关键配置项。")

        while issues:
            for issue in issues:
                self._print(f"\n字段: {issue.field_name}")
                self._print(f"问题: {issue.message}")
                self._print(f"当前值: {issue.current_value!r}")

                while True:
                    raw = self._input(f"请输入 {issue.field_name} 的值: ").strip()
                    if not raw:
                        self._error(f"{issue.field_name} 是必需项，不能为空")
                        continue
                    
                    try:
                        new_value = self._parse_field_value(issue.field_name, raw)
                    except ValueError as e:
                        self._error(str(e))
                        continue

                    setattr(config, issue.field_name, new_value)

                    new_issues = validator.get_missing_required_issues(config)
                    same_issue = [x for x in new_issues if x.field_name == issue.field_name]
                    if same_issue:
                        self._error(f"{issue.field_name} 仍不合法：{same_issue[0].message}")
                        continue
                    break

            issues = validator.get_missing_required_issues(config)

        self._print("关键配置项已补齐。")
        return config

    def _parse_field_value(self, field_name: str, raw: str):
        if raw.lower() == "null":
            return None

        if field_name in ("model", "api_key", "api_base", "custom_llm_provider"):
            return raw

        if field_name in ("temperature", "top_p", "presence_penalty",
                          "frequency_penalty", "timeout", "retry_delay"):
            return float(raw)

        if field_name in ("max_tokens", "seed", "max_retries"):
            return int(raw)

        if field_name in ("stop", "fallbacks"):
            # 支持逗号分隔输入
            items = [x.strip() for x in raw.split(",") if x.strip()]
            return items

        if field_name in ("response_format", "extra"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{field_name} 必须是合法 JSON: {e}")

        return raw


def create_arg_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Mama AI Chat - 配置支持配置文件、环境变量和命令行参数"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="指定配置文件路径（默认：当前目录下的 .mama.config.yml）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="AI 模型名称，例如：gpt-4, ollama/llama2"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        dest="api_key",
        help="API 密钥"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        dest="api_base",
        help="API 基础 URL"
    )
    
    parser.add_argument(
        "--custom-llm-provider",
        type=str,
        dest="custom_llm_provider",
        help="自定义 LLM 提供商"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="温度参数 (0-2)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        help="top_p 参数 (0-1)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        dest="max_tokens",
        help="最大 token 数"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        help="请求超时时间（秒）"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        dest="max_retries",
        help="最大重试次数"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子"
    )
    
    return parser

