import traceback
from typing import Optional
from .chat_interface import ChatInterface, AIModel, ChatSession, Message


class ChatManager:
    """聊天管理器，协调界面和AI后端"""

    def __init__(self, interface: ChatInterface, ai_model: AIModel):
        self.interface = interface
        self.ai_model = ai_model
        self.session = ChatSession(messages=[])

    def run(self) -> None:
        """运行聊天循环"""
        self.interface.display_system_message("欢迎使用AI聊天助手！输入 'exit' 或 'quit' 退出。")

        while True:
            # 获取用户输入
            user_input = self.interface.get_user_input()

            if user_input is None:
                break

            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', '退出']:
                self.interface.display_system_message("再见！")
                break

            # 添加用户消息
            user_message = Message(role='user', content=user_input)
            self.session.messages.append(user_message)

            # 获取AI回复
            try:
                ai_content = self._get_ai_response()
                ai_message = Message(role='assistant', content=ai_content)
                self.session.messages.append(ai_message)
                self.interface.display_message(ai_message)
            except Exception as e:
                print(traceback.format_exc())
                self.interface.display_error(f"发生错误: {str(e)}")

    def _get_ai_response(self) -> str:
        """获取AI回复（支持流式或完整）"""
        # 这里可以选择流式或完整模式
        return self.ai_model.generate_response_complete(self.session)

    def _handle_config_command(self, user_input: str) -> None:
        if not self.config:
            self.interface.display_error("当前未启用配置管理。")
            return

        parts = user_input.split(maxsplit=3)

        if len(parts) == 1:
            self.interface.display_system_message(
                "配置命令：\n"
                "/config show               查看当前配置\n"
                "/config set <key> <value>  设置配置项\n"
                "/config validate           校验配置\n"
                "/config save               保存配置到文件\n"
            )
            return

        sub = parts[1].lower()

        if sub == "show":
            config_dict = asdict(self.config)
            for k, v in config_dict.items():
                self.interface.display_system_message(f"{k} = {v!r}")
            return

        if sub == "validate":
            if not self.config_manager:
                self.interface.display_error("未提供配置管理器。")
                return
            issues = self.config_manager.validate(self.config)
            if not issues:
                self.interface.display_system_message("配置校验通过。")
            else:
                for issue in issues:
                    self.interface.display_error(f"{issue.field_name}: {issue.message}，当前值={issue.current_value!r}")
            return

        if sub == "save":
            if not self.config_manager:
                self.interface.display_error("未提供配置管理器。")
                return
            self.config_manager.save(self.config)
            self.interface.display_system_message("配置已保存。")
            return

        if sub == "set":
            if len(parts) < 4:
                self.interface.display_error("用法: /config set <key> <value>")
                return

            key = parts[2]
            raw_value = parts[3]

            if key not in LiteLLMConfig.__dataclass_fields__:
                self.interface.display_error(f"未知配置项: {key}")
                return

            editor = ConfigInteractiveEditor(self.interface)
            try:
                value = editor._parse_field_value(key, raw_value)
            except ValueError as e:
                self.interface.display_error(str(e))
                return

            old_value = getattr(self.config, key)
            setattr(self.config, key, value)

            if self.config_manager:
                issues = self.config_manager.validate(self.config)
                key_issues = [x for x in issues if x.field_name == key]
                if key_issues:
                    setattr(self.config, key, old_value)
                    self.interface.display_error(f"设置失败: {key_issues[0].message}")
                    return

            self.interface.display_system_message(f"已更新 {key} = {value!r}")

            # 如果模型相关配置变化，重建 ai_model
            if key in {"model", "api_key", "api_base", "custom_llm_provider", "timeout", "max_tokens"}:
                try:
                    self.ai_model = LiteLLMModel(self.config)
                    self.interface.display_system_message("AI 模型已根据新配置重新初始化。")
                except Exception as e:
                    setattr(self.config, key, old_value)
                    self.interface.display_error(f"重建 AI 模型失败，已回滚: {e}")
            return

        self.interface.display_error(f"未知子命令: {sub}")