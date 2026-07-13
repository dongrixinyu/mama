import sys
from typing import Optional
from mama.core.chat_interface import ChatInterface, Message
from mama.model.litellm_model import LiteLLMModel, LiteLLMConfig



class CLIInterface(ChatInterface):
    """命令行界面实现"""

    def __init__(self, prompt: str = "你: ", ai_prefix: str = "AI: "):
        self.prompt = prompt
        self.ai_prefix = ai_prefix

        self.backend = LiteLLMModel(
            LiteLLMConfig(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=1024,
                fallbacks=[
                    "claude-3-5-sonnet-20240620",
                    "gemini/gemini-1.5-flash",
                ],
            )
        )

    def display_message(self, message: Message) -> None:
        """显示消息"""
        if message.role == 'user':
            prefix = self.prompt
        else:
            prefix = self.ai_prefix

        print(f"\n{prefix}{message.content}")

    def get_user_input(self) -> Optional[str]:
        """获取用户输入"""
        try:
            print()  # 空行
            user_input = input(self.prompt)
            return user_input
        except (EOFError, KeyboardInterrupt):
            print()  # 换行
            return None

    def display_error(self, error: str) -> None:
        """显示错误"""
        print(f"\n❌ 错误: {error}", file=sys.stderr)

    def display_system_message(self, message: str) -> None:
        """显示系统消息"""
        print(f"\n💬 {message}")

    def clear_screen(self) -> None:
        """清屏"""
        import os
        os.system('clear' if os.name != 'nt' else 'cls')

