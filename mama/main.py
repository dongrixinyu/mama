from pathlib import Path

from mama.config import (
    ConfigManager,
    ConfigInteractiveEditor,
    create_arg_parser,
)
from mama.core.chat_manager import ChatManager
from mama.interface.cli_interface import CLIInterface
from mama.model.litellm_model import LiteLLMModel, LiteLLMConfig


def main():
    # 解析命令行参数
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # 创建CLI界面
    interface = CLIInterface()

    # 创建配置管理器（支持自定义配置文件路径）
    config_manager = ConfigManager(args.config if hasattr(args, 'config') and args.config else None)
    
    # 如果默认配置文件不存在，创建一个
    if not config_manager.config_path.exists():
        default_config = LiteLLMConfig()
        config_manager.save(default_config)
        interface.display_system_message(
            f"未找到配置文件 {config_manager.config_path}，已自动生成默认配置文件。"
        )

    # 按优先级加载配置：配置文件 < 环境变量 < 命令行参数
    litellm_config = config_manager.load(cli_args=args)

    # 检查是否有必须补齐的关键配置项
    missing_issues = config_manager.get_missing_required_issues(litellm_config)
    if missing_issues:
        editor = ConfigInteractiveEditor(interface)
        try:
            litellm_config = editor.repair_required_only(litellm_config, config_manager)
            config_manager.save(litellm_config)
            interface.display_system_message("关键配置已补齐并保存。")
        except Exception as e:
            interface.display_error(f"配置补齐失败: {e}")
            return

    # 创建AI模型
    try:
        ai_model = LiteLLMModel(litellm_config)
    except Exception as e:
        interface.display_error(f"初始化 AI 模型失败: {e}")
        return

    # 创建聊天管理器
    chat_manager = ChatManager(interface, ai_model)

    # 运行聊天
    chat_manager.run()


if __name__ == "__main__":
    main()

