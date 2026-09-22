# mama

Mama 提供 CLI 和基于 Qt（PySide6）的 OpenAI 兼容模型调用界面。

## 图形界面

```bash
python3 -m mama --gui
# 或兼容旧入口：
python3 foreign_llm_gui.py
```

首次启动会创建 `~/.mama/models.json`（权限为仅当前用户可读写）。点击“编辑配置”填写：

- 配置名称
- 完整的 OpenAI 兼容 Chat Completions URL，例如 `https://example.com/v1/chat/completions`
- API Key
- 模型名、temperature、max tokens 与超时

可以保存、切换、新增及删除多个端点配置。请求在后台 Qt 线程执行，输出与 reasoning 内容会实时显示；“终止”会在下一个流式数据块到达时取消请求。成功响应同时保存到 `~/.mama/responses/`。

> 不要将 API Key 提交到仓库。旧版 Tkinter 脚本中硬编码的 Key 已移除。

## CLI

```bash
python3 -m mama --help
python3 -m mama --model gpt-4o-mini --api-key "$OPENAI_API_KEY"
```

CLI 仍使用原来的 LiteLLM 配置路径及优先级；GUI 的端点配置独立保存在 `~/.mama/models.json`。

