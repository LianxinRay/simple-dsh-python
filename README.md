# simple-dsh-python

DeepSeek Harness **P0 + P1 + P2** 的最小 Python 移植：一个零依赖（仅标准库，Python ≥ 3.11）的插件化 agent harness，复刻 `deepseek-harness` 的核心架构契约。已用真实 DeepSeek API 端到端验证：模型流式输出 → 工具调用（创建并读回文件）→ 会话日志落盘回放。

## 模块与上游对应

| 包 | 上游包 | 职责 |
|---|---|---|
| `simple_dsh.cordis` | vendored Cordis | 插件运行时：`Context` 服务仓库、`inject` 依赖挂载、四种事件派发（`emit`/`waterfall`/`parallel`/`serial`）、可逆 effect 生命周期 |
| `simple_dsh.llm` | `packages/llm/llm` | 对话词汇：`Message` / `ContentBlock`（text、reasoning、image、tool-call、tool-result）/ `StreamChunk`、adapter seam、`StreamAssembler` |
| `simple_dsh.session` | `packages/core/session` | append-only `SessionEvent` 日志（单一事实源）、`derive_messages()` 投影、JSONL 持久化与回放 |
| `simple_dsh.prompts` | `packages/core/system-prompt` | prompt 段落注册（优先级排序）与装配 |
| `simple_dsh.tools` | `packages/core/tools` | 工具注册表 + 守卫执行管道（`tools/pre-execute` → guards → `tools/execute` → `tools/post-execute` 三段 waterfall，异常归一化为 `is_error`） |
| `simple_dsh.agent` | `packages/core/agent` + `agent-loop` | `Agent` 句柄（inbox、`prompt()`/`inject()`/`cancel()`）与 turn/step 状态机 |
| `simple_dsh.llm.deepseek` | `packages/llm/llm-deepseek` | DeepSeek/OpenAI 兼容 SSE 流式适配：请求映射、`OpenAiStreamTranslator`（增量 → `StreamChunk`）、stdlib-only chunked 传输、opener 可注入以便无网测试 |
| `simple_dsh.credentials` | `packages/credentials` | env-over-`.env` 凭证解析，`require()` 缺失即报错 |
| `simple_dsh.tools.fs_tools` | `packages/fs` | `read_file`/`write_file`/`edit_file`/`list_directory`，工作区根目录逃逸拒绝 |
| `simple_dsh.tools.shell` | `packages/shell` | `bash` 工具：子进程执行、输出截断、非零退出码 → `is_error` |
| `simple_dsh.guard` | `packages/guard` | 工具调用超时（`tools/execute` 环绕 waterfall）、连续重复调用拒绝 guard |
| `simple_dsh.app` | profile/bundle 组合 | `create_app()` 一行组装全部服务与守卫，`create_agent()` 接线会话落盘 |
| `simple_dsh.approval` | `packages/interaction` | `ctx.approval` 一次性确认；`tools/pre-execute` 上的门控监听器；无 responder 一律拒绝（fail-closed） |
| `simple_dsh.compaction` | `packages/compaction` | 超 token 预算时摘要旧历史，`compaction/summary` 事件作为派生边界，kept 消息原文保留，日志不重写 |
| `simple_dsh.tools.todo` | `packages/todo` | `todo_write` 工具；全量快照落日志，重放取最新；不进派生历史 |
| `simple_dsh.tools.subagent` | `packages/subagent` | `delegate` 工具：子 agent 独立 session、共享上下文，返回最终回答与步数 meta |
| `simple_dsh.tools.web` | `packages/web` | `web_fetch`：stdlib 抓取 + HTML 正文提取 + 截断 |
| `simple_dsh.session.sqlite_store` | `packages/session` SQLite 后端 | 事件存 SQLite；单调 `SCHEMA_VERSION`，不匹配即拒绝打开 |
| `simple_dsh.preset` | `packages/preset` | JSON 组合文档（model/工具开关/审批门控/guard/压缩预算），深合并默认值，缺引用即报错 |

## 保留的架构不变量

- **模型可见 ⟺ 已入日志**：每步先把输入 append 进日志，再从日志派生模型历史；所有事件 payload 在 append 时校验为无损 JSON。
- **waterfall 语义**：监听器收到 `(payload, next)`；不调用 `next()` 即短路，返回非 `None` 替换上游结果，委托后返回 `None` 传播下游结果。
- **turn/step 词汇**：`turn/start → step/start → user/message → assistant/chunk* → assistant/message → tool/call → tool/result → step/end → turn/end`；工具调用欠一次后续请求（continuation step）。
- **`agent/pre-step` 决定模型看到什么**：返回改写后的消息列表；返回 `REJECT` 哨兵或空列表的首个 claim 会让 turn 无 step 关闭。
- **inject 语义**：`agent.inject()` 的合成上下文在 inbox 中等待，直到真实 prompt 唤醒驱动器。

## 与上游的刻意差异

- TypeScript declaration merging（可扩展事件/块词汇）→ Python 运行时注册（`Session.register_projection`、普通字符串事件类型）。
- 类型安全由 dataclass + 运行时 JSON 校验承担，无 pydantic 依赖。
- `REJECT` 哨兵替代 TS 的 `reject | enter(messages)` 联合返回类型。

## 使用

### 配置

在项目根目录创建 `.env`（已被 `.gitignore` 排除，不会入库）：

```dotenv
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com   # 可选，此为默认值
```

环境变量优先于 `.env`；缺 key 时 `require()` 直接报错。

### 运行

```bash
python -m unittest discover -s tests   # 79 个单测
python examples/context_demo.py        # Context 机制最小演示（注册表 + waterfall + effect）
python examples/echo_demo.py           # 端到端 demo（脚本化 adapter，含 JSONL 回放）
python examples/deepseek_demo.py       # 真实 DeepSeek API demo（无 DEEPSEEK_API_KEY 自动跳过）
```

### 用 preset 定制组合

```json
{
  "model": "deepseek-chat",
  "tools": {"shell": false, "web": true},
  "approval": {"tools": ["write_file", "edit_file"]},
  "compaction": {"max_tokens": 4000}
}
```

```python
ctx = create_app("./workspace", preset="mypreset.json", env_path=".env",
                 approval_responder=my_responder)
```

一行组装（含 DeepSeek adapter、fs/shell 工具、超时与重复调用 guard、会话落盘）：

```python
from simple_dsh.app import create_agent, create_app

ctx = create_app("./workspace", env_path=".env", log_path="session.jsonl")
agent = create_agent(ctx)
await agent.prompt("建一个 hello.txt，然后读回来告诉我内容")
await agent.when_idle()
```

最小手动示例：

```python
import asyncio
from simple_dsh.agent import Agent
from simple_dsh.cordis import Context
from simple_dsh.llm import LlmRegistry
from simple_dsh.prompts import SystemPrompt
from simple_dsh.tools import ToolRegistry

async def main():
    ctx = Context()
    ctx.service("llm", LlmRegistry())        # 注册你的 adapter
    ctx.service("systemPrompt", SystemPrompt())
    ctx.service("tools", ToolRegistry(ctx))
    agent = Agent(ctx, model="your-model")
    await agent.prompt("你好")
    await agent.when_idle()

asyncio.run(main())
```

## 项目结构

```
simple_dsh/
  cordis/       插件运行时（Context / Service / 事件 / effect）
  llm/          对话词汇、assembler、deepseek 适配器
  session/      事件日志、投影、JSONL 与 SQLite 持久化
  tools/        注册表与管道、fs 工具、shell 工具、todo、subagent、web
  agent/        Agent 句柄与 turn/step 驱动器
  prompts.py    system prompt 装配
  credentials.py  env/.env 凭证
  guard.py      超时与重复调用守卫
  approval.py   审批交互（fail-closed）
  compaction.py 上下文压缩
  preset.py     JSON 组合文档
  app.py        create_app() 组合入口
tests/          79 个单元测试（stdlib unittest）
examples/       context_demo / echo_demo / deepseek_demo
```

## 后续路线（对应上游优先级）

- **P3**：terminal（PTY）、LSP、sandbox（进程约束）、jobs/workflow、SDK/ACP、Web host/client 等外围能力与交付形态
