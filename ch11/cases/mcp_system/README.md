# 手搓一个完整的 MCP 体系（§11.5）

## 这个案例干什么

从零写一个 MCP **服务端**和**客户端**，跑完一整圈协议交互，而不是只调用某个现成工具。

| 文件 | 干什么 |
|---|---|
| `server.py` | MCP Server：3 个工具（如统计字数、检索片段）+ 1 个 Resource + 1 个 Prompt |
| `client.py` | MCP Client：拉起/连上服务端 → `list_tools` → `call_tool` → `read_resource` → 取 Prompt |
| `sample_book/` | 两篇自带语料，作为 Resource 的默认内容 |

这个案例刻意选了**最小的闭环**：不接数据库、不接外部 API，只验证「协议本身怎么跑通」。

## 对应书哪一节

第 11 章 §11.5「手搓一个完整的 MCP 体系」：§11.5.2 手搓第一个 MCP Server、§11.5.3 手搓一个 MCP Client、§11.5.4 协议本质（JSON-RPC + 能力协商）。

## 怎么跑

```bash
pip install "mcp>=2.0"

# 一条命令跑完整闭环（client 会自动拉起 server）
python ch11/cases/mcp_system/client.py

# 也可以单独起服务端，再用别的客户端连
python ch11/cases/mcp_system/server.py --transport stdio
python ch11/cases/mcp_system/server.py --transport streamable-http --host 127.0.0.1 --port 8080
```

服务端支持 `--transport` / `--host` / `--port` 三个参数（`stdio`、`sse`、`streamable-http`）。

## 依赖什么

| 依赖 | 说明 |
|---|---|
| `mcp`（**2.x**） | 1.x 的 `FastMCP` 在 2.x 改名 `MCPServer`，导入路径也不一样；代码按 2.x 写 |
| 其他 | 无。不需要 API key，不需要数据库，不需要模型权重 |

**如果你装的是 mcp 1.x**：`from mcp.server.mcpserver import MCPServer` 会失败。要么升到 2.x，要么把导入改回 `from mcp.server.fastmcp import FastMCP`（书里那一版写法）。

## 数据从哪来

`sample_book/` 里有两篇自带的示例文档，作为 Resource 返回内容的默认来源 —— **开箱即跑**，不需要你准备任何文件。

想指向自己的目录：设置环境变量 `MCP_BOOK_ROOTS="名字=/路径"`，工具参数传逻辑名即可。

## 提醒

- 这是**协议演示**。生产环境的 MCP 服务要考虑鉴权、超时、资源隔离和路径穿越防护 —— 本案例为了把协议讲清楚，把这些都简化了。
- Resource URI 用的是**逻辑名**而不是绝对路径。把绝对路径拼进 URI 在 mcp 2.x 默认配置下读不出来（RFC 6570 的 `{var}` 不匹配 `/`，且 2.x 默认拒绝绝对路径），这不是 bug，是安全门的正常行为。
