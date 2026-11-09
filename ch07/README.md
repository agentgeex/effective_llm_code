# 第 7 章 MCP 协议：大模型的「神经突触」

## 这章讲什么

MCP（Model Context Protocol）是大模型调外部能力的标准接口 —— 相当于给模型插上「神经突触」，让它能查天气、查股票、读写数据库。这一章从函数调用的协议讲起，一路手写到能跑的服务端和客户端。

读完之后你会有三个能力：看懂一个 MCP Server 怎么定义工具、自己写一个、以及知道它和「直接写个 HTTP 接口」的区别在哪。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`weather_server.py`](weather_server.py) | 最小 MCP Server：1 个天气工具 + 未覆盖城市的兜底 | **A 可真跑** |
| [`stock_server.py`](stock_server.py) | 3 个资源 + 提示词 + 股票查询工具 | **A 可真跑** |
| [`task_server.py`](task_server.py) | 任务管理服务：建 / 列 / 完成 3 个工具，含非法参数校验 | **A 可真跑** |
| [`mcp_server.py`](mcp_server.py) | 完整示例：SSE 传输、4 工具 + 2 资源 + 1 提示词（含除零保护） | **A 可真跑** |
| [`mcp_server1.py`](mcp_server1.py) | SSE 版服务端，1 资源 + 2 提示词 | **A 可真跑** |
| [`mcp_client.py`](mcp_client.py) | 客户端：连上 `mcp_server.py`，列工具、读资源 | **A 可真跑** |
| [`mcp_server_mysql.py`](mcp_server_mysql.py) | 把 MySQL 包装成 MCP 工具（`query_database` / `list_tables` / `list_databases`） | **B 要条件**（要一个真实 MySQL + 库 + 口令；缺口令时启动即报清晰错误） |
| [`mcp_count_qw.py`](mcp_count_qw.py) | MCP + 大模型组合：工具负责取数、模型负责组织回答 | **B 要条件**（MCP 半边已跑通；模型半边要 `DEEPSEEK_ARK_API_KEY`） |
| [`function_calling_demo.py`](function_calling_demo.py) | 用 litellm 演示函数调用：模型返回 `tool_calls`，本地执行后回填 | **B 要条件**（要 `DEEPSEEK_API_KEY`；依赖可 import、端点可达） |

## 怎么跑

MCP 服务端有两种传输方式，跑法完全不同：

**一、stdio（标准输入输出）** —— 由客户端拉起，不能自己在终端里「跑起来看看」：

```bash
# 这样跑会阻塞在读 stdin 上，属正常现象，不是卡死
python3 ch07/weather_server.py
```

**二、SSE / HTTP** —— 服务端独立起，用浏览器或客户端连：

```bash
# 终端 1：起服务
python3 ch07/mcp_server.py --sse          # 监听 8011
python3 ch07/mcp_server1.py               # 监听 8002

# 终端 2：连上去看
python3 ch07/mcp_client.py
```

想用官方 Inspector 点点看（需要 Node 环境）：

```bash
npx @modelcontextprotocol/inspector
```

## 依赖与前置

| 依赖 | 谁需要 | 说明 |
|---|---|---|
| `mcp` 2.x | `weather_server.py` | 1.x 的 `FastMCP` 在 2.x 改名 `MCPServer`，导入路径也变了 |
| `fastmcp` | `stock_server.py` / `task_server.py` | 本仓库实测 4.1.4 |
| `pymysql` | `mcp_server_mysql.py` | 另需一个可连的 MySQL |
| `litellm` | `function_calling_demo.py` | 本仓库实测 1.106.6 |
| `openai` | `mcp_count_qw.py` | 另需 `DEEPSEEK_ARK_API_KEY` |

**这一章大部分文件不需要 key、不需要 GPU。** 只有 `mcp_server_mysql.py`（要数据库）、`mcp_count_qw.py` 和 `function_calling_demo.py`（要模型 key）除外。

## 常见报错

**`ModuleNotFoundError: No module named 'mcp.server.fastmcp'`**
你装的是 mcp 2.x，而书里那版代码写的是 1.x 的导入路径。2.x 里 `FastMCP` 改名 `MCPServer`，路径是 `mcp.server.mcpserver`。两个版本不兼容，看代码前先确认自己装的是哪个。

**`FileNotFoundError: [Errno 2] No such file or directory: 'python'`**
书里的客户端配置写的是 `python`，而很多 Linux 机器只有 `python3`。改成 `python3` 或 `sys.executable` 就好。书里 §7.3 的几处配置都有这个前提。

**`(1045, "Access denied for user 'root'@'172.17.0.1' (using password: YES)")`**
MySQL 拒绝了连接。`172.17.0.1` 是 Docker 网桥地址 —— 说明服务端从容器网络连出去的，而 MySQL 的账号授权里没有这个来源。换成你自己的库名、用户名和口令，代码不用改。**注意：本仓库的口令读 `MYSQL_PASSWORD` 环境变量，不硬编码。**

**连公共 MCP 平台（modelscope 之类）报 410 / 连接失败**
那些公开 SSE 地址已经失效（实测返回 `410 Gone`）。这一章的例子全部用本地 server，不依赖任何第三方在线服务。

**stdio 服务端「跑起来没反应」**
正常。stdio 传输靠标准输入输出通信，没客户端连它就没有输出，它会一直等。要验证功能，用 `mcp_client.py` 或 Inspector。
