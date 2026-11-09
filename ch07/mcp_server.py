"""
【本书配套示例】最小 FastMCP 服务端：2 个 resource + 1 个 prompt + 4 个 tool（加减乘除）
① 这是什么：用 fastmcp 写的一个完整 MCP Server，演示 resource / resource template / prompt /
   tool 四类原语。工具是加减乘除四则运算，divide 带除零保护。
② 对应书稿：ch07 §7.3.2 FastMCP、§7.3.3 开发MCP服务器端；
   同一份服务端的来历另见 ch11 §11.5.2 手搓第一个 MCP Server。
③ 怎么跑：
   # SSE 传输（书稿 §7.3.3「模式二」那种部署方式），监听 http://127.0.0.1:8011/sse
   .venv/bin/python ch07/mcp_server.py --sse
   # stdio 传输（由客户端以子进程拉起，书稿默认方式）
   .venv/bin/python ch07/mcp_server.py
   # 配套客户端（连 8011）
   .venv/bin/python ch07/mcp_client.py
④ 依赖：第三方包 fastmcp>=2.0（实测 4.1.4）+ 其依赖 mcp（实测 2.3.0）。无需 key、无外部服务。
   🔴 书稿写 `from mcp.server.fastmcp import FastMCP`（mcp 1.x 自带封装）；mcp 2.3.0 把该导入路径
   **故意**做成抛错，实测原文：
       ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where
       FastMCP was renamed to MCPServer ... or pin 'mcp<2' to keep running v1 code.
   故本文件改用第三方包 `from fastmcp import FastMCP`（书稿 v1 的 FastMCP 正是它的前身），
   不要照抄书稿的 import 行。
⑤ 实测解释器与结果：python
   （Python 3.10.12 / fastmcp 4.1.4 / mcp 2.3.0）， 复测
   - SSE 起服务：GET http://127.0.0.1:8011/sse → HTTP 200, content-type: text/event-stream
   - list_tools → ['add', 'subtract', 'multiply', 'divide']
   - list_resources → ['config://version']；list_resource_templates → ['weather://{city}/today']
   - list_prompts → ['customer_service_prompt']
   - call_tool('add', {a:5,b:3}) → 8.0
   - call_tool('divide', {a:1,b:0}) → 客户端默认抛 ToolError：
       Error calling tool 'divide': 除数不能为零
     （传 raise_on_error=False 时返回 is_error=True，内容同上；除零保护生效）
   - read_resource('config://version') → '2.0.1'
   - get_prompt('customer_service_prompt', {user_query:'怎么退货？'})
       → '你是一个专业客服助手\n\n用户问题：怎么退货？'
⑥ 与书稿的差异：
   1. 外壳：import 由 `mcp.server.fastmcp` 换成第三方 `fastmcp`（原因见 ④），业务逻辑一字未改。
   2. prompt 返回值：书稿返回 list[dict]（含 system 角色）。MCP 协议的 prompt 只允许 user/assistant，
      fastmcp 4.x 实测报 `MCPError: messages[0] must be Message or str, got dict`；
      改为返回单个字符串（系统指令在前、用户问题在后），语义不变。
   3. 末尾加 `--sse` 开关：书稿让你手工把 mcp.run() 改成 SSE，这里一个开关切换两种传输。
   4 个工具的名称/签名/docstring、config://version 资源、weather://{city}/today 资源模板与书稿一致。
"""

import sys

from fastmcp import FastMCP
import json

mcp = FastMCP(name="MyAssistantServer")


@mcp.resource("weather://{city}/today")
async def get_weather(city: str) -> dict:
    """Fetch today's weather for a city."""
    return_d = {f'{city}':33.1}
    return json.dumps(return_d)

@mcp.resource("config://version")
def get_version(): 
    return "2.0.1"


# 说明：MCP 的 prompt 消息只支持 user / assistant 角色，system 不在协议里。
# 书稿写法返回 [{"role":"system"...},{"role":"user"...}]，在 fastmcp 4.x 会直接报错，
# 故合并成单个字符串返回（系统指令在前，用户问题在后）。
@mcp.prompt()
def customer_service_prompt(user_query: str) -> str:
    return f"你是一个专业客服助手\n\n用户问题：{user_query}"

@mcp.tool()
def add(a: float, b: float) -> float:
    """加法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之和
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """减法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之差 (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """乘法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之积
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """除法运算

    参数:
    a: 被除数
    b: 除数

    返回:
    两数之商 (a / b)

    异常:
    ValueError: 当除数为零时
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse", host="127.0.0.1", port=8011)
    else:
        mcp.run()  # 默认 stdio：由客户端（如 mcp/mcp_client.py）以子进程方式拉起