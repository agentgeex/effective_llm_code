"""
【本书配套示例】第二个 FastMCP 服务端：1 个 resource + 2 个 prompt（省份介绍 / 代码调试）
① 这是什么：专演示 prompt 原语的服务端。test://hello 资源 + introduce_china_province /
   debug_code 两个提示模板；debug_code 是「贴代码 + 贴报错 → 让模型定位问题」的对话模板。
② 对应书稿：ch07 §7.3.2 FastMCP、§7.3.3 开发MCP服务器端。
③ 怎么跑：
   .venv/bin/python ch07/mcp_server1.py       # SSE，监听 http://0.0.0.0:8002/sse
   # 验证：curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8002/sse   → 200
   # 客户端：任意 fastmcp Client 连 http://127.0.0.1:8002/sse
   #   （ch07/mcp_client.py 写的是 8011，连本文件要改这一行地址）
④ 依赖：第三方包 fastmcp>=2.0（实测 4.1.4）+ mcp（实测 2.3.0）。
   不需要 psycopg2 —— 书稿里那两行 PostgreSQL 导入本就是注释状态，本章用不到数据库。
⑤ 实测解释器与结果：python
   （Python 3.10.12 / fastmcp 4.1.4 / mcp 2.3.0）， 复测
   - SSE 起服务：GET http://127.0.0.1:8002/sse → HTTP 200, content-type: text/event-stream
   - list_resources → ['test://hello']；read_resource('test://hello') → 'Hello, World!'
   - list_prompts → ['introduce_china_province', 'debug_code']
   - get_prompt('introduce_china_province', {province:'浙江'})
       → 四段式模板正文（历史沿革 / 人文地理风俗 / 经济发展 / 旅游建议）
   - get_prompt('debug_code', {code:'print(1/0)', error_message:'ZeroDivisionError'})
       → 含代码块与报错原文的调试指令全文
⑥ 与书稿的差异：
   1. import：书稿 `from mcp.server.fastmcp import FastMCP` 在 mcp 2.3.0 实测抛
      ModuleNotFoundError（2.x 改名为 `mcp.server.MCPServer`），这里用第三方包 `from fastmcp import FastMCP`。
   2. 构造参数：fastmcp 的 FastMCP 不接受 host/port（书稿写在构造函数里），
      移到 `mcp.run(transport="sse", host="0.0.0.0", port=8002)`。
   3. debug_code 返回值：书稿用 mcp 1.x 的 base.SystemMessage/UserMessage 拼多角色消息，
      2.x 已无此路径；这里合并成单个字符串，语义等价、两种 SDK 都能接收。
   资源与两个 prompt 的正文措辞与书稿一致。
"""

# 导入必要的模块
# FastMCP 是**第三方包 fastmcp**（pip install fastmcp）。mcp 2.x 里
# `mcp.server.fastmcp` 这个导入路径已被刻意做成不可用（会抛 ModuleNotFoundError），
# 官方 SDK 2.x 内置的同类叫 MCPServer。这里用第三方 fastmcp，写法与书稿一致。
from fastmcp import FastMCP
#import psycopg2  # PostgreSQL 数据库连接库
import json  # JSON 处理库
#from psycopg2.extras import RealDictCursor  # 使查询结果以字典形式返回的游标

# host/port 不写进构造函数（fastmcp 的 FastMCP 不接受这两个参数），改到 mcp.run(...) 里传。
mcp = FastMCP("MCP Test Server")


@mcp.resource("test://hello")
def hello() -> str:
    """简单的测试资源"""
    return "Hello, World!"

# 中国省份介绍
@mcp.prompt()
def introduce_china_province(province: str) -> str:
    """介绍中国省份

    参数:
    province: 省份名称
    """
    return f"""
    请介绍这个省份：{province}

    要求介绍以下内容：
    1. 历史沿革
    2. 人文地理、风俗习惯
    3. 经济发展状况
    4. 旅游建议
    """

@mcp.prompt()
def debug_code(code: str, error_message: str) -> str:
    """调试代码的对话式提示模板（返回单个字符串：fastmcp 与官方 MCPServer 都接收）

    参数:
    code: 需要调试的代码
    error_message: 错误信息
    """
    # 书稿原版用 mcp 1.x 的 base.SystemMessage / UserMessage 拼多角色消息，
    # 那条导入路径在 2.x 已不存在；这里合并成一段文本，语义等价、两种 SDK 都能跑。
    return (
        "你是一位专业的代码调试助手。请仔细分析用户提供的代码和错误信息，"
        "找出问题所在并提供修复方案。\n\n"
        "我的代码有问题，请帮我修复：\n"
        f"```\n{code}\n```\n"
        f"错误信息：\n{error_message}"
    )


if __name__ == "__main__":
    # sse 传输：SSE 客户端连 http://<host>:8002/sse
    mcp.run(transport="sse", host="0.0.0.0", port=8002)
