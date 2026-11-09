"""
【本书配套示例】用官方 mcp SDK（非 FastMCP）写一个最小 MCP Server：1 个工具 get_weather
对应章节：ch07 §7.2.2 自己动手写一个 MCP Server
运行方式：
    # 这是 stdio 服务端，直接 `python ch07/weather_server.py` 会阻塞在读 stdin 上，属正常。
    # 1) 用 MCP Inspector 调试（书稿对应小节 的命令）
    npx @modelcontextprotocol/inspector python ch07/weather_server.py
    # 2) 用任意 MCP 客户端以子进程方式拉起（本文件末尾给了 8 行自测脚本）
依赖：mcp>=2.0（实测 **2.3.0**）。缺 mcp → ImportError: No module named 'mcp'
    实测解释器：.venv/bin/python（mcp 2.3.0；venv-onnx 里没有 mcp，会 ModuleNotFoundError）
实测状态：✅ 跑通（本文件重跑取证）
    客户端：fastmcp 4.1.4 的 PythonStdioTransport 拉起本文件
        tools = ['get_weather']
        get_weather({"city":"Beijing"}) -> Beijing天气：晴，温度 28°C，湿度 45%
        get_weather({"city":"Nowhere"}) -> Nowhere天气：数据未覆盖，温度 --°C，湿度 --
        （客户端打印的完整形式是 get_weatherOutput(result='...') —— 那是 fastmcp
          客户端的结构化输出包装，不是服务端多包了一层；文本内容就是上面这两行。）
    实测时间：

与书稿差异（**这条是硬差异，必须说清**）：
    书稿对应小节 用的是 mcp 1.x 的低层 API：
        from mcp.server import Server
        @server.list_tools()   # 手动返回 Tool(...) 列表，自己写 inputSchema
        @server.call_tool()    # 手动分发 name/arguments
    本机装的是 mcp **2.3.0**，`Server` 已经没有 `list_tools` / `call_tool` 这两个装饰器，
    照抄书稿会直接报（我实测过）：
        AttributeError: 'Server' object has no attribute 'list_tools'
    2.x 里对应的东西叫 `mcp.server.MCPServer`，并且**不再接受手写 JSON Schema** ——
    工具的 schema 由函数签名与类型注解自动推导。所以本文件按 2.x 的写法重写：
        from mcp.server import MCPServer
        mcp = MCPServer("weather-server")
        @mcp.tool()
        def get_weather(city: str) -> str: ...
        mcp.run_stdio_async()   # 等价于 1.x 的 stdio_server() + server.run(...)
    业务逻辑（三城市模拟数据、未覆盖城市兜底）与书稿**完全一致**，只是外壳换了。
    想读「手写 inputSchema」那种老写法，请看 mcp 1.x 的源码历史；2.x 已无此路径。
"""

import asyncio

from mcp.server import MCPServer

# 创建一个 MCP Server 实例
mcp = MCPServer("weather-server")

# 模拟天气数据（实际项目中这里会调真实 API）
WEATHER_DATA = {
    "Beijing": {"temp": 28, "condition": "晴", "humidity": "45%"},
    "Shanghai": {"temp": 32, "condition": "多云", "humidity": "70%"},
    "Shenzhen": {"temp": 30, "condition": "阵雨", "humidity": "85%"},
}


@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息

    参数:
        city: 城市名称，如 'Beijing'、'Shanghai'
    """
    info = WEATHER_DATA.get(city, {"temp": "--", "condition": "数据未覆盖", "humidity": "--"})
    return f"{city}天气：{info['condition']}，温度 {info['temp']}°C，湿度 {info['humidity']}"


if __name__ == "__main__":
    # 启动 Server：通过标准输入/输出与客户端通信（1.x 里要自己开 stdio_server()）
    asyncio.run(mcp.run_stdio_async())

# ---------------------------------------------------------------------------
# 自测（不需要 Inspector）：
#
#   .venv-core/bin/python - <<'PY'
#   import asyncio, sys
#   from fastmcp import Client
#   from fastmcp.client.transports import PythonStdioTransport
#   async def main():
#       async with Client(PythonStdioTransport("ch07/weather_server.py", python_cmd=sys.executable)) as c:
#           print([t.name for t in await c.list_tools()])
#           print((await c.call_tool("get_weather", {"city": "Beijing"})).data)
#   asyncio.run(main())
#   PY
# ---------------------------------------------------------------------------
