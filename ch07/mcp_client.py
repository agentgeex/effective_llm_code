"""
【本书配套示例】FastMCP 客户端：连一个运行中的 MCP Server（SSE），列工具 / 读资源
① 这是什么：MCP 客户端最小闭环 —— 连上 SSE 服务端 → list_tools → read_resource，
   并把输出 schema 自动生成的工具清单打成 JSON 打印，让人看清「工具描述就是给模型的说明书」。
② 对应书稿：ch07 §7.3.5 MCP部署与客户端调用。
   书稿对应小节 的 client_demo.py 是同一职责的另一种写法（mcp SDK 低层 ClientSession + stdio）。
③ 怎么跑：
   # 1) 先起服务端（本文件连 8011，对应 ch07/mcp_server.py）
   .venv/bin/python ch07/mcp_server.py --sse
   # 2) 再跑客户端
   .venv/bin/python ch07/mcp_client.py
④ 依赖：第三方包 fastmcp>=2.0（实测 4.1.4）+ mcp（2.3.0）；需要本地 8011 端口上有一个服务端。
⑤ 实测解释器与结果：python
   （fastmcp 4.1.4 / mcp 2.3.0）， 复测，rc=0
   - Available tools → add / subtract / multiply / divide（每个都带 input_schema、output_schema）
   - Available resource → TextResourceContents(uri='config://version', text='2.0.1')
⑥ 与书稿的差异（**这条必须说清**）：
   1. **不是书稿对应小节 那段代码**。书稿那版是
          from mcp import ClientSession, StdioServerParameters
          from mcp.client.stdio import stdio_client
      以 stdio 拉起 task_server.py，调 create_task / list_tasks，并演示 call_tool；
      本文件是 fastmcp 高层 Client + SSE，连 mcp_server.py，只演示 list_tools / read_resource。
      两者都能跑（书稿版  实测通过，命令与输出见 ch07/README §6），但不是同一份实现。
   2. 文件里注释掉的 3 个 modelscope 公网 SSE 地址**已失效**： 实测
      curl https://mcp.api-inference.modelscope.net/ba385c5ffb0d42/sse → HTTP 410 Gone。
      一律连本地 ch07/mcp_server.py。
   3. call_tool 那两行仍是注释状态（书稿此处只演示 list/read）；要验证可取消注释，
      add(5,3) 应得 8.0。
"""

## https://gofastmcp.com/servers/prompts

from fastmcp import Client
import asyncio
import json

async def main():
    # 测试 mcp 客户端的功能
    async with Client("http://127.0.0.1:8011/sse") as mcp_client:
    #async with Client("https://mcp.api-inference.modelscope.net/ba385c5ffb0d42/sse") as mcp_client:
    #async with Client("https://mcp.api-inference.modelscope.net/56711a274cfb41/sse") as mcp_client:
    #async with Client("https://mcp.api-inference.modelscope.net/ebd85a2d2d3849/sse") as mcp_client:
        #print(dir(mcp_client))  
        tools = await mcp_client.list_tools()
        resource = await mcp_client.read_resource("config://version")
        print(f"Available tools: {tools}")
        print(f"Available resource: {resource}")
        dict_list = [tool.__dict__ for tool in tools]
        tools_description = json.dumps(dict_list,indent=6,ensure_ascii=False)
        print(f"Available tools: {tools_description}")

        # result = await mcp_client.call_tool("add", {"a": 5, "b": 3})
        # print(f"Result: {result[0].text}")


if __name__ == "__main__":
    asyncio.run(main())       