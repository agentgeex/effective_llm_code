#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：连上 server.py，走完 initialize → list → call → resource → prompt → 错误路径一整圈
# ② 对应书稿：ch11 §11.5.3「手搓一个 MCP Client」
# ③ 怎么跑：cd <仓库根> && python ch11/cases/mcp_system/client.py
# ④ 依赖：mcp 2.3.0；无 key、无外部服务
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 rc=0（✅ 全部步骤通过）
# ⑥ 与书稿的差异：书稿是 mcp 1.x；本文件适配 2.x 的 snake_case 字段（protocol_version / uri_template / resource_templates），API 名称未变
# ══════════════════════════════════════════════════════════════════
"""
ch11/cases/mcp_system/client.py — 连上 server.py，走完一整圈

对应书稿：ch11 §11.5.3「手搓一个 MCP Client」。

================================================================
⚠️ 版本差异（书稿写的是 mcp 1.x，本文件跑在 mcp 2.x 上）
================================================================
书稿的 Client 骨架（stdio 那部分）在 2.x 上**几乎不用改**：

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()
            await session.call_tool(name, args)

四个 API 名字一个没变。真正改的是**四个字段名**（2.x 全面转向 snake_case）
和一个新事实：

    书稿写法（1.x，驼峰）                 本文件（2.x，下划线）
    ------------------------------------  --------------------------------
    init_result.protocolVersion            init_result.protocol_version
    template.uriTemplate                   template.uri_template
    result.resourceTemplates               result.resource_templates
    （无）                                  result.structuredContent /
                                           result.structured_content

以及一件 1.x 时代不需要知道的事：**`list_resources()` 只列"静态资源"**，
模板形式的资源（URI 里带 {占位符} 的）要到 `list_resource_templates()` 里找。
书稿 §11.5.3 只调了 `list_resources()`，然后打印出 `book://meta/{book_path}`
—— 那是 1.x 的行为，2.x 里这个调用返回空列表。

================================================================
怎么跑
================================================================
    # 默认：stdio，自动把 server.py 当子进程拉起来（书稿的方式）
    uv run python ch11/cases/mcp_system/client.py

    # 换一本书稿（不给就用自带的 sample_book/）
    uv run python ch11/cases/mcp_system/client.py --book /path/to/your/book

    # 连一个已经在跑的 HTTP server（先在另一个终端起 server.py --transport streamable-http）
    uv run python ch11/cases/mcp_system/client.py --http http://127.0.0.1:8765/mcp

退出码：所有步骤都成功 → 0；任何一步失败 → 1（方便脚本判断）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server.py"

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  OK   {msg}")


def bad(step: str, exc: BaseException) -> None:
    failures.append(step)
    print(f"  FAIL {step}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 四步：list_tools / list_resources / call_tool / read_resource
# 函数名与书稿一致
# ---------------------------------------------------------------------------
async def list_tools(session: ClientSession) -> list[str]:
    result = await session.list_tools()
    print("\n=== 可用工具 ===")
    for tool in result.tools:
        first_line = (tool.description or "").split("\n")[0]
        print(f"  🔧 {tool.name}: {first_line}")
    return [t.name for t in result.tools]


async def list_resources(session: ClientSession) -> None:
    # 2.x 拆成两张表：静态资源 / 资源模板
    static = await session.list_resources()
    templates = await session.list_resource_templates()

    print("\n=== 可用资源（静态）===")
    if not static.resources:
        print("  （无。静态资源是 URI 完全固定的那种）")
    for res in static.resources:
        print(f"  📄 {res.name} → {res.uri}")

    print("\n=== 可用资源（模板）===")
    if not templates.resource_templates:
        print("  （无）")
    for tpl in templates.resource_templates:
        print(f"  📄 {tpl.name} → {tpl.uri_template}")


async def list_prompts(session: ClientSession) -> None:
    result = await session.list_prompts()
    print("\n=== 可用 Prompt ===")
    if not result.prompts:
        print("  （无）")
    for p in result.prompts:
        print(f"  📝 {p.name}: {p.description or ''}")
        if p.arguments:
            for a in p.arguments:
                print(f"       - {a.name} (required={a.required})")


async def call_tool(session: ClientSession, tool_name: str, arguments: dict) -> str | None:
    print(f"\n=== 调用 [{tool_name}] 入参: {json.dumps(arguments, ensure_ascii=False)} ===")
    try:
        result = await session.call_tool(tool_name, arguments)
    except Exception as exc:  # noqa: BLE001
        bad(f"call_tool({tool_name})", exc)
        return None

    if result.is_error:
        bad(f"call_tool({tool_name})", RuntimeError("服务端把这次调用标记为错误"))
        return None

    text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
    print(text)
    return text


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def run(session: ClientSession, book: str) -> None:
    # 1) 握手 + 能力协商
    init = await session.initialize()
    print("✅ MCP 连接建立成功")
    print(f"   服务器: {init.server_info.name}")
    print(f"   版本:   {init.server_info.version}")
    print(f"   协议:   {init.protocol_version}")
    caps = init.capabilities
    print(f"   能力:   {[k for k in ('tools', 'resources', 'prompts') if getattr(caps, k, None)]}")

    # 2) 发现
    tools = await list_tools(session)
    await list_resources(session)
    await list_prompts(session)

    # 3) 调用 —— 三个工具都打一遍
    await call_tool(session, "count_chars", {"book_path": book})
    await call_tool(session, "get_chapter_toc", {"book_path": book, "chapter_file": "ch01_为什么要手搓.md"})
    await call_tool(session, "search_book", {"keyword": "原语", "book_path": book})

    # 4) 读 Resource（逻辑名，不是绝对路径 —— 原因见 server.py 注释）
    print("\n=== 读取 Resource: book://meta/sample ===")
    try:
        resource = await session.read_resource("book://meta/sample")
        payload = json.loads(resource.contents[0].text)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        bad("read_resource(book://meta/sample)", exc)

    # 5) 取 Prompt
    print("\n=== 取 Prompt: review_chapter ===")
    try:
        prompt = await session.get_prompt("review_chapter", {"chapter_file": "ch01_为什么要手搓.md"})
        for msg in prompt.messages:
            content = msg.content
            text = getattr(content, "text", None) or str(content)
            print(f"[{msg.role}]\n{text}")
    except Exception as exc:  # noqa: BLE001
        bad("get_prompt(review_chapter)", exc)

    print(f"\n工具清单（供核对）: {tools}")

    # 6) 反向验证：故意调一个不存在的方法，确认错误路径也通
    print("\n=== 反向验证：调一个不存在的工具 ===")
    try:
        r = await session.call_tool("no_such_tool", {})
        if r.is_error:
            ok("服务端正确返回了 is_error=True（没有崩连接）")
            print(f"      返回内容: {[getattr(i, 'text', '')[:120] for i in r.content]}")
        else:
            print("  ⚠️ 未报错，这不符合预期")
    except Exception as exc:  # noqa: BLE001
        ok(f"服务端拒绝了未知工具（{type(exc).__name__}），连接未崩")


async def main() -> int:
    parser = argparse.ArgumentParser(description="书稿助手 MCP Client")
    parser.add_argument("--book", default="sample", help="书稿目录或已注册的逻辑名（默认 sample）")
    parser.add_argument("--http", default="", help="连已有的 Streamable HTTP server，如 http://127.0.0.1:8765/mcp")
    parser.add_argument("--python", default=sys.executable, help="stdio 模式下用哪个解释器起 server.py")
    args = parser.parse_args()

    if args.http:
        from mcp.client.streamable_http import streamable_http_client

        print(f"→ 传输: Streamable HTTP  {args.http}")
        async with streamable_http_client(args.http) as (read, write):
            async with ClientSession(read, write) as session:
                await run(session, args.book)
    else:
        params = StdioServerParameters(command=args.python, args=[str(SERVER)], env=None)
        print(f"→ 传输: stdio  子进程: {args.python} {SERVER.name}")
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await run(session, args.book)

    print()
    if failures:
        print(f"❌ 失败 {len(failures)} 步: {failures}")
        return 1
    print("✅ 全部步骤通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 运行出错: {type(exc).__name__}: {exc}")
        sys.exit(1)
