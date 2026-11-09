#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：“书稿助手” MCP Server：3 个工具 + 1 个 Resource + 1 个 Prompt
# ② 对应书稿：ch11 §11.5.2「手搓第一个 MCP Server」
# ③ 怎么跑：cd <仓库根> && python ch11/cases/mcp_system/server.py --transport stdio（也可由 client.py 拉起）
# ④ 依赖：mcp 2.3.0；无 key、无外部服务
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测（client 端到端 rc=0）
# ⑥ 与书稿的差异：书稿是 mcp 1.x（FastMCP / mcp.server.fastmcp）；本文件跑在 mcp 2.x（MCPServer / mcp.server.mcpserver），工具行为一致
# ══════════════════════════════════════════════════════════════════
"""
ch11/cases/mcp_system/server.py — "书稿助手" MCP Server

对应书稿：ch11 §11.5.2「手搓第一个 MCP Server」。

================================================================
⚠️ 版本差异（书稿写的是 mcp 1.x，本文件跑在 mcp 2.x 上）
================================================================
书里是这样开头：

    from mcp.server.fastmcp import FastMCP, Context
    mcp = FastMCP("BookTools Server")

mcp 2.x 把 `FastMCP` 改名为 `MCPServer`，并且 `mcp.server.fastmcp` 这个
导入路径被刻意做成不可用（会抛 ModuleNotFoundError，提示你去读迁移指南）。
所以本文件写成：

    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("BookTools Server")

对应关系：

    书稿（mcp 1.x）                       本文件（mcp 2.x）
    ------------------------------------  ------------------------------------
    from mcp.server.fastmcp import FastMCP  from mcp.server.mcpserver import MCPServer
    FastMCP("名字")                        MCPServer("名字", version="1.0.0")
    @mcp.tool()                            @mcp.tool()            （同名同形）
    @mcp.resource("book://meta/{p}")       @mcp.resource(...)     （同名同形）
    mcp.run()                              mcp.run(transport="stdio")
    mcp.run(transport="streamable-http",
            host=..., port=...)            mcp.run(transport="streamable-http",
                                                    host=..., port=...)

工具函数体、docstring 写法、Resource 返回字符串这三件事，1.x 与 2.x 完全一样，
**没改一行**。改的只是"从哪个模块拿那个类"。

**本文件没有把 mcp 降到 1.x** —— `pyproject.toml` 里 mcp 的版本约束一个字没动。
理由：为一个教学示例让所有人退回 1.x 不划算，而 2.x 有等价 API。

================================================================
三个工具里哪两个是书稿的、哪一个是本实现新增
================================================================
书稿 §11.5.2 实际只定义了 **2 个工具 + 1 个 Resource**：
    count_chars       （书里同名工具）
    get_chapter_toc   （书里同名工具）
    book://meta/{book_path}  （Resource，书里同名资源）

本实现多出一个：
    search_book       （新增。书里没有。加它是因为本节演示要凑够 3 个工具，
                       而书稿那句「已经覆盖了 MCP Server 的全部三种原语」
                       其实只说对了一半——见下）

另外补了 1 个 Prompt：
    review_chapter    （新增。书稿正文大讲 Tools / Resources / Prompts 三种原语，
                       但 §11.5.2 的代码只有 Tools 和 Resources 两种，
                       没有 Prompt。补上之后这句话才成立。）

================================================================
怎么跑
================================================================
    # stdio（默认，给 client.py 当子进程用；单独跑会一直等 stdin）
    uv run python ch11/cases/mcp_system/server.py

    # Streamable HTTP（独立服务，client.py --http 连它）
    uv run python ch11/cases/mcp_system/server.py --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

HERE = Path(__file__).resolve().parent
DEFAULT_BOOK = "sample"


# ---------------------------------------------------------------------------
# 辅助：逻辑名 → 目录
# ---------------------------------------------------------------------------
# 书稿里工具直接收绝对路径（作者本机 /path/to/your/...），读者没有那份书稿，
# 原样跑必然 FileNotFoundError。所以这里允许两种传法：
#   1) 逻辑名  "sample"  → 本目录下的 sample_book/（默认值，开箱即跑）
#   2) 真路径  "/any/dir" 或 "./my_book" → 原样使用（书稿的调用方式仍然成立）
# 额外挂载：环境变量 MCP_BOOK_ROOTS="名字=/绝对路径,名字2=/另一个路径"
def registered_roots() -> dict[str, Path]:
    roots: dict[str, Path] = {"sample": HERE / "sample_book"}
    for item in os.environ.get("MCP_BOOK_ROOTS", "").split(","):
        if "=" not in item:
            continue
        name, _, raw = item.partition("=")
        if name.strip() and raw.strip():
            roots[name.strip()] = Path(raw.strip()).expanduser()
    return roots


def resolve_book(book_path: str) -> Path:
    if not book_path:
        book_path = DEFAULT_BOOK
    roots = registered_roots()
    if book_path in roots:
        return roots[book_path].resolve()
    return Path(book_path).expanduser().resolve()


def log(msg: str) -> None:
    """日志一律走 stderr。

    stdio 传输下 stdout 是协议通道，往里写一个字节就会破坏 JSON-RPC 帧。
    """
    print(f"[server] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 创建 Server 实例
# ---------------------------------------------------------------------------
mcp = MCPServer(
    "BookTools Server",
    version="1.0.0",
    instructions="书稿助手：统计字符数、读章节结构、全文检索。全部本地文件操作，不需要任何密钥。",
)


# ---------------------------------------------------------------------------
# Tool 1（书稿原有）：统计字符数
# ---------------------------------------------------------------------------
@mcp.tool()
def count_chars(book_path: str = DEFAULT_BOOK) -> str:
    """统计指定书稿根目录下所有 .md 文件的字符数。

    Args:
        book_path: 书稿根目录的绝对路径；也可以传已注册的逻辑名（默认 "sample"）

    Returns:
        Markdown 表格形式的各文件字符数明细及总计
    """
    root = resolve_book(book_path)
    if not root.exists():
        return f"错误：路径 '{root}' 不存在"

    md_files = sorted(root.rglob("*.md"))
    if not md_files:
        return f"在 '{root}' 下没找到任何 .md 文件"

    total = 0
    lines = [f"# 书稿字符数统计: {root.name}\n", "| 文件 | 字符数 |", "|------|------|"]
    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8")
            # 书稿原文的简化计数：去掉空格和换行后按 Unicode 字符数统计。
            # 注意：不区分中英文，英文单词会被按字母拆开计数。
            count = len(text.replace(" ", "").replace("\n", ""))
            lines.append(f"| {f.relative_to(root)} | {count} |")
            total += count
        except Exception as exc:  # noqa: BLE001 - 单个文件读失败不该拖垮整次调用
            lines.append(f"| {f.relative_to(root)} | (读取失败: {exc}) |")
    lines += ["", f"**总计: {total} 字符**"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2（书稿原有）：读章节结构
# ---------------------------------------------------------------------------
@mcp.tool()
def get_chapter_toc(book_path: str, chapter_file: str) -> str:
    """获取指定章节文件的目录结构（解析 H2 / H3 标题）。

    Args:
        book_path: 书稿根目录；也可以传已注册的逻辑名（如 "sample"）
        chapter_file: 章节文件名，如 'ch01_为什么要手搓.md'

    Returns:
        JSON 数组，每项含 level 与 title
    """
    file_path = resolve_book(book_path) / chapter_file
    if not file_path.exists():
        return json.dumps({"error": f"文件 '{chapter_file}' 不存在"}, ensure_ascii=False)

    structure = []
    for line in file_path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## "):
            structure.append({"level": 2, "title": line[3:].strip()})
        elif line.startswith("### "):
            structure.append({"level": 3, "title": line[4:].strip()})
    return json.dumps(structure, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool 3（本实现新增，书稿没有）：全文检索
# ---------------------------------------------------------------------------
@mcp.tool()
def search_book(keyword: str, book_path: str = DEFAULT_BOOK, limit: int = 20) -> str:
    """在书稿全部 .md 文件里做逐行全文检索。

    纯本地字符串匹配，不调任何外部服务，不需要密钥。

    Args:
        keyword: 要搜的关键词
        book_path: 书稿根目录，或已注册的逻辑名（默认 "sample"）
        limit: 最多返回多少条命中（默认 20）

    Returns:
        每条命中一行：相对路径:行号: 该行内容（JSON 字符串）
    """
    if not keyword:
        return json.dumps({"error": "keyword 不能为空"}, ensure_ascii=False)

    root = resolve_book(book_path)
    if not root.is_dir():
        return json.dumps({"error": f"目录不存在: {root}"}, ensure_ascii=False)

    hits = []
    for f in sorted(root.rglob("*.md")):
        try:
            for lineno, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
                if keyword in line:
                    hits.append({
                        "file": str(f.relative_to(root)),
                        "line": lineno,
                        "text": line.strip()[:160],
                    })
                    if len(hits) >= max(1, limit):
                        break
        except Exception:  # noqa: BLE001
            continue
        if len(hits) >= max(1, limit):
            break

    return json.dumps(
        {"keyword": keyword, "count": len(hits), "hits": hits},
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Resource（书稿原有）：书稿元信息
# ---------------------------------------------------------------------------
# 书稿写的是 @mcp.resource("book://meta/{book_path}")，client 端读的时候把
# 绝对路径拼进去：book://meta//path/to/your/...
#
# 这个写法在 mcp 2.x 默认配置下读不出来，两个原因叠加：
#   ① 资源模板里的 {var} 不匹配 "/"，绝对路径拼进去后整个 URI 匹配不上；
#   ② 2.x 新增了 ResourceSecurity 门禁，默认 reject_absolute_paths=True，
#      带绝对路径的资源 URI 直接被拒。
# 实测：改成 {+book_path}（保留展开）+ 关掉这两道门，书稿的 URI 能读通
# （见 README「与书稿的差异」第 4 条）。但那样等于教读者关掉一道安全门，
# 而且书稿自己那句注释也说了「绝对路径仅为方便演示，生产环境应使用相对标识符」。
#
# 所以这里改成逻辑名：book://meta/{book_name}，默认 sample。
@mcp.resource("book://meta/{book_name}", name="book_meta", description="书稿元信息：根目录、文件数、文件清单")
def book_meta(book_name: str = DEFAULT_BOOK) -> str:
    """把书稿的元信息作为一个 Resource 暴露出来。

    Resource 是客户端可以订阅的数据源，不需要模型主动调用即可注入上下文。
    """
    root = resolve_book(book_name)
    if not root.is_dir():
        return json.dumps(
            {"error": f"未知的书稿名或路径: {book_name}", "known": sorted(registered_roots())},
            ensure_ascii=False,
        )
    md_files = sorted(root.rglob("*.md"))
    return json.dumps(
        {
            "name": book_name,
            "root": str(root),
            "total_files": len(md_files),
            "files": [str(f.relative_to(root)) for f in md_files],
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Prompt（本实现新增）：审一章的提纲
# ---------------------------------------------------------------------------
@mcp.prompt(name="review_chapter", description="生成一份「审这一章」的提示模板")
def review_chapter(chapter_file: str = "") -> str:
    """Prompt 是第三种原语，控制方是用户。

    书稿 §11.5.1 讲了三原语，但 §11.5.2 的代码只实现了 Tools 与 Resources。
    这里补一个最朴素的 Prompt：用户点一下，就得到一段可用的审稿指令。
    """
    target = chapter_file or "（未指定，请先调用 get_chapter_toc 看有哪些章）"
    return (
        f"请审阅这一章：{target}\n\n"
        "审查顺序（三层递进，别混着看）：\n"
        "1. 结构：论证链是否完整？段落职责是否重复？有无跑题？\n"
        "2. 事实：数据与引用是否可核验？术语前后是否一致？\n"
        "3. 语言：标点体例、用词统一、重复表述。\n\n"
        "每条问题请给出：位置（标题或引用原句）+ 问题类型 + 改写建议。"
    )


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="书稿助手 MCP Server（mcp 2.x）")
    # 书里给了 --transport / --host / --port 三个参数，
    # 但书稿的代码里没有 argparse —— 那两个命令其实不起作用。
    # 这里把参数真的接上（见 README 差异第 3 条）。
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="传输方式，默认 stdio（由 client 作为子进程启动）",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    log(f"transport = {args.transport}")
    log(f"已注册书稿: {', '.join(f'{k} -> {v}' for k, v in registered_roots().items())}")
    log("工具: count_chars / get_chapter_toc / search_book")
    log("Resource: book://meta/{book_name}   Prompt: review_chapter")

    if args.transport == "stdio":
        mcp.run()  # stdout 从此归协议所有，日志必须走 stderr
    elif args.transport == "sse":
        log(f"SSE 监听 http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        log(f"Streamable HTTP 监听 http://{args.host}:{args.port}/mcp")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
