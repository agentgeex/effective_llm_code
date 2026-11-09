#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：把 §13.3.4「LLM 只当指令调度员」跑一遍 —— 书稿版消息组装 vs 修正版，
#    并用本地校验器按 OpenAI tool_calls 契约判定消息序列是否合法
# ② 对应书稿：ch13 §13.3.4，（代码块全文；关键行）
# ③ 怎么跑：cd <仓库根> && python ch13/tool_dispatch_demo.py
#           DEEPSEEK_API_KEY=xxx .venv/bin/python ch13/tool_dispatch_demo.py --live
# ④ 依赖：离线 mock 仅标准库（json / math / argparse）；--live 需 openai SDK。无外部服务、无权重
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 rc=0（书稿版 3 处违规 / 修正版合法）；
#    .venv-core（3.10.12）、系统 python3（3.10.12）同为 rc=0；
#    --live 未跑（本机 DEEPSEEK_API_KEY / OPENAI_API_KEY 均不存在）
# ⑥ 与书稿的差异：书稿版把 messages.append(response.choices[0].message) 写在 for 循环内，
#    于是有 2 个 tool_call 时 assistant 会被插两次 → 消息序列非法；
#    本文件 = 修正版（assistant 只 append 一次），同时保留书稿版做对照。
#    单 tool_call 时两版序列等价 —— 这正是该 bug 能在书里躺很久的原因
# ══════════════════════════════════════════════════════════════════
"""
ch13/tool_dispatch_demo.py — 「LLM 只当指令调度员」：把算术交给代码，模型只决定调什么

对应书稿：ch13 §13.3.4「架构师的正解：LLM 作为"指令调度员"」

书稿那段代码的思路是对的：让模型输出 tool_calls，Python 用 eval 算，再把结果回灌。
但它的**消息组装**有一处会真炸的缺陷（下面 FIX 段有对照），本文件把两种组装都跑一遍：

    python3 ch13/tool_dispatch_demo.py            # 离线 mock（默认，不需要 key）
    DEEPSEEK_API_KEY=xxx python3 ch13/tool_dispatch_demo.py --live

书稿缺陷（P1，照抄就报错，但只在模型一次返回 ≥2 个 tool_call 时触发）：
  书里 `messages.append(response.choices[0].message)` 写在 `for tool_call in tool_calls` **循环体内**。
  模型返回 2 个工具调用时，消息序列变成：
      assistant(tool_calls=[A,B]) → tool(A) → assistant(tool_calls=[A,B]) → tool(B)
  第二个 assistant 后面只跟了 B，A 没人应答 —— OpenAI 侧报
      400 "An assistant message with 'tool_calls' must be followed by tool messages
           responding to each tool_call_id"
  （本机无 key，**这句 400 原文未在真实 API 上复现**；下面的校验器是按该规则实现的本地检查，
    证据等级 = 本地规则校验，不是 API 回执。）

实测解释器：.venv/bin/python / .venv-core/bin/python / /usr/bin/python3（均 3.10.12，实测）
            离线 mock 只用标准库；--live 另需 openai SDK（.venv 内已装）
"""

import argparse
import json
import math
import os
import sys

# ---- 示例 1：书稿原版（message 被重复 append） ----
# ============================================================ 书稿原样：工具实现

def calculate(expression: str) -> str:
    """执行数学表达式并返回结果。支持 +、-、*、/、**（幂）、sqrt、sin、cos 等。"""
    try:
        # 书稿注释写「使用 Python 内置 eval，但限制安全上下文」
        # ⚠️ 注意：清空 __builtins__ 挡得住 open()，挡不住属性链逃逸
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算。当你需要进行精确的算术运算时，使用这个函数而不是自己计算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

USER_QUERY = (
    "我们公司四个季度的营收分别是 847.32 万、1293.58 万、2146.07 万、1805.94 万，"
    "帮我算一下全年总营收和季度平均值。"
)

# ============================================================ 消息校验器（按 OpenAI 规则）

def validate_tool_messages(msgs):
    """检查消息序列是否满足 tool_calls 的应答要求。返回违规列表（空 = 合法）。"""
    bad = []
    seen_ids = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        ids = [tc["id"] for tc in (m.get("tool_calls") or [])]
        if ids:
            if any(x in seen_ids for x in ids):
                bad.append(f"第{i}条：assistant 的 tool_calls {ids} 重复出现（模型只返回了一次）")
            seen_ids.extend(ids)
            want, j = list(ids), i + 1
            while j < len(msgs) and msgs[j].get("role") == "tool":
                tid = msgs[j].get("tool_call_id")
                if tid in want:
                    want.remove(tid)
                else:
                    bad.append(f"第{j}条：tool 消息的 tool_call_id={tid} 不属于紧邻的 assistant")
                j += 1
            for miss in want:
                bad.append(f"第{i}条：tool_call {miss} 之后没有对应的 tool 消息应答")
            i = j
            continue
        i += 1
    return bad


# ============================================================ 书稿版组装

def build_book_style(tool_calls, results):
    """书里的组装方式（append 在循环体内）。"""
    messages = [{"role": "user", "content": USER_QUERY}]
    assistant = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    for tc, res in zip(tool_calls, results):
        if tc["function"]["name"] == "calculate":
            # ↓↓↓ 书里：这一行在 for 循环**里面**
            messages.append(assistant)
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": res}
            )
    return messages


# ============================================================ 修正版组装

def build_fixed(tool_calls, results):
    """修正：assistant 整条只 append 一次，再逐条 append tool 应答。"""
    messages = [{"role": "user", "content": USER_QUERY}]
    if not tool_calls:
        return messages, "模型没要求调工具 —— 此时书稿代码什么都不打印，本文件补了一条提示"
    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
    for tc, res in zip(tool_calls, results):
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
    return messages, None


# ============================================================ 离线 mock

def mock_tool_calls():
    """伪造模型的两个 tool_call：全年总营收 + 季度平均。"""
    return [
        {
            "id": "call_sum_0001",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": json.dumps({"expression": "847.32+1293.58+2146.07+1805.94"}),
            },
        },
        {
            "id": "call_avg_0002",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": json.dumps({"expression": "(847.32+1293.58+2146.07+1805.94)/4"}),
            },
        },
    ]


def dump(msgs):
    for i, m in enumerate(msgs):
        role = m.get("role")
        if m.get("tool_calls"):
            ids = ",".join(tc["id"] for tc in m["tool_calls"])
            print(f"  [{i}] {role:9s} tool_calls=[{ids}]")
        elif role == "tool":
            print(f"  [{i}] {role:9s} tool_call_id={m['tool_call_id']}  content={m['content'][:28]}")
        else:
            c = (m.get("content") or "")[:34].replace("\n", " ")
            print(f"  [{i}] {role:9s} {c}...")


def run_offline():
    tcs = mock_tool_calls()
    results = [
        calculate(json.loads(tc["function"]["arguments"])["expression"]) for tc in tcs
    ]

    print("=" * 78)
    print("离线 mock：模拟模型一次返回 2 个 tool_call（总营收 + 平均值）")
    print("=" * 78)
    print("\n模型要做的两件事：")
    for tc in tcs:
        print(f"  {tc['id']}  calculate({json.loads(tc['function']['arguments'])['expression']})")
    print("Python 真算出来的结果：")
    for r in results:
        print(f"  {r}")

    book = build_book_style(tcs, results)
    print("\n【书稿版】消息序列（append 在 for 循环内）：")
    dump(book)
    vb = validate_tool_messages(book)
    print(f"  校验：{'❌ 非法，%d 处违规' % len(vb) if vb else '✓ 合法'}")
    for v in vb:
        print(f"    - {v}")

    fixed, note = build_fixed(tcs, results)
    print("\n【修正版】消息序列（assistant 只 append 一次）：")
    dump(fixed)
    vf = validate_tool_messages(fixed)
    print(f"  校验：{'❌ 非法，%d 处违规' % len(vf) if vf else '✓ 合法'}")

    print("\n【单 tool_call 时】—— 书稿版也合法，这就是这个 bug 难被发现的原因：")
    one = tcs[:1]
    b1 = build_book_style(one, results[:1])
    v1 = validate_tool_messages(b1)
    print(f"  书稿版 校验：{'❌ ' + str(v1) if v1 else '✓ 合法（序列与修正版等价）'}")

    print("\n" + "=" * 78)
    print("结论（本机能确认的）：")
    print("  1. 1 个 tool_call → 书稿版合法；2 个 → 书稿版把 assistant 消息重复插入，")
    print("     第二个 assistant 后面的 tool 应答缺一个 id（校验器判定违规）。")
    print("  2. 修正版在 1 个和 2 个 tool_call 下都合法。")
    print("  3. 真实 API 的 400 原文未复现（无 key）；上面是本地按规则校验的结果。")
    print("  4. 算术本身与模型无关：6,092.91 与 1,523.2275 都是 Python 算的，模型只负责「调什么」。")
    print("=" * 78)
    return 0


def run_live(model):
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("✗ --live 需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY", file=sys.stderr)
        return 1
    from openai import OpenAI

    base = os.getenv("DEEPSEEK_BASE_URL") or (
        "https://api.deepseek.com/v1" if os.getenv("DEEPSEEK_API_KEY") else None
    )
    client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": USER_QUERY}],
        tools=TOOLS, tool_choice="auto",
    )
    m = r.choices[0].message
    tcs = [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in (m.tool_calls or [])
    ]
    print(f"模型返回 {len(tcs)} 个 tool_call：")
    for tc in tcs:
        print(f"  {tc['id']} {tc['function']['arguments']}")
    if not tcs:
        print("模型没调工具，直接回了：", (m.content or "")[:120])
        return 0
    results = [calculate(json.loads(tc["function"]["arguments"])["expression"]) for tc in tcs]
    book = build_book_style(tcs, results)
    vb = validate_tool_messages(book)
    print(f"\n书稿版序列校验：{'❌ %d 处违规' % len(vb) if vb else '✓ 合法'}")
    for v in vb:
        print("  -", v)
    fixed, _ = build_fixed(tcs, results)
    print(f"修正版序列校验：{'❌' if validate_tool_messages(fixed) else '✓ 合法'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="§13.3.4 指令调度员 demo")
    ap.add_argument("--live", action="store_true", help="真调模型（需 key）")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    if a.live:
        return run_live(a.model or ("deepseek-chat" if os.getenv("DEEPSEEK_API_KEY") else "gpt-4o"))
    return run_offline()


if __name__ == "__main__":
    sys.exit(main())
