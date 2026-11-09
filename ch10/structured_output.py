# -*- coding: utf-8 -*-
"""结构化输出：JSON 解析兜底 + 提示词约束模式的操作分发

① 这是什么
   书稿「大模型输出数据处理」的两个可运行示例：
   1) parse_model_output —— 模型输出不是合法 JSON 时的兜底解析（书稿原样 + 修复版对照）；
   2) handle_model_output —— 按模型输出末尾的 action 分发到 send_email / query_database / call_api。
   同时把书稿里未定义的三个执行函数补成假实现，让分发逻辑真的跑出结果。

② 对应书稿：第 10 章
   - §10.3.4 大模型输出数据的处理，书稿对应小节（parse_model_output）
   - §10.4.2.2 提示词约束，书稿对应小节（提示词里约定的 JSON 输出格式，纯文本，本文件仅引用）
   - §10.4.2.2 提示词约束，书稿对应小节（handle_model_output）

③ 怎么跑（仓库根目录执行）
   .venv/bin/python ch10/structured_output.py

④ 依赖
   - 包：零（标准库 json / re）
   - 服务：无。不需要 API key

⑤ 实测解释器
   python（Python 3.10.12）
   实测 rc=0，输出（原样）：
       [1] 纯 JSON                   → OK   {'action': 'call_api', 'params': {'url': '/ping'}}
       [2] ```json 围栏               → 💥 书稿版解析失败：模型输出无法解析为 JSON，触发重试
       [2'] 同一输入走修复版          → OK   {'action': 'call_api'}
       [3] 尾随逗号                     → 💥 书稿版解析失败：模型输出无法解析为 JSON，触发重试
       [3'] 同一输入走修复版          → OK   {'action': 'call_api'}
       [4] 分发：send_email  → 邮件发送结果：已发送给 张三
       [5] 分发：query_database → 查询结果：3 行
       [6] 无操作指令，原样返回文本 → 这段话里没有 JSON，直接当回复

⑥ 与书稿的差异（四处）
   1. 书稿对应小节 用了 json 却没 `import json`，且 OutputFormatError 全书从未定义
      → 照抄必 NameError。本文件补 `import json` 并定义最小异常类。
   2. 书稿注释说「去除尾随逗号」，实现只去围栏 → 尾随逗号仍然解析失败。
      本文件保留书稿原函数（含该缺陷）用于对照，另给 `parse_model_output_fixed`：
      围栏按 ``` 切 + 正则去 `,}` / `,]` + 找不到 JSON 时退回 {} 片段。
   3. 同一章 handle_model_output 用 `split("```json")[-1]`，能处理前置解说；
      与 parse_model_output 的 `removeprefix` 写法不一致。本文件两者都原样保留，对照可见。
   4. handle_model_output 里 send_email / query_database / call_api 书稿未定义（示意），
      本文件补三个假实现，使分发逻辑可实跑；分发分支本身与书稿一致。
"""

import json
import re

# ---- 示例 1：parse_model_output，原样照抄，含书稿原文的两处缺陷 ----
class OutputFormatError(Exception):
    """书稿引用了这个异常但没定义，这里补上最小定义。"""


def parse_model_output(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 常见修复：去除代码块标记、尾随逗号   ← 注释这么说，但下面只去了围栏
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise OutputFormatError("模型输出无法解析为 JSON，触发重试")


# ---- 补: 修复版（建议改法，非书稿原文） ----
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def parse_model_output_fixed(raw: str) -> dict:
    """真的能修：① 去掉围栏（不管前面有没有解说文字）② 去掉尾随逗号"""
    cleaned = raw.strip()
    # ① 有围栏就取围栏里的内容；没有就取第一个 { 到最后一个 } 之间的片段
    if "```" in cleaned:
        parts = cleaned.split("```")
        # 形如 ["解说", "json\n{...}", ""] 或 ["", "json\n{...}", ""]
        body = max(parts, key=len)
        body = body.split("\n", 1)[1] if "\n" in body else body
    else:
        i, j = cleaned.find("{"), cleaned.rfind("}")
        if i == -1 or j <= i:
            raise OutputFormatError("模型输出里找不到 JSON 片段，触发重试")
        body = cleaned[i:j + 1]
    # ② 尾随逗号
    body = _TRAILING_COMMA.sub(r"\1", body).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise OutputFormatError(f"模型输出无法解析为 JSON，触发重试：{e}")


# ---- 补: 书稿对应小节 引用了但未定义的三个执行函数（假实现） ----
def send_email(to: str = "", **kw) -> str:
    return f"已发送给 {to or '（未指定）'}"


def query_database(sql: str = "") -> str:
    return f"{3 if 'orders' in sql else 0} 行"


def call_api(url: str = "", **kw) -> str:
    return f"GET {url} → 200"


# ---- 示例 2：handle_model_output，原样照抄 ----
def handle_model_output(raw_output: str) -> str:
    """解析模型输出，提取操作指令并执行"""
    try:
        if "```json" in raw_output:
            json_str = raw_output.split("```json")[-1].split("```")[0]
            action = json.loads(json_str)
        else:
            lines = raw_output.strip().split("\n")
            action = json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        return raw_output

    if action.get("action") == "send_email":
        result = send_email(**action.get("params", {}))
        return f"邮件发送结果：{result}"
    elif action.get("action") == "query_database":
        result = query_database(action.get("params", {}).get("sql"))
        return f"查询结果：{result}"
    elif action.get("action") == "call_api":
        result = call_api(**action.get("params", {}))
        return f"API 调用结果：{result}"

    return raw_output


def main() -> None:
    cases = [
        ("纯 JSON", '{"action": "call_api", "params": {"url": "/ping"}}'),
        ("```json 围栏", '好的，我来调用。\n```json\n{"action": "call_api"}\n```'),
        ("尾随逗号", '{"action": "call_api",}'),
    ]
    n = 0
    for name, raw in cases:
        n += 1
        try:
            print(f"[{n}] {name:24s} → OK   {parse_model_output(raw)}")
        except OutputFormatError as e:
            print(f"[{n}] {name:24s} → 💥 书稿版解析失败：{e}")
            print("    （书稿注释说会「去除尾随逗号」，但代码只去掉了围栏 → 注释与实现不符，见「书稿问题」）")
            print(f"[{n}'] 同一输入走修复版{'':<9s} → OK   {parse_model_output_fixed(raw)}")

    out4 = handle_model_output(
        '```json\n{"action": "send_email", "params": {"to": "张三"}}\n```')
    out5 = handle_model_output(
        '{"action": "query_database", "params": {"sql": "select * from orders"}}')
    out6 = handle_model_output("这段话里没有 JSON，直接当回复")
    print(f"\n[4] 分发：send_email  → {out4}")
    print(f"[5] 分发：query_database → {out5}")
    print(f"[6] 无操作指令，原样返回文本 → {out6}")


if __name__ == "__main__":
    main()
