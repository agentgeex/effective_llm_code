#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：机制 2 分节生成：按节推进，每节写完更新记忆卡并落盘
# ② 对应书稿：ch11 §11.2，（机制 2）+（实践演示框架）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/novel_gen/section_writer.py --mock（真 LLM 去掉 --mock）
# ④ 依赖：openai（经 common/gpt/gptservice.py）；mock 模式零 key
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 --mock rc=0（6 节正文 + trace）
# ⑥ 与书稿的差异：补 summarize_section()（书稿手写摘要）；遍历提纲落盘；新增 memory_card_trace.json 与 --mock
# ══════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────
# 书稿出处：ch11 §11.2「长篇小说生成」（实践演示：一个可运行的框架）
# 本文件对应书稿的这几块：
#   · def write_section(memory, chapter, section, prompt_hint, temperature=0.8)  → 原样保留
#   · def consistency_check(memory, new_content)（temperature=0.3）              → 原样保留
#   · demo_workflow() 里「写一节 → memory.add_summary(...) → 再写下一节」的循环   → 扩成遍历整个提纲
# 相对书稿的改动：
#   1. 书稿每写完一节，是**人手写** add_summary("陈远发现父亲的旧木盒…")
#      → 6 节就得手写 6 条，没法脚本化
#      → 补 summarize_section()：真模式让模型按「发生了什么 + 埋了什么伏笔」压缩到 200 字内；
#        mock 模式直接从本节场景里抽一句话。★ 书里没这个函数，是本文件唯一新增的机制
#   2. 书稿 demo 只演示 2 节、章节号写死
#      → 改成读 outputs/outline.json 遍历全部章节，产出 outputs/ch01_s01.md 这类真实文件
#   3. 书稿把 section 内容 print 出来就完了
#      → 落盘：每节一个 .md（带元信息头）+ memory_card_trace.json（记录每节真实拿到的记忆卡前缀）
#        —— trace 是「锚定真的生效了」的证据，不是空口说
#   4. --mock：书稿没有。mock 走同一个 LLM 接口，且真的读 prompt 里的摘要与场景再产出正文
# 改动原则：只改到能跑，不改教学意图（记忆卡结构 / 5 节窗口 / 温度 0.8、0.3 全部照书）
# ─────────────────────────────────────────────────────────
"""
机制 2：分节生成（Section-Level Writing）

一句话：不要整章一次性生成；按 800~1500 字一节推进，每节结束后更新记忆卡，
        下一节拿到的前缀里就有上一节的摘要（这就是「金鱼记忆」的解法）。

跑法：
    uv run python ch11/cases/novel_gen/section_writer.py --mock          # 无 key，mock 全流程
    export DEEPSEEK_API_KEY=*** && uv run python ch11/cases/novel_gen/section_writer.py   # 真 LLM

前置：先跑 outline_anchor.py 生成 outputs/outline.json（本文件也支持 --all 一步跑完两个机制）
定级：B（mock 全流程已验证；真 LLM 需 key，本机无 key 未验证）
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:            # 让同级 outline_anchor.py 可被 import
    sys.path.insert(0, str(HERE))

from outline_anchor import (             # noqa: E402
    LLM, MemoryCard, RECENT_KEEP, TEMP_CHECK, TEMP_WRITE,
    build_memory, generate_outline, load_outline, save_outline,
)

OUTPUT_DIR = HERE / "outputs"
OUTLINE_PATH = OUTPUT_DIR / "outline.json"

WRITE_SYSTEM = (
    "你是一位经验丰富的小说作家。请根据大纲、角色设定和已有章节摘要，"
    "创作指定内容。保持人物性格和语言风格的一致，自然融入已有伏笔。"
    "单次产出控制在800~1500字。"
)
CHECK_SYSTEM = (
    "你是一位严格的文字编辑。请检查以下新生成的小说内容是否与角色设定"
    "和已有情节一致。重点关注：1）人物行为是否符合性格设定 "
    "2）对话风格是否一致 3）是否存在与已有情节矛盾之处。"
    "如有问题，逐条指出。如无问题，回复'一致性检查通过'。"
)
SUMMARY_SYSTEM = (
    "你是小说编辑助手。请把这一节压缩成不超过 200 字的摘要，"
    "只写两件事：发生了什么、埋了什么伏笔。不要评价文字好坏。"
)


def write_section(llm: LLM, memory: MemoryCard, chapter: int, section: int,
                  prompt_hint: str, temperature: float = TEMP_WRITE) -> str:
    """分节生成：用记忆卡作为上下文，撰写小说的一个小节（书稿同名函数）"""
    context = memory.build_prompt_context()
    return llm.chat(
        WRITE_SYSTEM,
        f"{context}\n\n【当前任务】撰写第{chapter}章第{section}节。{prompt_hint}",
        temperature,
    )


def consistency_check(llm: LLM, memory: MemoryCard, new_content: str) -> str:
    """校对 Agent：一致性检查（书稿同名函数，温度 0.3）"""
    context = memory.build_prompt_context()
    return llm.chat(
        CHECK_SYSTEM,
        f"{context}\n\n【待检查内容】\n{new_content}",
        TEMP_CHECK,
    )


def summarize_section(llm: LLM, body: str) -> str:
    """★ 书稿没有这个函数：书里由人手写 add_summary，这里把它自动化。

    产出格式对齐书稿示例：「陈远发现父亲的旧木盒，内有神秘图纸和一把钥匙。」
    """
    text = llm.chat(SUMMARY_SYSTEM, f"【本节正文】\n{body}", TEMP_CHECK).strip()
    text = text.replace("\n", " ")
    return text[:200]


def section_path(ch: int, sec: int) -> Path:
    return OUTPUT_DIR / f"ch{ch:02d}_s{sec:02d}.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="机制 2：分节生成")
    ap.add_argument("--mock", action="store_true", help="用假模型跑通数据流（无需 key）")
    ap.add_argument("--topic", default="父亲的旧木盒")
    ap.add_argument("--outline", default=str(OUTLINE_PATH))
    ap.add_argument("--all", action="store_true", help="提纲不存在时顺手先跑机制 1")
    args = ap.parse_args()

    llm = LLM(mock=args.mock)
    outline_path = Path(args.outline)
    if not outline_path.exists():
        if not args.all:
            raise SystemExit(f"缺少 {outline_path}。先跑 outline_anchor.py，或加 --all 一步生成。")
        print(f"[补跑机制1] {outline_path} 不存在，先按题目生成提纲")
        save_outline(generate_outline(llm, args.topic), outline_path)

    data = load_outline(outline_path)
    memory = build_memory(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[模式] {'mock（假模型，数据流真跑）' if args.mock else '真 LLM'}")
    print(f"[提纲] {data.get('title')}｜窗口 RECENT_KEEP={RECENT_KEEP}")

    trace = {"recent_keep": RECENT_KEEP, "sections": []}
    reports = []

    for ch in data["chapters"]:
        for sec in ch["sections"]:
            ch_no, sec_no = ch["no"], sec["no"]
            # 关键：生成前先记下「这一节到底拿到了哪些前文摘要」——锚定的证据
            included = [f"第{s['chapter']}章第{s['section']}节：{s['summary']}"
                        for s in memory.chapter_summaries[-RECENT_KEEP:]]
            context = memory.build_prompt_context()

            body = write_section(llm, memory, ch_no, sec_no, sec["goal"])
            check = consistency_check(llm, memory, body)
            summary = summarize_section(llm, body)
            memory.add_summary(ch_no, sec_no, summary)

            path = section_path(ch_no, sec_no)
            path.write_text(
                f"<!-- 第{ch_no}章第{sec_no}节｜{sec['title']}｜{len(body)} 字 -->\n\n{body}\n",
                encoding="utf-8",
            )

            trace["sections"].append({
                "id": f"ch{ch_no:02d}_s{sec_no:02d}",
                "chapter": ch_no,
                "section": sec_no,
                "title": sec["title"],
                "goal": sec["goal"],
                "props": sec.get("props", []),
                "context_chars": len(context),
                "prior_summaries_in_context": included,
                "body_chars": len(body),
                "summary": summary,
                "file": str(path.relative_to(HERE)),
            })
            reports.append((f"ch{ch_no:02d}_s{sec_no:02d}", check))

            print(f"  ✔ 第{ch_no}章第{sec_no}节 {sec['title']} → {path.name} "
                  f"（{len(body)} 字；前缀带 {len(included)} 条前文摘要）")

    (OUTPUT_DIR / "memory_card_trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = "\n\n".join(f"## {sid}\n\n{txt}" for sid, txt in reports)
    (OUTPUT_DIR / "consistency_report.md").write_text(report_md + "\n", encoding="utf-8")

    print(f"\n[产出] {len(reports)} 节正文 → {OUTPUT_DIR}/ch*.md")
    print(f"[产出] 记忆卡轨迹 → memory_card_trace.json")
    print(f"[产出] 一致性报告 → consistency_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
