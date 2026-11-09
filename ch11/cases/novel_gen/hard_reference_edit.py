#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：机制 3 硬引用修改：把目标段原文贴进 prompt，并列出引用了它的其它节
# ② 对应书稿：ch11 §11.2，（机制 3）+（实践演示框架）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/novel_gen/hard_reference_edit.py --mock --target ch01_s02
# ④ 依赖：jieba + openai（经 common/gpt/gptservice.py）；mock 模式零 key
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 --mock rc=0
# ⑥ 与书稿的差异：补 find_references()（书稿靠人眼找引用）；落盘对照与引用报告；新增 --mock
# ══════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────
# 书稿出处：ch11 §11.2「长篇小说生成」（实践演示：一个可运行的框架）
#           + 行 655-678（机制 3 的原理说明：硬引用 = 把待改内容直接贴进 prompt）
# 本文件对应书稿的这几块：
#   · def polish_text(original, style_instruction)（temperature=0.6）  → 原样保留
#   · demo_workflow() 第 5 步「润色 Agent：硬引用修改」里那个手写的 original_dialogue → 改成从产物里自动取
# 相对书稿的改动：
#   1. 书稿的「硬引用」是**人工**从第 2 节里挑一段对话贴进 prompt
#      → 节数一多就没人愿意手工找；而原文（行 660-676）自己说了「模型找会找错、会漏半截」
#      → 补 find_references()：先用词面重合度把「引用了这一节的其它节」找出来，
#        再自动截取目标节里**含对话的段落**硬贴进 prompt。找的目标仍然是「一段文本」，
#        贴的动作仍然是硬贴 —— 只是把「人眼找」换成「脚本找」，机制本身没变
#   2. 书稿只改一处、不告诉你改完还要连带改什么
#      → 补一份 reference_report：列出所有引用了目标节的其它节 + 命中的原句，
#        提示哪些需要同步修改（这是「硬引用」在长篇里真正有价值的地方）
#   3. 书稿 print 结果
#      → 落盘 outputs/edit_<节>.md + outputs/reference_report_<节>.json
#   4. --mock：书稿没有。mock 走同一个 LLM 接口，真的读【原文】再改写
# 改动原则：只改到能跑，不改教学意图
#
# 说明：书稿「关键技术要点 3」讲的**基线重置**（用上一轮产物当下轮输入）本轮未实现，
#       flag 在 README 的局限里，不假装做了。
# ─────────────────────────────────────────────────────────
"""
机制 3：硬引用修改（Hard-Reference Editing）

一句话：不要让模型去长文里「找」要改的段落 —— 你自己把那段贴进 prompt，
        模型只做「改」一件事。同时把引用了这段的其它节列出来，提示一起改。

跑法：
    uv run python ch11/cases/novel_gen/hard_reference_edit.py --mock --target ch02_s01
    uv run python ch11/cases/novel_gen/hard_reference_edit.py --mock --target ch02_s01 \
        --style "让老吴的反问更冷，陈远的回答更短"
    # 真 LLM：export DEEPSEEK_API_KEY=*** 后去掉 --mock

前置：先跑 section_writer.py（需要 outputs/ 里的正文与 memory_card_trace.json）
定级：B（mock 已验证；真 LLM 需 key，本机无 key 未验证）
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from outline_anchor import KNOWN_OBJECTS, LLM, TEMP_POLISH   # noqa: E402

OUTPUT_DIR = HERE / "outputs"
TRACE_PATH = OUTPUT_DIR / "memory_card_trace.json"

POLISH_SYSTEM = (
    "你是一位文字润色专家。请对以下原文进行润色，严格遵循风格要求。"
    "仅修改语言表达，不改动事实信息和情节走向。"
)

DEFAULT_STYLE = "保持老吴用反问句和引经据典的风格，让陈远的回应更冷淡但更有分量。"


def polish_text(llm: LLM, original: str, style_instruction: str) -> str:
    """硬引用修改：直接把待修改原文传入，结合风格指令进行润色（书稿同名函数，温度 0.6）"""
    return llm.chat(
        POLISH_SYSTEM,
        f"【风格要求】{style_instruction}\n\n【原文】\n{original}",
        TEMP_POLISH,
    )


# ───────────────── 书稿没写：把这节的引用关系找出来 ─────────────────
def load_trace() -> dict:
    if not TRACE_PATH.exists():
        raise SystemExit(f"缺少 {TRACE_PATH}。先跑 section_writer.py 生成正文。")
    return json.loads(TRACE_PATH.read_text(encoding="utf-8"))


def section_body(section: dict) -> str:
    path = HERE / section["file"]
    if not path.exists():
        raise SystemExit(f"缺少正文 {path}。先跑 section_writer.py。")
    text = path.read_text(encoding="utf-8")
    # 去掉第一行的 <!-- 元信息 -->
    return re.sub(r"^<!--.*?-->\n+", "", text, flags=re.S).strip()


def section_terms(section: dict, body: str) -> list:
    """这一节的「可检索特征」：提纲里标的道具 + 正文里的高频词（jieba 切）

    书稿没写这一步（书里靠人眼找）。本文件的做法是机械的、可解释的：
    道具名是人工标的「必现词」，jieba 高频词是补充，两者都再过一遍 df 过滤（见下）。
    jieba 没装就只用道具名，不报错。
    """
    terms = []
    for t in section.get("props", []) + KNOWN_OBJECTS:
        if t and t in body and t not in terms:
            terms.append(t)
    try:
        import jieba
        from collections import Counter
        words = [w for w in jieba.cut(body) if 2 <= len(w) <= 4 and not w.isdigit()]
        for w, n in Counter(words).most_common(10):
            if n >= 2 and w not in terms:
                terms.append(w)
    except ImportError:
        pass          # 没装 jieba 也能跑，只是特征词少一些
    return terms


def find_references(target: dict, sections: list) -> tuple:
    """找出引用了目标节的其它节，返回 (命中列表, 被丢掉的全局词)。

    判据（机械、可解释，不用模型）：
      1. 取目标节的道具名当特征词（如「铜钥匙」「符号」）；
      2. **先剔掉「全局词」**：在超过一半的节里都出现的词（例如贯穿全书的「旧木盒」），
         它们命中所有节，等于没信息量（这是 TF-IDF 里 df 太高的词要降权的那件事）；
      3. 剩下的一次词在别的节正文里出现 ≥1 次 → 判为引用，粒度落到「行」，
         报告能直接指出该改哪一句。
    """
    body = section_body(target)
    terms = section_terms(target, body)

    all_bodies = {sec["id"]: section_body(sec) for sec in sections}
    df = {t: sum(1 for b in all_bodies.values() if t in b) for t in terms}
    keep_limit = max(2, len(sections) // 2)
    common_terms = [t for t, n in df.items() if n > keep_limit]
    terms = [t for t in terms if t not in common_terms]

    hits = []
    for sec in sections:
        if sec["id"] == target["id"]:
            continue
        other = all_bodies[sec["id"]]
        matched = {}
        for term in terms:
            lines = [l.strip() for l in other.splitlines() if term in l]
            if lines:
                matched[term] = lines[:2]
        if matched:
            hits.append({
                "id": sec["id"],
                "title": sec["title"],
                "score": sum(len(v) for v in matched.values()),
                "matched_terms": sorted(matched),
                "sample_lines": matched,
            })
    hits.sort(key=lambda h: (-h["score"], h["id"]))
    return hits, common_terms


def extract_excerpt(body: str) -> str:
    """从这一节里截出「含对话的段落」，作为硬引用的原文。

    书稿是人工挑一段贴进 prompt（行 655-678）；这里把「挑」自动化，
    但贴的动作没变：仍然是**一段具体文本**贴进 prompt，不让模型去全文里搜。
    """
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    dialog = [p for p in paras if "“" in p or "”" in p]
    if dialog:
        return "\n\n".join(dialog)
    return "\n\n".join(paras[1:3]) if len(paras) > 2 else body


def main() -> int:
    ap = argparse.ArgumentParser(description="机制 3：硬引用修改")
    ap.add_argument("--mock", action="store_true", help="用假模型跑通数据流（无需 key）")
    ap.add_argument("--target", default="ch02_s01", help="要修改的节 id，如 ch02_s01")
    ap.add_argument("--style", default=DEFAULT_STYLE)
    args = ap.parse_args()

    trace = load_trace()
    sections = trace["sections"]
    target = next((s for s in sections if s["id"] == args.target), None)
    if target is None:
        ids = ", ".join(s["id"] for s in sections)
        raise SystemExit(f"没有 {args.target}。现有节：{ids}")

    body = section_body(target)
    excerpt = extract_excerpt(body)
    refs, common_terms = find_references(target, sections)

    print(f"[模式] {'mock（假模型，数据流真跑）' if args.mock else '真 LLM'}")
    print(f"[目标] {target['id']} {target['title']}（全文 {len(body)} 字）")
    print(f"[硬引用] 从全文里截出 {len(excerpt)} 字贴进 prompt（模型不用自己找）")
    if common_terms:
        print(f"[引用检查] 剔掉的全局词（过半的节都有，命中它们说明不了什么）："
              f"{'/'.join(common_terms)}")
    print(f"[引用检查] 另有 {len(refs)} 节引用了本节内容：")
    for r in refs:
        print(f"  · {r['id']} {r['title']}（命中 {r['score']} 处，涉及 "
              f"{'/'.join(r['matched_terms'])}）")
        for term, lines in r["sample_lines"].items():
            print(f"      - 「{term}」：{lines[0][:40]}…")
    if not refs:
        print("  （无其它节引用本节内容）")

    llm = LLM(mock=args.mock)
    polished = polish_text(llm, excerpt, args.style)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edit_path = OUTPUT_DIR / f"edit_{target['id']}.md"
    edit_path.write_text(
        f"<!-- 硬引用修改：{target['id']}｜风格：{args.style} -->\n\n"
        f"## 原文（硬贴进 prompt 的那一段）\n\n{excerpt}\n\n"
        f"## 修改后\n\n{polished}\n",
        encoding="utf-8",
    )
    report_path = OUTPUT_DIR / f"reference_report_{target['id']}.json"
    report_path.write_text(json.dumps({
        "target": target["id"],
        "title": target["title"],
        "style": args.style,
        "excerpt_chars": len(excerpt),
        "ignored_common_terms": common_terms,
        "referencing_sections": refs,
        "todo": [f"{r['id']} 需同步检查：{('/'.join(r['matched_terms']))}" for r in refs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[产出] {edit_path.name}")
    print(f"[产出] {report_path.name}")
    if refs:
        print(f"[待办] 以下节需要同步检查（改了本节，它们的引用可能过期）："
              f"{', '.join(r['id'] for r in refs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
