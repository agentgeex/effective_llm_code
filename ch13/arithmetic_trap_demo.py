#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：把 §13.3.1 的三组「精确算术」prompt 离线复算 —— 用 Decimal 给出精确和；
#    --live 时真调一次模型，把模型输出与精确值逐字比对
# ② 对应书稿：ch13 §13.3.1（三组算术 prompt：三季营收 / 四季营收 / 五个四位小数）
# ③ 怎么跑：cd <仓库根> && python ch13/arithmetic_trap_demo.py
#           DEEPSEEK_API_KEY=xxx .venv/bin/python ch13/arithmetic_trap_demo.py --live
#           OPENAI_API_KEY=xxx   .venv/bin/python ch13/arithmetic_trap_demo.py --live --model gpt-4o
# ④ 依赖：离线仅标准库（decimal / re / argparse）；--live 需 openai SDK。无外部服务、无权重
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 rc=0；
#    .venv-core（3.10.12）、系统 python3（3.10.12）同为 rc=0；
#    --live 未跑（本机 DEEPSEEK_API_KEY / OPENAI_API_KEY 均不存在）
# ⑥ 与书稿的差异：书稿是三次真实调 gpt-4o 的脚本；本文件默认离线复算（不依赖模型）。
#    示例 3 书稿写 36,925.04 = 精确值 36,925.0352 按「四舍五入到分」所得（ROUND_HALF_UP，差 +0.0048），
#    **不是截断**（截断是 36,925.03）。书稿称模型输出该值，本机无 key，该观测未验证
# ══════════════════════════════════════════════════════════════════
"""
ch13/arithmetic_trap_demo.py — 「确定性算术」陷阱的三组 prompt，用 Decimal 给出精确答案

对应书稿：ch13 §13.3.1「一个让老板崩溃的 demo」（三组 prompt）
  - 示例 1：三季营收 1,284.73 / 2,951.06 / 3,702.49
  - 示例 2：四季营收 847.32 / 1,293.58 / 2,146.07 / 1,805.94
  - 示例 3：五个四位小数 23,847.3291 / 1,293.5847 / 2,146.0732 / 1,805.9463 / 7,832.1019

本文件做两件事：
  1. 用 decimal.Decimal 算出三组的**精确答案**（书稿给的答案也在文件里，可对照）
  2. 有 key 时真的调一次模型，把模型输出与精确答案逐字比对

用法：
    python3 ch13/arithmetic_trap_demo.py              # 离线：只算精确答案（默认，不需要 key）
    python3 ch13/arithmetic_trap_demo.py --offline    # 同上，显式
    DEEPSEEK_API_KEY=xxx python3 ch13/arithmetic_trap_demo.py --live
    OPENAI_API_KEY=xxx   python3 ch13/arithmetic_trap_demo.py --live --model gpt-4o

实测解释器：.venv/bin/python / .venv-core/bin/python / /usr/bin/python3（均 3.10.12，实测）
            离线模式只用标准库 decimal；--live 另需 openai SDK（.venv 内已装）

⚠️ 边界（必须说清）：
  离线模式证明的是「精确答案是多少」，**不能**证明「模型会算错」。
  书稿说「在我实际测试中 gpt-4o 输出了 36,925.04」——那是作者的一次实测观测，
  本机没有 key，**这条未验证**。要复现必须自己带 key 跑 --live，且模型每次输出都可能不同。

数值口径（示例 3，必读）：
  精确和 = 36,925.0352。书稿写的 36,925.04 是「四舍五入到分（保留 2 位小数）」的结果：
      Decimal("36925.0352").quantize(Decimal("0.01"), ROUND_HALF_UP) = 36,925.04（差 +0.0048）
  截断（ROUND_DOWN）会得到 36,925.03，所以书稿那个值**不是截断**。
  书稿把它叫「末尾两位小数错了」，严格说这是「没按 prompt 的『不要四舍五入』保留 4 位」。
"""

import argparse
import os
import re
import sys
from decimal import Decimal

# ---------------------------------------------------------------- 书稿的三组数据

CASES = [
    # ---- 示例 1：三季营收 ----
    {
        "name": "§13.3.1 示例 1（三季营收）",
        "nums": ["1284.73", "2951.06", "3702.49"],
        "book_answer": "7,938.28",
        "book_comment": "书稿自己写「这次它算对了」",
        "result_key": "总和",
    },
    # ---- 示例 2：四季营收 ----
    {
        "name": "§13.3.1 示例 2（四季营收）",
        "nums": ["847.32", "1293.58", "2146.07", "1805.94"],
        "book_answer": "6,092.91",
        "book_comment": "书稿自己写「这次也对了」",
        "result_key": "总和",
    },
    # ---- 示例 3：五个四位小数 ----
    {
        "name": "§13.3.1 示例 3（五个四位小数）",
        "nums": ["23847.3291", "1293.5847", "2146.0732", "1805.9463", "7832.1019"],
        "book_answer": "36,925.04",
        "book_comment": "书稿称这是模型输出的**错值**，正确值 36,925.0352（末两位被四舍五入）",
        "result_key": "总和",
    },
]


def exact_sum(nums):
    """精确求和：全程 Decimal，不碰 float。"""
    total = Decimal("0")
    for n in nums:
        total += Decimal(n)
    return total


def fmt(d):
    """按书稿的千分位写法格式化，保留 Decimal 的原始位数。"""
    s = format(d, "f")
    if "." in s:
        ip, fp = s.split(".")
        fp = fp.rstrip("0") or "0"
        s = ip + "." + fp
    return f"{Decimal(s):,}"


def parse_builtin_answer(text):
    """从模型回复里抠出第一个像金额的数字（容忍千分位）。"""
    if not text:
        return None
    m = re.findall(r"-?\d[\d,]*\.\d+|-?\d[\d,]*", text)
    return m[0].replace(",", "") if m else None


def run_offline():
    print("=" * 78)
    print("离线模式：只用 Decimal 算精确答案（不调模型）")
    print("=" * 78)
    for c in CASES:
        d = exact_sum(c["nums"])
        book = c["book_answer"].replace(",", "")
        same = Decimal(book) == d
        print(f"\n{c['name']}")
        print(f"  数据  : {', '.join(c['nums'])}")
        print(f"  精确和: {fmt(d)}   （Decimal 原值 {d}）")
        print(f"  书稿写: {c['book_answer']}")
        print(f"  对照  : {'一致' if same else '不一致 —— 书稿写的不是精确值'}  ← {c['book_comment']}")
    print("\n" + "=" * 78)
    print("结论（只陈述本机能确认的）：")
    print("  1. 三组的精确答案已算出，可复算，不依赖任何模型。")
    print("  2. 示例 3 的精确和 36,925.0352 与书稿「正确答案」一致；书稿称模型输出 36,925.04，")
    print("     那是「四舍五入到分」的结果（差 +0.0048，不是截断；截断为 36,925.03）——")
    print("     本机无 key，**该观测未验证**。")
    print("  3. 前两组书稿自己就写了「算对了」，所以本节论点的证据链靠的是示例 3 那一次抽样。")
    print("     ⚠️ 这不是代码错误，是**论据强度**问题：同一模型同一 prompt 多次跑结果会变。")
    print("=" * 78)


def run_live(model):
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("✗ --live 需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。", file=sys.stderr)
        print("  没有 key 就跑离线模式：python3 ch13/arithmetic_trap_demo.py", file=sys.stderr)
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("✗ --live 需要 openai SDK（本机请用 .venv/bin/python 运行）", file=sys.stderr)
        return 1

    base = os.getenv("DEEPSEEK_BASE_URL") or (
        "https://api.deepseek.com/v1" if os.getenv("DEEPSEEK_API_KEY") else None
    )
    client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)

    print("=" * 78)
    print(f"在线模式：model={model}")
    print("=" * 78)
    bad = 0
    for c in CASES:
        d = exact_sum(c["nums"])
        prompt = (
            "请精确计算以下数字的总和（不要四舍五入）：\n"
            + "\n".join(f" {n}" for n in c["nums"])
        )
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            text = r.choices[0].message.content
        except Exception as e:  # 网络/额度/模型名，原样打印
            print(f"\n{c['name']}\n  ✗ 调用失败：{type(e).__name__}: {e}")
            bad += 1
            continue
        got = parse_builtin_answer(text)
        ok = got is not None and Decimal(got) == d
        bad += 0 if ok else 1
        print(f"\n{c['name']}")
        print(f"  精确值: {fmt(d)}")
        print(f"  模型说: {(text or '').strip()[:120]}")
        print(f"  判定  : {'✓ 对' if ok else '✗ 错/无法解析'}")
    print("\n" + "=" * 78)
    print(f"汇总：{len(CASES) - bad}/{len(CASES)} 次与精确值一致。单次结果不构成稳定性结论。")
    return 0 if bad == 0 else 2


def main():
    ap = argparse.ArgumentParser(description="§13.3.1 精确算术对照")
    ap.add_argument("--live", action="store_true", help="真调模型（需 key）")
    ap.add_argument("--offline", action="store_true", help="只算精确答案（默认）")
    ap.add_argument("--model", default=None, help="模型名，默认 deepseek-chat 或 gpt-4o")
    a = ap.parse_args()

    if a.live:
        model = a.model or ("deepseek-chat" if os.getenv("DEEPSEEK_API_KEY") else "gpt-4o")
        return run_live(model)
    run_offline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
