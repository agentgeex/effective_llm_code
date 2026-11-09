# -*- coding: utf-8 -*-
"""① 这是什么
    ch06 §6.7 DSPy 的最小可跑版：**不写 Prompt，只声明输入输出**，让编译器去挑示例与措辞。
    把书稿那张「Signature → Module → optimizer.compile()」的三段式落成能真跑的东西，
    并补上书稿只说没做的两件事：
      · 编译产物**可序列化**（.save / .load）—— 书稿「Prompt 成为可版本管理的资产」
      · 换优化器/换模型时**只重编译，不改业务代码**

② 对应书稿
    第 6 章 §6.7.1「DSPy：当编译器开始替你写 Prompt」。
    这一节的三段式是：`Signature` 声明输入输出 → `Module`（Predict / ChainOfThought）
    → `optimizer.compile()` 自动挑示例。本文件把这三段落成能真跑的东西，
    并补上书稿只说没做的两件事：
      · 编译产物**可序列化**（.save / .load）—— Prompt 因此成为可版本管理的资产
      · 换优化器 / 换模型时**只重编译，不改业务代码**

③ 怎么跑
    # ① 默认：离线桩 LM（不联网、不要 key），整条 DSPy 链路真跑
    .venv/bin/python ch06/dspy_min.py

    # ② 有 key 时跑真模型（本机当前**没有** key，这条路未验证）
    export DEEPSEEK_API_KEY=<你的key>
    .venv/bin/python ch06/dspy_min.py --live

    # ③ 只看编译器挑中了哪几个示例
    .venv/bin/python ch06/dspy_min.py --show-demos

④ 依赖
    包：dspy 2.6.13（本机 .venv 里有；**注意 .venv 没装 torch，也不需要**）
    服务：默认无（离线桩）；`--live` 需要 OpenAI 兼容端点
    权重：无
    落盘：./.dspy_demo/compiled_judge.json（编译产物，已进 .gitignore）

⑤ 实测解释器 + 结果（sgaiedge17）
    解释器：python（Python 3.10.12 / dspy 2.6.13）
    `python ch06/dspy_min.py` → **rc=0**，当次 stdout 原文（未改写）：
        [LM] 离线桩 DummyLM（不是真模型）：链路真跑，**效果不代表任何模型能力**

        [数据] 训练集 5 条 / 验证集 4 条（书稿只给了 1 条示例，其余是本文件补的）
        [1] Predict        → verdict=可退货 | reason=质量问题且在7天内
            签名声明的输出字段: ['verdict', 'reason']
        [2] ChainOfThought → verdict=可退货 | reason=质量问题且在7天内
            签名声明的输出字段: ['reasoning', 'verdict', 'reason']  ← 比 Predict 多 reasoning，代码里一个字没改
        Bootstrapped 2 full traces after 2 examples for up to 1 rounds, amounting to 2 attempts.
        [3] compile        → 编译产物 demos=2 条（从 5 条训练集里自动挑的）
        [4] save/load      → compiled_judge.json 1797 字节；重载后 demos=2 条，内容一致
        [5] devset         → 4/4 命中（★ 桩 LM 的固定答案，**不代表优化效果**）

    `--live`（无 key）→ **rc=2**，打印：
        ❌ --live 需要 DEEPSEEK_API_KEY / OPENAI_API_KEY；本机当前没有。
    `--show-demos` → 挑中的 2 条：verdict=可退货 / 可退货（**上限 2 是 max_bootstrapped_demos 设的**，
    不是「只有 2 条能过 metric」；桩 LM 的答案固定，所以这个分布没有统计意义）。

⑥ 与书稿的差异（三处，都是读代码时会踩到的）
    ① `accuracy_metric`：书里只提到要用它，没给实现。本文件自带一份最小 metric
       （判 `verdict` 字段是否一致），这样不绑死书稿的具体写法。
    ② `dspy.configure(lm=...)`：书里**有**写，在 §6.7.1 的示例里就有
       `dspy.LM('openai/gpt-4o-mini')` + `dspy.configure(lm=lm)` 这两行。
       别因为翻得快而以为漏了 —— 没有它，编译和调用都会报「没配 LM」。
    ③ `ChainOfThought` 比 `Predict` **多一个 `reasoning` 输出字段**：
       这也是 §6.7.1 那个坑的来源 —— 你按 `['verdict','reason']` 断言，
       实际拿到的是 `['reasoning','verdict','reason']`，于是
       `ValueError: Expected dict_keys(['verdict','reason']) but got dict_keys([...])`。
       改断言字段即可，不用改模型。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

try:
    import dspy
    from dspy.utils import DummyLM
except ImportError:                                  # pragma: no cover
    print("❌ 缺 dspy。装：uv pip install --python <解释器> dspy", file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / ".dspy_demo"

# ─────────────────────── 书稿原样：Signature ───────────────────────
class ReturnJudge(dspy.Signature):
    """判断用户的退货申请是否合理。"""
    order_info = dspy.InputField()        # 订单信息
    return_request = dspy.InputField()    # 用户退货请求
    verdict = dspy.OutputField()          # 判定结果（可退货/不可退货）
    reason = dspy.OutputField()           # 判定理由


# ─────────────────── 书稿补白：metric（书稿漏了） ───────────────────
def accuracy_metric(gold, pred, trace=None) -> bool:
    """书稿对应小节 引用了 accuracy_metric 却没定义。这里给最小实现：判 verdict 是否一致。

    注意：DSPy 的 metric 签名是 (gold, pred, trace)；trace 在 bootstrap 阶段非 None，
    可以用它来筛「模型走通了推理链」的样本。本脚本只比最终字段，够用。
    """
    try:
        return str(gold.verdict).strip() == str(pred.verdict).strip()
    except AttributeError:
        return False


# ───────────────────── 数据：书稿只给 1 条，这里补到 5 + 4 ─────────────────────
def build_trainset():
    rows = [
        dict(order_info="订单#N001，商品AX200耳机，已签收3天",
             return_request="耳机有杂音，要求退货",
             verdict="可退货", reason="商品存在质量问题，在7天无理由退货期内"),
        dict(order_info="订单#N002，商品BX500键盘，已签收1天",
             return_request="不想要了，申请退货",
             verdict="可退货", reason="七天无理由退货期内"),
        dict(order_info="订单#N003，商品CZ900鼠标，已签收40天",
             return_request="用不惯，想退",
             verdict="不可退货", reason="已超出7天无理由退货期"),
        dict(order_info="订单#N004，商品DX100音箱，已签收2天，已拆封使用",
             return_request="音质不行，退货",
             verdict="可退货", reason="音质问题属质量问题，拆封不影响质量问题的退货"),
        dict(order_info="订单#N005，商品EX300屏幕，已签收5天，人为磕碰",
             return_request="屏幕碎了，退货",
             verdict="不可退货", reason="人为损坏不在质量保修范围"),
    ]
    return [dspy.Example(**r).with_inputs("order_info", "return_request") for r in rows]


def build_devset():
    rows = [
        dict(order_info="订单#D001，AX200耳机，已签收1天", return_request="杂音，退货",
             verdict="可退货", reason="质量问题且在期内"),
        dict(order_info="订单#D002，BX500键盘，已签收3天", return_request="不喜欢，退货",
             verdict="可退货", reason="七天无理由"),
        dict(order_info="订单#D003，CZ900鼠标，已签收60天", return_request="坏了，退货",
             verdict="不可退货", reason="超出退换期"),
        dict(order_info="订单#D004，EX300屏幕，已签收2天，摔过", return_request="碎了，退货",
             verdict="不可退货", reason="人为损坏"),
    ]
    return [dspy.Example(**r).with_inputs("order_info", "return_request") for r in rows]


# ───────────────────── LM：离线桩 / 真模型 ─────────────────────
STUB_ANSWERS = {
    # DummyLM 模式 2：按「最后一条 message 里出现的子串」匹配答案。
    # ★ ChainOfThought 会自动多要一个 reasoning 字段 —— 不给就报字段不匹配（实测踩过）。
    "N001": {"reasoning": "签收3天在7天内，且反馈质量问题",
             "verdict": "可退货", "reason": "质量问题且在7天内"},
    "N002": {"reasoning": "签收1天，七天无理由", "verdict": "可退货", "reason": "七天无理由退货期内"},
    "N003": {"reasoning": "签收40天，远超7天", "verdict": "不可退货", "reason": "已超出7天无理由退货期"},
    "N004": {"reasoning": "音质问题属质量问题", "verdict": "可退货", "reason": "质量问题，拆封不影响退货"},
    "N005": {"reasoning": "人为磕碰不在保修范围", "verdict": "不可退货", "reason": "人为损坏不在质量保修范围"},
    "D001": {"reasoning": "签收1天，杂音属质量问题",
             "verdict": "可退货", "reason": "质量问题且在期内"},
    "D002": {"reasoning": "签收3天，七天无理由",
             "verdict": "可退货", "reason": "七天无理由"},
    "D003": {"reasoning": "签收60天，远超7天", "verdict": "不可退货", "reason": "超出退换期"},
    "D004": {"reasoning": "人为摔坏，不在保修范围",
             "verdict": "不可退货", "reason": "人为损坏"},
}


def configure_lm(live: bool) -> str:
    if live:
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        model = os.environ.get("DSPY_MODEL", "openai/deepseek-chat")
        if not key:
            print("❌ --live 需要 DEEPSEEK_API_KEY / OPENAI_API_KEY；本机当前没有。", file=sys.stderr)
            print("   去掉 --live 走离线桩（链路一样，只是模型是假的）。", file=sys.stderr)
            return ""
        lm = dspy.LM(model, api_key=key, api_base=base, max_tokens=256, temperature=0)
        dspy.settings.configure(lm=lm)
        return f"真模型 {model} @ {base}（**本机未验证**，本次没有 key）"
    dspy.settings.configure(lm=DummyLM(STUB_ANSWERS))
    return "离线桩 DummyLM（不是真模型）：链路真跑，**效果不代表任何模型能力**"


def main() -> int:
    ap = argparse.ArgumentParser(description="DSPy 最小可跑示例（默认离线桩）")
    ap.add_argument("--live", action="store_true", help="用真模型（需 API key）")
    ap.add_argument("--show-demos", action="store_true", help="打印编译器挑中的 few-shot 示例")
    ap.add_argument("--keep", action="store_true", help="保留 .dspy_demo/（默认跑完删掉）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（BootstrapFewShot 会抽样）")
    args = ap.parse_args()

    # ★ 固定种子：BootstrapFewShot 会随机抽训练样本，不固定则「挑中哪几条 demo」每次可能不同，
    #   注释里写的实测输出就对不上了。实测 3 次原本也一致（训练集只有 5 条），但固定住更稳。
    random.seed(args.seed)

    lm_desc = configure_lm(args.live)
    if not lm_desc:
        return 2
    print(f"[LM] {lm_desc}\n")

    trainset, devset = build_trainset(), build_devset()
    print(f"[数据] 训练集 {len(trainset)} 条 / 验证集 {len(devset)} 条"
          f"（书稿只给了 1 条示例，其余是本文件补的）")

    # ── ① Predict：最朴素的模块，一个字段进、一个字段出
    q = dict(order_info="订单#N001，商品AX200耳机，已签收3天",
             return_request="耳机有杂音，要求退货")
    plain = dspy.Predict(ReturnJudge)
    r1 = plain(**q)
    print(f"[1] Predict        → verdict={r1.verdict} | reason={r1.reason}")
    print(f"    签名声明的输出字段: {list(plain.signature.output_fields)}")

    # ── ② ChainOfThought：同样的签名，多了 reasoning 字段
    cot = dspy.ChainOfThought(ReturnJudge)
    r2 = cot(**q)
    print(f"[2] ChainOfThought → verdict={r2.verdict} | reason={r2.reason}")
    # ★ ChainOfThought 没有 .signature，要穿过 .predict 拿（实测踩到）
    print(f"    签名声明的输出字段: {list(cot.predict.signature.output_fields)}"
          f"  ← 比 Predict 多 reasoning，代码里一个字没改")

    # ── ③ 编译：BootstrapFewShot 去训练集里挑「能通过 metric」的示例
    optimizer = dspy.BootstrapFewShot(metric=accuracy_metric,
                                     max_bootstrapped_demos=2, max_labeled_demos=2)
    compiled = optimizer.compile(dspy.ChainOfThought(ReturnJudge), trainset=trainset)
    demos = getattr(compiled.predict, "demos", [])
    print(f"[3] compile        → 编译产物 demos={len(demos)} 条（从 {len(trainset)} 条训练集里自动挑的）")
    if args.show_demos:
        for i, d in enumerate(demos, 1):
            print(f"      demo{i}: verdict={d.get('verdict')} | {str(d.get('reason'))[:40]}")

    # ── ④ 序列化：这是书稿「Prompt 是可版本管理资产」的具体形态
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "compiled_judge.json"
    compiled.save(str(ckpt))
    size = ckpt.stat().st_size
    reloaded = dspy.ChainOfThought(ReturnJudge)
    reloaded.load(str(ckpt))
    demos2 = getattr(reloaded.predict, "demos", [])
    def _norm(ds):
        # demos 里混着 dspy.Example 和 plain dict（实测两种都有），统一归一后再比
        return json.dumps([d.toDict() if hasattr(d, "toDict") else dict(d) for d in ds],
                          ensure_ascii=False, sort_keys=True, default=str)

    same = _norm(demos) == _norm(demos2)
    print(f"[4] save/load      → {ckpt.name} {size} 字节；重载后 demos={len(demos2)} 条，"
          f"内容{'一致' if same else '不一致 ⚠️'}")

    # ── ⑤ 在验证集上跑一遍（桩 LM 下这个数字没有意义，只证明 evaluate 链路通）
    hit = 0
    for ex in devset:
        pred = compiled(order_info=ex.order_info, return_request=ex.return_request)
        hit += bool(accuracy_metric(ex, pred))
    print(f"[5] devset         → {hit}/{len(devset)} 命中"
          + ("（★ 桩 LM 的固定答案，**不代表优化效果**）" if not args.live else ""))

    print(f"""
[DSPy 到底做了什么]
  · Signature：把「输入什么、输出什么」写成类 —— 这是接口，不是 Prompt
  · Module   ：Predict（直答）/ ChainOfThought（先推理再答）—— 换模块不改签名
  · compile  ：拿 metric + trainset 去**搜**示例与措辞，产出可序列化的程序对象
  · 换模型   ：重跑一次 compile；业务代码（上面那段 q + 调用）一个字不用改
  · 本脚本证的是机制；「比手写 Prompt 好」需要真模型 + 真评测集，本机没有，不声称
""")

    if not args.keep:
        shutil.rmtree(OUT, ignore_errors=True)
        print("（.dspy_demo/ 已清理；要留档加 --keep）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
