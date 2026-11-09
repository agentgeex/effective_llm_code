#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：同一批 prompt 的微调前后原文并排对比（不评分、不排名）
# ② 对应书稿：ch11 §11.3.4，（双盲对比评测）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/finetune/infer_compare.py
# ④ 依赖：torch、transformers、peft；adapter/（train_lora.py 产物）；权重 Qwen2.5-0.5B-Instruct（本机缓存）
# ⑤ 实测解释器：python（需 torch 的环境）（torch 2.6.0+cu124 / peft 0.21.2 / CUDA True）
#     真跑：先跑 train_lora.py 生成 adapter/，再跑本文件 → rc=0，base 与 base+LoRA 原文并排输出
# ⑥ 与书稿的差异：书稿用 LLM-as-Judge 盲评 + 关键路径回测（需 GPT-4o/Claude key）；本文件只并排原文，不做评分
# ══════════════════════════════════════════════════════════════════
"""infer_compare.py — ch11 §11.3.4「它到底变好了没有」

书稿 §11.3.4 给了两种验证：LLM-as-Judge 盲评 + 关键路径回测。
**两种都要 API key**（一个调 GPT-4o，一个调 Claude），本机没有。
所以这里做的是更朴素、但完全不掺假的第一层：**同一批 prompt，base 与
base+LoRA 各生成一次，原文并排放**。

不评分、不排名 —— 谁好谁坏留给读者看原文判断。
判词在这里一文不值，原文才有价值。

3 个 prompt 的分工（刻意选成三个难度）：
  P1  训练集里出现过的问法       → 看「学没学会」
  P2  同一知识点、换个说法       → 看「是不是只背了那 6 句话」
  P3  书稿 §11.3.4 的关键路径题  → 看「专业语气有没有迁移过去」

产物：outputs/compare.md
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
ADAPTER_DIR = HERE / "adapter"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    dict(tag="P1", kind="训练集内问法",
         q="环形伪影是什么？"),
    dict(tag="P2", kind="同知识点、换说法（训练集里没有这句）",
         q="图像上一圈一圈的纹路，专业上怎么讲？"),
    dict(tag="P3", kind="换到没训过的知识点",
         q="探测器某个通道增益漂移时，我应该检查哪些部件？"),
]


def gen(model, tok, prompt: str, max_new: int = 200) -> tuple[str, float]:
    import torch

    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    t = time.time()
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=False,               # 贪心：去掉随机性，前后两次才可比
            num_beams=1,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.time() - t
    new_tokens = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip(), dt


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} / cuda {cuda}")

    if not ADAPTER_DIR.exists():
        print(f"❌ 没有 {ADAPTER_DIR}，先跑 train_lora.py")
        return 1

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("加载基座模型 ...")
    base = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16 if cuda else torch.float32)
    if cuda:
        base = base.cuda()
    base.eval()

    # ── 必须分两轮，不能交叉 ──────────────────────────────────────
    # 🔴 踩过的坑：PeftModel.from_pretrained(base, ...) 是**就地注入**，
    #    它会把传给它的 base 的每一层替换成 LoraLayer 包装器 —— 不是拷贝。
    #    所以「先挂 adapter，再交替调 base / ft」等于同一个模型跑两遍，
    #    输出一模一样，看起来像「微调没效果」。
    #    正确做法：先在**未注入**的 base 上把所有 prompt 跑完，再挂 adapter 跑第二轮。
    rows = [dict(p, base="", ft="", base_s=0.0, ft_s=0.0, peak_mib=0) for p in PROMPTS]

    print("\n── 第一轮：微调前（base，未注入 adapter）──")
    for i, p in enumerate(PROMPTS):
        out, sec = gen(base, tok, p["q"])
        rows[i]["base"] = out
        rows[i]["base_s"] = round(sec, 2)
        print(f"    [{p['tag']}] {sec:.1f}s  {out[:40]!r}")

    print("\n挂载 LoRA adapter（就地注入 base）...")
    ft = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    ft.eval()

    print("── 第二轮：微调后（base + LoRA）──")
    for i, p in enumerate(PROMPTS):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        out, sec = gen(ft, tok, p["q"])
        peak = torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0
        rows[i]["ft"] = out
        rows[i]["ft_s"] = round(sec, 2)
        rows[i]["peak_mib"] = round(peak)
        print(f"    [{p['tag']}] {sec:.1f}s  {out[:40]!r}")

    # 自检：两轮输出若完全相同，说明注入逻辑又错了，直接报错而不是写出一份假报告
    same = [r["tag"] for r in rows if r["base"] and r["base"] == r["ft"]]
    if same and len(same) == len(rows):
        print(f"\n❌ 全部 {len(rows)} 个 prompt 前后输出完全相同 —— adapter 未生效，"
              f"不写 compare.md")
        return 2
    if same:
        print(f"\n⚠️ {same} 前后输出相同（可能该 prompt 太容易/太难，属正常）")

    # ── 写 compare.md ────────────────────────────────────────────
    L = []
    L.append("# 微调前后对比（同一批 prompt，贪心解码，原文照抄）\n")
    L.append(f"- 基座：`{DEFAULT_MODEL}`（本地缓存，零下载）")
    L.append(f"- 增量：LoRA r=8 / alpha=16，可训练 4.40M / 494.03M = 0.89%")
    tr = OUT_DIR.parent / "outputs" / "train_summary.json"
    if tr.exists():
        ts = json.loads(tr.read_text(encoding="utf-8"))
        lrs = " / ".join(f"{s['stage']} lr={s['lr']:g}" for s in ts["stages"])
        L.append(f"- 训练：train {ts['splits']['train']} 条 × 1 epoch"
                 f"（课程学习两阶段），共 "
                 f"{sum(s.get('steps',0) for s in ts['stages'])} 步；{lrs}")
        L.append(f"- 探针（固定 val 批次）loss：{ts.get('probe_loss_first')} → "
                 f"{ts.get('probe_loss_last')}"
                 f"（{ts.get('loss_drop_pct'):+.1f}%）")
    L.append(f"- 解码：`do_sample=False`（贪心），`max_new_tokens=200`，"
             f"`repetition_penalty=1.1`")
    L.append(f"- 硬件：{'CUDA ' + torch.cuda.get_device_name(0) if cuda else 'CPU'}\n")
    L.append("> ⚠️ 这里**不给评分、不排优劣**。评分要调裁判模型（书稿用 GPT-4o / Claude），")
    L.append("> 本机无 key。原文摆在这里，好坏自己判断。\n")
    L.append("---\n")

    for r in rows:
        L.append(f"## {r['tag']} · {r['kind']}\n")
        L.append(f"**Prompt**：{r['q']}\n")
        L.append(f"### 微调前（base）　{r['base_s']}s\n")
        L.append("```text")
        L.append(r["base"] or "（空）")
        L.append("```\n")
        L.append(f"### 微调后（base + LoRA）　{r['ft_s']}s\n")
        L.append("```text")
        L.append(r["ft"] or "（空）")
        L.append("```\n")
        L.append(f"<sub>该轮峰值显存 {r['peak_mib']} MiB</sub>\n")
        L.append("---\n")

    L.append("## 怎么读这三组输出\n")
    L.append("| | 想验证什么 | 看什么 |")
    L.append("|---|---|---|")
    L.append("| P1 | 有没有学会这套说话方式 | 微调后应出现"
             "「术语 + 参数 + 影响 + 处置」的结构化句式 |")
    L.append("| P2 | 是不是只背了训练集那 6 句话 | 微调后即使换了说法，"
             "仍应给出领域内容（若答成「我不确定」说明是死背） |")
    L.append("| P3 | 专业语气能不能迁移到没训过的问题上 | "
             "微调后应提到探测器/校准/通道这类词 |")
    L.append("")
    L.append("如果 P2/P3 与 P1 差距极大，说明模型只记住了问法模板 —— "
             "这正是模板合成数据的必然局限（见 README 局限第 1 条）。")

    (OUT_DIR / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    meta = {
        "model": DEFAULT_MODEL,
        "prompts": [p["q"] for p in PROMPTS],
        "decode": {"do_sample": False, "max_new_tokens": 200,
                   "repetition_penalty": 1.1},
        "timing": [{"tag": r["tag"], "base_s": r["base_s"], "ft_s": r["ft_s"],
                    "peak_mib": r["peak_mib"]} for r in rows],
    }
    (OUT_DIR / "compare_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n落盘: outputs/compare.md, outputs/compare_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
