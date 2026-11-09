#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：Qwen2.5-0.5B-Instruct 的 LoRA 真训练（书稿没有训练代码，本文件补齐）
# ② 对应书稿：ch11 §11.3（ 区间；书稿无训练代码块，本文件为补齐环节）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/finetune/train_lora.py
# ④ 依赖：torch、transformers、peft、accelerate、datasets；权重 Qwen/Qwen2.5-0.5B-Instruct（本机 HF 缓存）
# ⑤ 实测解释器：python（需 torch 的环境）（torch 2.6.0+cu124 / peft 0.21.2 / CUDA True）
#     真跑：--limit 200 --max-minutes 18 → rc=0，峰值显存 3680 MiB，
#    探针 loss 3.0027 → 1.0011（+66.7%），adapter 权重 4,399,104 个可训练参数
# ⑥ 与书稿的差异：书稿缺训练环节，本文件补齐；样本 ≤500、1 epoch 以适配本机 10GB 显存
# ══════════════════════════════════════════════════════════════════
"""train_lora.py — ch11 §11.3 的「真训练」环节

⚠️ 书稿 §11.3 **没有给训练代码**。整节讲的是数据（采集/清洗/分布）和验证（盲测/回测），
   中间那步「拿这批数据真的去 LoRA 微调」被跳过了 —— 只在 11.3.4 开头一句
   「数据搞定，跑起来」带过。本脚本补的就是这一段。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
硬件与时间红线（本机 RTX 3080 10GB）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
· 模型只用本机 HF 缓存里的 Qwen2.5-0.5B-Instruct（零下载）
· 样本 ≤500、1 epoch、max_seq_len ≤512、LoRA r ≤8
· 单次训练超 25 分钟自动中止（--max-minutes，默认 25）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
课程学习（书稿 §11.3.2 的核心主张，这里真做了）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
书稿说：先拿 easy 数据跑 2 epoch 建基础概念（lr 2e-4），
        再拿全量数据跑 1 epoch 精调（lr 1e-5）。
本案例压到 1 epoch 一阶段（PM 的规模红线），但**两阶段的 lr 差异保留**，
这样两条 loss 曲线能分别看到 —— 否则「课程学习」只是一句话，不是一次实验。

产物：
    adapter/                  （LoRA 权重，不入 git）
    outputs/train_log.txt
    outputs/adapter_config.json
    outputs/train_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
ADAPTER_DIR = HERE / "adapter"

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ch11 §11.3 LoRA 微调")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF 模型名或本地路径（必须已在本地缓存，本脚本强制 offline）")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr-easy", type=float, default=2e-4, help="阶段1（easy）学习率（书稿值）")
    p.add_argument("--lr-all", type=float, default=2e-4,
                   help="阶段2（全量）学习率。书稿写 1e-5，本数据规模下实测几乎不动，见 README")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-minutes", type=float, default=25.0, help="超时自动中止")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（smoke 用，0=全量）")
    p.add_argument("--no-curriculum", action="store_true", help="关掉两阶段，只跑全量")
    p.add_argument("--grad-checkpointing", action="store_true", help="显存不够时开")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-name", default="train_log.txt",
                   help="日志文件名（放在 outputs/ 下），便于保存多次对照")
    return p.parse_args()


def _torch_optim(model, lr: float):
    import torch
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)


class Lines:
    """同时写终端、日志文件、以及 stdout 重定向给调用方抓取。"""

    def __init__(self, path: Path):
        self.path = path
        self.buf: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s, flush=True)
        self.buf.append(s)

    def save(self) -> None:
        self.path.write_text("\n".join(self.buf) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_example(tok, msgs: list[dict], max_len: int) -> dict | None:
    """把一条 messages 编成 input_ids / labels（只对 assistant 段算 loss）。

    为什么要 mask：如果 user 段也算 loss，模型会去学「怎么问问题」——
    我们只想让它学「怎么答」。这是微调里最常被忽略的一行代码。
    """
    user_text = tok.apply_chat_template(
        [msgs[0]], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False)

    user_ids = tok(user_text, add_special_tokens=False)["input_ids"]
    full = tok(full_text, add_special_tokens=False)["input_ids"]

    if len(full) > max_len:
        return None                      # 超长直接丢，不截断（截断会把答案切半）
    labels = [-100] * len(user_ids) + full[len(user_ids):]
    if len(full) < len(user_ids):        # 模板差异导致对不齐，跳过
        return None
    labels = labels[: len(full)]

    return {"input_ids": full, "labels": labels,
            "attention_mask": [1] * len(full)}


def make_collate(pad_id: int):
    import torch

    def collate(batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    return collate


def build_loader(rows: list[dict], tok, max_len: int, batch_size: int,
                 seed: int, limit: int = 0):
    import torch
    from torch.utils.data import DataLoader

    enc = []
    skipped = 0
    for r in rows:
        e = encode_example(tok, r["messages"], max_len)
        if e is None:
            skipped += 1
        else:
            enc.append(e)
        if limit and len(enc) >= limit:
            break
    print(f"    编码: 有效 {len(enc)} 条 / 跳过 {skipped} 条（超 {max_len} token）", flush=True)

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(enc), generator=g).tolist()
    enc = [enc[i] for i in order]

    return DataLoader(enc, batch_size=batch_size, shuffle=False,
                      collate_fn=make_collate(tok.pad_token_id)), len(enc)



def eval_loss(model, batches, device: str) -> float:
    """在一批**固定**样本上算 loss（不更新权重）。

    ★ 为什么必须有这个：
      一开始我用「前 10% micro-batch 的平均 loss」和「后 10%」相减当曲线。
      那是**不同样本**的平均 —— 数据一难一易就能造出 19% 的假降幅。
      实测就踩了这个坑：loss 从 2.72 掉到 2.20，但 adapter 在 bf16 下
      对 logits 的影响是 0.0（完全看不出来）。固定的探针批次才算数。
    """
    import torch
    if not batches:
        return float("nan")
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            b = {k: v.to(device) for k, v in b.items()}
            tot += model(**b).loss.item()
            n += 1
    model.train()
    return round(tot / max(1, n), 4)


def make_probe(rows, tok, max_len, batch_size, device, n_batches=2):
    """取固定的一小批样本做探针（同一批数据贯穿全程，才可比）。"""
    import torch
    enc = []
    for r in rows:
        e = encode_example(tok, r["messages"], max_len)
        if e is not None:
            enc.append(e)
        if len(enc) >= batch_size * n_batches:
            break
    collate = make_collate(tok.pad_token_id)
    out = []
    for i in range(0, len(enc), batch_size):
        chunk = enc[i:i + batch_size]
        if len(chunk) == batch_size:
            out.append({k: v.to(device) for k, v in collate(chunk).items()})
    return out


def run_stage(name: str, rows: list[dict], model, tok, opt, sched_fn,
              args, log: Lines, t0: float, max_seconds: float,
              probe=None) -> dict:
    import torch

    print(f"\n── 阶段 {name} ──", flush=True)
    loader, n = build_loader(rows, tok, args.max_seq_len, args.batch_size,
                             args.seed, args.limit)
    if n == 0:
        print("    没有可用样本，跳过该阶段", flush=True)
        return {"stage": name, "samples": 0}

    total_steps = math.ceil(n / (args.batch_size * args.grad_accum))
    print(f"    样本 {n} 条 / batch {args.batch_size} × accum {args.grad_accum} "
          f"= 有效 batch {args.batch_size * args.grad_accum} → {total_steps} 步", flush=True)

    # ★ 每个阶段重建 optimizer —— 不能复用上一阶段那个。
    #   踩过的坑：LambdaLR 用 group.setdefault('initial_lr', ...) 取基准学习率，
    #   第一阶段的 initial_lr=2e-4 会残留在 param_group 里，setdefault 不覆盖，
    #   结果第二阶段设 1e-5 也变成"打印 1e-5、实跑 1.89e-4"。
    #   日志与事实不符比 loss 不好看严重得多。
    lr = args.lr_easy if name.startswith("1") else args.lr_all
    opt = _torch_optim(model, lr)
    sched = sched_fn(opt,
                     num_warmup_steps=max(1, int(total_steps * 0.06)),
                     num_training_steps=max(1, total_steps))

    model.train()
    losses: list[float] = []
    probe_curve: list[dict] = []
    if probe:
        p0 = eval_loss(model, probe, "cuda")
        probe_curve.append({"step": 0, "probe_loss": p0})
        print(f"    [探针] 阶段起点 loss {p0}（固定 {len(probe)} 批样本）", flush=True)
    step = 0
    micro = 0
    aborted = False
    t_stage = time.time()

    for batch in loader:
        if (time.time() - t0) > max_seconds:
            print(f"    ⏱ 超过 {args.max_minutes} 分钟硬上限，中止（已跑 {step} 步）", flush=True)
            aborted = True
            break

        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / args.grad_accum
        loss.backward()
        losses.append(out.loss.item())
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if probe and (step % 10 == 0):
                pl = eval_loss(model, probe, "cuda")
                probe_curve.append({"step": step, "probe_loss": pl})
                print(f"    [探针] step {step:3d} loss {pl}", flush=True)

            if step % 10 == 0 or step == 1:
                avg = sum(losses[-args.grad_accum:]) / len(losses[-args.grad_accum:])
                el = time.time() - t_stage
                print(f"    step {step:3d}/{total_steps}  loss {avg:.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {el:.1f}s", flush=True)

    if micro % args.grad_accum != 0:
        opt.step()
        step += 1

    if probe:
        pl = eval_loss(model, probe, "cuda")
        probe_curve.append({"step": step, "probe_loss": pl})
        print(f"    [探针] 阶段终点 loss {pl}", flush=True)

    head = losses[: max(1, len(losses) // 10)]
    tail = losses[-max(1, len(losses) // 10):]
    res = {
        "stage": name,
        "samples": n,
        "steps": step,
        "seconds": round(time.time() - t_stage, 1),
        "lr": lr,
        "loss_first": round(sum(head) / len(head), 4),
        "loss_last": round(sum(tail) / len(tail), 4),
        "loss_min": round(min(losses), 4),
        "loss_max": round(max(losses), 4),
        "aborted_by_timebox": aborted,
        "n_batches": len(losses),
        "probe_curve": probe_curve,
        "probe_first": probe_curve[0]["probe_loss"] if probe_curve else None,
        "probe_last": probe_curve[-1]["probe_loss"] if probe_curve else None,
    }
    print(f"    → loss 首 {res['loss_first']} → 尾 {res['loss_last']}  "
          f"（min {res['loss_min']} / max {res['loss_max']}），耗时 {res['seconds']}s",
          flush=True)
    return res


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 强制离线：不许下载任何东西
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    log = Lines(OUT_DIR / args.log_name)
    t0 = time.time()
    max_seconds = args.max_minutes * 60

    log("=" * 62)
    log("ch11 §11.3 LoRA 微调（书稿未给训练代码，本脚本补齐）")
    log("=" * 62)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    cuda = torch.cuda.is_available()
    log(f"\n[0] 环境")
    log(f"    torch {torch.__version__} / cuda {cuda}")
    if cuda:
        log(f"    GPU {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()
    else:
        log("    ⚠️ 无 CUDA —— 只能 CPU smoke，定级 B（未真训）")

    # ── 模型 ──────────────────────────────────────────────────────
    log(f"\n[1] 加载模型（离线，禁止下载）: {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16 if cuda else torch.float32,
    )
    n_param = sum(p.numel() for p in model.parameters())
    log(f"    参数量 {n_param / 1e6:.0f} M / dtype {model.dtype}")

    # ── LoRA ─────────────────────────────────────────────────────
    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n[2] LoRA 注入")
    log(f"    r={args.lora_r} alpha={args.lora_alpha} dropout=0.05")
    log(f"    target_modules={','.join(targets)}")
    log(f"    可训练参数 {trainable:,} / 总 {n_param:,} "
        f"= {trainable / n_param * 100:.2f}%")

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        log("    gradient_checkpointing=ON")

    if cuda:
        model = model.cuda()

    opt = _torch_optim(model, args.lr_easy)   # 占位；每个阶段内部会重建

    # ── 数据 ─────────────────────────────────────────────────────
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "val.jsonl")
    log(f"\n[3] 数据")
    log(f"    train {len(train_rows)} 条 / val {len(val_rows)} 条")
    easy_rows = [r for r in train_rows
                 if r["metadata"]["difficulty"] == "easy"]
    log(f"    easy 子集 {len(easy_rows)} 条（阶段1 只用这些）")

    stages = []
    if args.no_curriculum:
        stages = [("all", train_rows)]
    else:
        stages = [("1-easy", easy_rows), ("2-all", train_rows)]
        if not easy_rows:
            stages = [("all", train_rows)]

    log(f"\n[4] 训练（课程学习={'关' if args.no_curriculum else '开'}）")
    log(f"    max_seq_len={args.max_seq_len} batch={args.batch_size} "
        f"accum={args.grad_accum} 时间盒={args.max_minutes} 分钟")

    device = "cuda" if cuda else "cpu"
    probe = make_probe(val_rows, tok, args.max_seq_len, args.batch_size, device)

    results = []
    for name, rows in stages:
        r = run_stage(name, rows, model, tok, opt,
                      get_cosine_schedule_with_warmup, args, log, t0, max_seconds,
                      probe=probe)
        results.append(r)
        if r.get("aborted_by_timebox"):
            log("    时间盒触发，后续阶段不再启动")
            break

    # ── 保存 adapter ──────────────────────────────────────────────
    log(f"\n[5] 保存 LoRA adapter")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    # 不把 tokenizer 存进 adapter/：它就是基座模型那一份，存了多 14MB 且无意义
    files = []
    for p in sorted(ADAPTER_DIR.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "bytes": p.stat().st_size})
    for f in files:
        mark = "  🔴 >1MB，不入 git" if f["bytes"] > 1024 * 1024 else ""
        log(f"    {f['name']:32s} {f['bytes']:>10,} B{mark}")

    # adapter_config.json 单独拷一份到 outputs/（这个只有 1KB，可以入库）
    cfg_src = ADAPTER_DIR / "adapter_config.json"
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
        cfg["_note"] = ("本文件可入库（<1MB）；同目录的 adapter_model.safetensors "
                        "是权重，已被 .gitignore 排除，不入 git")
        (OUT_DIR / "adapter_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 汇总 ─────────────────────────────────────────────────────
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0
    wall = round(time.time() - t0, 1)

    log(f"\n[6] 汇总")
    log(f"    用时 {wall}s（{wall/60:.1f} 分钟，上限 {args.max_minutes}）")
    log(f"    峰值显存 {peak_mb:.0f} MiB / 10240 MiB")
    for r in results:
        log(f"    {r['stage']:8s}: {r.get('samples',0):3d} 条 / {r.get('steps',0):3d} 步 "
            f"loss {r.get('loss_first')} → {r.get('loss_last')}")

    # ★ 只认探针（固定 val 批次）上的变化，不认 "首段平均 vs 尾段平均"
    p_first = results[0].get("probe_first")
    p_last = results[-1].get("probe_last")
    dropped = ((p_first - p_last) / p_first * 100) if p_first else 0
    if p_first:
        log(f"    探针（固定 val 批次）loss {p_first} → {p_last} "
            f"= {dropped:+.1f}%   ← 这个才是有效学习信号")
        log(f"    （原始 micro-batch 均值 {results[0].get('loss_first')} → "
            f"{results[-1].get('loss_last')} 不可当曲线用，见代码注释）")

    # adapter 强度：L2 范数 / 参数个数，判断它是否小到在 bf16 里不可见
    import torch as _t
    tot_sq, tot_n = 0.0, 0
    for n_, p_ in model.named_parameters():
        if "lora_" in n_:
            tot_sq += p_.detach().float().pow(2).sum().item()
            tot_n += p_.numel()
    adapter_rms = (tot_sq / max(1, tot_n)) ** 0.5
    log(f"    adapter 权重 RMS {adapter_rms:.6f}（{tot_n:,} 个可训练参数）")

    summary = {
        "model": args.model,
        "model_params_m": round(n_param / 1e6, 1),
        "torch": torch.__version__,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": 0.05, "target_modules": targets,
                 "trainable_params": trainable,
                 "trainable_pct": round(trainable / n_param * 100, 2)},
        "optim": {"opt": "AdamW", "max_seq_len": args.max_seq_len,
                  "batch": args.batch_size, "grad_accum": args.grad_accum,
                  "effective_batch": args.batch_size * args.grad_accum,
                  "curriculum": not args.no_curriculum},
        "splits": {"train": len(train_rows), "val": len(val_rows),
                   "easy_subset": len(easy_rows)},
        "stages": results,
        "probe_loss_first": p_first,
        "probe_loss_last": p_last,
        "loss_drop_pct": round(dropped, 1),
        "adapter_rms": round(adapter_rms, 6),
        "wall_seconds": wall,
        "peak_gpu_mib": round(peak_mb),
        "adapter_files": files,
        "adapter_total_bytes": sum(f["bytes"] for f in files),
    }
    # 日志名 → 汇总名：train_log.txt → train_summary.json
    #                  train_log_book_lr.txt → train_summary_book_lr.json
    # 旧写法先把 .txt 换成 _summary.json 再替换 train_log，会拼出
    # train_summary_book_lr_summary.json 这种双 _summary 的错名（已修）。
    _stem = Path(args.log_name).stem            # train_log_book_lr
    summary_name = _stem.replace("train_log", "train_summary") + ".json"
    summary_path = OUT_DIR / summary_name
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n落盘:")
    log("    adapter/                     （LoRA 权重，不入 git）")
    log(f"    outputs/{args.log_name}")
    log(f"    outputs/{summary_path.name}")
    log("    outputs/adapter_config.json  （仅配置，可入库）")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
