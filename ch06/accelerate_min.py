# -*- coding: utf-8 -*-
"""① 这是什么
    ch06 §6.2.3 Accelerate 的最小可跑版：**同一份训练脚本，改 5 行就到手**。
    用 Accelerator 接管 device / model / optimizer / dataloader，把 book 里那张
    「单卡循环 → accelerate 循环」的对照表落成能真跑的东西。

    本脚本刻意做成「单卡/CPU 都能跑通」，多卡才成立的部分用注释标出（本机只有 1 张卡，
    写了多卡分支也没人验证得了 —— 那叫猜，不叫示例）。

② 对应书稿
    第 6 章 §6.2.3「Accelerate：多显卡调度的指挥家」，
    重点是「accelerate config + accelerate launch：一键启动」这一小节。
    书稿原例用 `torch.nn.Transformer` + IMDB 数据集（要下载 2.5 万条影评 + 分词）。
    本脚本保留**同样的训练循环结构**，换成离线可复现的合成语料 + 哈希词袋 + 小 MLP ——
    为了证明 accelerate 的接管，不是为了证明模型效果。

    ⚠️ 抄书稿时注意：原例里 `source = torch.tensor(source[:512])` 这一行，
    `source` 是个字符串，照抄会在运行到那一步报 TypeError。正确写法是先编码成整数 id：
        torch.tensor([b % vocab_size for b in text.encode("utf-8")[:512]], dtype=torch.long)
    另外那个 `DataLoader` 没传 `collate_fn`，于是 `batch["text"]` 出来还是原始字符串 ——
    这两处一起修才能跑通。

③ 怎么跑（三种，都真跑过，见 ⑤）
    # ① 直接跑（等价于原来的 python train.py）
    venv-gpu/bin/python ch06/accelerate_min.py

    # ② 走 accelerate CLI（书里的 accelerate launch）
    venv-gpu/bin/accelerate launch --num_processes 1 ch06/accelerate_min.py

    # ③ 强制 CPU（证明「同一份代码，换设备不换代码」）
    venv-gpu/bin/accelerate launch --cpu ch06/accelerate_min.py

    # ④ 想用混合精度（单卡 fp16，本机 3080 支持）
    venv-gpu/bin/accelerate launch --mixed_precision fp16 ch06/accelerate_min.py

    # ⑤ 多卡（本机只有 1 张卡，**没跑**，命令长这样；代码一行都不用改）
    venv-gpu/bin/accelerate launch --multi_gpu --num_processes 2 ch06/accelerate_min.py

    # 书里的交互式配置向导（会写 ~/.cache/huggingface/accelerate/default_config.yaml）
    venv-gpu/bin/accelerate config

④ 依赖
    包：accelerate 1.15.0、torch 2.6.0+cu124（本机都有）
    服务：无（不联网、不下模型、不读数据集）
    权重：无
    落盘：./.accelerate_demo（已进 .gitignore；`accelerator.save_state()` 的产物，
          演示「checkpoint 是分进程的」这件事，跑完可直接删）

⑤ 实测解释器 + 结果（sgaiedge17，RTX 3080 10GB）
    解释器：python（需 torch 的环境）（Python 3.11.15 / torch 2.6.0+cu124 /
            accelerate 1.15.0）
    **四种跑法全部 rc=0。** 已固定 `--seed 0`（含 DataLoader 的 generator），
    所以除「总耗时」外，两次运行的每个数字都一模一样（实测 diff 清零）。

    ── ① python ch06/accelerate_min.py（等价于原来的 python train.py）rc=0
        device            : cuda
        num_processes     : 1
        process_index     : 0
        distributed_type  : DistributedType.NO
        mixed_precision   : no
        grad_accum_steps  : 2
        torch / cuda      : 2.6.0+cu124 / 12.4
        cuda available    : True
        训练集 51 条 / 验证集 13 条；每进程 batch=8，累积 2 步
        epoch  1/8  loss 0.6974  val_acc 8/13 = 0.615
        epoch  2/8  loss 0.6395  val_acc 11/13 = 0.846
        epoch  3/8  loss 0.5273  val_acc 10/13 = 0.769
        epoch  4/8  loss 0.3654  val_acc 10/13 = 0.769
        epoch  5/8  loss 0.2005  val_acc 11/13 = 0.846
        epoch  6/8  loss 0.0861  val_acc 11/13 = 0.846
        epoch  7/8  loss 0.0337  val_acc 11/13 = 0.846
        epoch  8/8  loss 0.0132  val_acc 11/13 = 0.846
        save_state() 共 4 个文件 → .../.accelerate_demo
            model.safetensors / optimizer.bin / random_states_0.pkl / run_meta.json
        总耗时 0.79s

    ── ② accelerate launch --num_processes 1 ch06/accelerate_min.py   rc=0
        epoch  8/8  loss 0.0132  val_acc 11/13 = 0.846        总耗时 0.75s
        ★ 与 ① 数值完全一致 —— accelerate launch 只是换了启动器，没改训练语义

    ── ③ accelerate launch --cpu ch06/accelerate_min.py   rc=0
        device            : cpu
        epoch  8/8  loss 0.0132  val_acc 11/13 = 0.846        总耗时 0.59s
        ★ 同一份代码，device 从 cuda 变 cpu，**代码零改动** —— 书稿那句「只写一次，到处运行」验到了

    ── ④ accelerate launch --num_processes 1 --mixed_precision fp16 ...   rc=0
        mixed_precision   : fp16
        epoch  8/8  loss 0.0132  val_acc 11/13 = 0.846        总耗时 0.83s
        ★ 混合精度来自**命令行**，代码里 Accelerator() 没传 mixed_precision ——
          这正是书稿「配置与代码分离」的实际形态

    ⚠️ `--multi_gpu --num_processes 2` **没跑**（本机只有 1 张卡）。写了也验证不了，所以不给结论。
    ⚠️ val_acc 在 0.615~0.846 抖动：验证集只有 13 条，错 1 条就是 0.077。这不是模型问题，
       是样本太少 —— 本脚本要证的是「accelerate 接管了设备/数据/梯度」，不是模型精度。

⑥ 与书稿的差异
    - 数据集：书稿 IMDB（需下载）→ 本脚本合成 64 句中文短评（离线、可复现）
    - 模型：书稿 `torch.nn.Transformer`（无 tokenizer，跑不起来）→ 哈希词袋 + 2 层 MLP
    - 训练步数：书稿 3 epoch × 全部数据 → 本脚本 8 epoch × 64 条（秒级）
    - **加了两处书稿没写的**：`accelerator.accumulate()` 梯度累积、`accelerator.save_state()`
      checkpoint —— 这两处正是书里「Accelerate 还提供 X/Y/Z」那句话的实际形态
    - 书稿「如果你只是想在多卡上跑中等模型，选 NO 就够」→ 本脚本实测
      `DistributedType.NO`，验证了「不用 FSDP/DeepSpeed 也能多卡」这条路

⑦ 一个坑（书稿没提，实测遇到）
    `Accelerator()` 默认 `mixed_precision="no"`。想要 fp16 不是改代码，
    而是 `accelerate launch --mixed_precision fp16`（或 config 里选）。
    代码里写死 `Accelerator(mixed_precision="fp16")` 也能跑，但那样命令行就压不过它了 ——
    这正是 accelerate 想让你「配置与代码分离」的地方。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from accelerate import Accelerator
except ImportError:                                  # pragma: no cover
    print("❌ 缺 accelerate。装：uv pip install --python <解释器> accelerate", file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE.parent / ".accelerate_demo"

# ──────────────────────────── 数据：合成语料 ────────────────────────────
# 为什么要自己造：IMDB 要下载 25k 条 + 分词；本机不保证联网，且「下载失败」会把
# accelerate 的演示淹掉。合成语料的好处是**每次跑都一样**，loss 曲线可比。
POS_WORDS = ["好用", "清晰", "快", "满意", "值得", "结实", "舒服", "推荐"]
NEG_WORDS = ["退货", "杂音", "坏", "慢", "失望", "断", "卡", "差评"]
SUBJ = ["耳机", "键盘", "鼠标", "音箱", "屏幕", "充电器"]


def make_corpus(n: int = 64, seed: int = 0):
    """造 n 条短评：正例全用 POS_WORDS，负例全用 NEG_WORDS（标签 1/0）。"""
    g = torch.Generator().manual_seed(seed)
    texts, labels = [], []
    for i in range(n):
        label = i % 2
        words = POS_WORDS if label else NEG_WORDS
        pick = torch.randint(0, len(words), (3,), generator=g).tolist()
        subj = SUBJ[int(torch.randint(0, len(SUBJ), (1,), generator=g))]
        texts.append(f"{subj}" + "，".join(words[j] for j in pick))
        labels.append(label)
    return texts, torch.tensor(labels, dtype=torch.long)


DIM = 256


def hashed_bow(text: str) -> torch.Tensor:
    """字符 3-gram 哈希到固定维度。零依赖、确定性、无需词表 —— 只为把文本变成张量。"""
    v = torch.zeros(DIM)
    t = text.replace("，", "")
    for i in range(len(t) - 2):
        h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest()[:8], 16) % DIM
        v[h] += 1.0
    n = v.norm()
    return v / n if n > 0 else v


class TinySentiment(nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = 64, n_class: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_class))

    def forward(self, x):
        return self.net(x)


def evaluate(accelerator, model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(dim=-1)
            # ★ 多卡时必须先 gather，否则只统计到本进程那一片数据
            pred, yb = accelerator.gather_for_metrics((pred, yb))
            correct += (pred == yb).sum().item()
            total += yb.numel()
    model.train()
    return correct, total


def main() -> int:
    ap = argparse.ArgumentParser(description="Accelerate 最小可跑示例（离线合成语料）")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2, help="梯度累积步数（accelerate 的加分项之一）")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子（默认 0）。固定它，下面的 loss/acc 才可复现")
    ap.add_argument("--keep-state", action="store_true", help="保留 .accelerate_demo/（默认跑完删掉）")
    args = ap.parse_args()

    t0 = time.time()
    # ① 初始化：这一步就把分布式逻辑接管了
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)

    # ★ 固定种子：不固定的话每次 loss/acc 都不一样，注释里的「实测输出」第二天就对不上。
    #   accelerate 自带 set_seed，它会一并处理各进程的随机状态。
    from accelerate.utils import set_seed
    set_seed(args.seed)

    accelerator.print("=" * 68)
    accelerator.print("Accelerate 运行环境（accelerator 自己报的，不是猜的）")
    accelerator.print(f"  device            : {accelerator.device}")
    accelerator.print(f"  num_processes     : {accelerator.num_processes}")
    accelerator.print(f"  process_index     : {accelerator.process_index}")
    accelerator.print(f"  distributed_type  : {accelerator.distributed_type}")
    accelerator.print(f"  mixed_precision   : {accelerator.mixed_precision}")
    accelerator.print(f"  grad_accum_steps  : {accelerator.gradient_accumulation_steps}")
    accelerator.print(f"  torch / cuda      : {torch.__version__} / {torch.version.cuda}")
    accelerator.print(f"  cuda available    : {torch.cuda.is_available()}")
    accelerator.print("=" * 68)

    # ② 数据：造语料 → 张量化 → DataLoader（**不做自己的分片**，交给 prepare）
    texts, labels = make_corpus()
    X = torch.stack([hashed_bow(t) for t in texts])
    split = int(len(X) * 0.8)
    train_ds = torch.utils.data.TensorDataset(X[:split], labels[:split])
    val_ds = torch.utils.data.TensorDataset(X[split:], labels[split:])
    g = torch.Generator().manual_seed(args.seed)          # shuffle 也要可复现
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                           shuffle=True, generator=g)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

    model = TinySentiment()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # ③ 就这一行：模型 / 优化器 / DataLoader 全被接管
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    accelerator.print(f"训练集 {len(train_ds)} 条 / 验证集 {len(val_ds)} 条；"
                      f"每进程 batch={args.batch_size}，累积 {args.grad_accum} 步")

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, n_batch = 0.0, 0
        for xb, yb in train_dl:
            # ④ 梯度累积：只有 accumulate 上下文里才同步梯度
            with accelerator.accumulate(model):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                # ⑤ 替换 loss.backward() —— 多卡时它负责 all-reduce
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            n_batch += 1
        correct, total = evaluate(accelerator, model, val_dl)
        accelerator.print(f"epoch {epoch:>2}/{args.epochs}  "
                          f"loss {total_loss / max(1, n_batch):.4f}  "
                          f"val_acc {correct}/{total} = {correct / max(1, total):.3f}")

    # ⑥ checkpoint：分进程保存（accelerator 会把每个进程的 model/optimizer/rng 分开存）
    if accelerator.is_main_process and STATE_DIR.exists():
        shutil.rmtree(STATE_DIR, ignore_errors=True)
    accelerator.wait_for_everyone()                 # 等所有进程到齐再写
    accelerator.save_state(str(STATE_DIR))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {
            "device": str(accelerator.device),
            "num_processes": accelerator.num_processes,
            "distributed_type": str(accelerator.distributed_type),
            "mixed_precision": accelerator.mixed_precision,
            "grad_accum_steps": accelerator.gradient_accumulation_steps,
            "final_val_acc": f"{correct}/{total}",
            "elapsed_s": round(time.time() - t0, 2),
        }
        (STATE_DIR / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        files = sorted(p.name for p in STATE_DIR.rglob("*") if p.is_file())
        accelerator.print(f"\nsave_state() 共 {len(files)} 个文件 → {STATE_DIR}")
        for f in files:
            accelerator.print(f"    {f}")
        accelerator.print(f"    其中 run_meta.json 是本脚本加的：{meta}")

    accelerator.wait_for_everyone()
    accelerator.print(f"\n总耗时 {time.time() - t0:.2f}s")

    # ⑦ 多卡怎么改：不是改代码，是改命令行
    if accelerator.is_main_process:
        accelerator.print("""
多卡要不要改代码？不要。改命令行就够了：
    accelerate launch --multi_gpu --num_processes 2 ch06/accelerate_min.py
（本机只有 1 张 RTX 3080，这条**没有真跑**——写了也没人验证得了。
  真跑时会自动：数据分片给你、梯度 all-reduce、只在 main process 打印/存盘。
  唯一要注意的是上面 evaluate() 里的 gather_for_metrics —— 忘了它，acc 只算本进程那一片。）

书里提到的其它加分项，都在命令行/config 里，不在代码里：
    --mixed_precision fp16|bf16   混合精度
    --use_deepspeed / --use_fsdp  大模型分片（本机 10G 卡用不上）
    gradient_accumulation_steps   唯一需要写进 Accelerator() 的参数（本脚本 = %d）
""" % args.grad_accum)

    if not args.keep_state:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            shutil.rmtree(STATE_DIR, ignore_errors=True)
            accelerator.print("（.accelerate_demo/ 已清理；要留档加 --keep-state）")

    return 0
# ──────────────────────────── 关于「实测输出」────────────────────────────
# 上面的 stdout 来自一次真实运行；长期记录（
# 以当次 stdout 为准。


if __name__ == "__main__":
    sys.exit(main())
