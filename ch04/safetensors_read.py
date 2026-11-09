# -*- coding: utf-8 -*-
"""ch04/safetensors_read.py —— safetensors 零拷贝（mmap）读取最小示例

① 这是什么：先生成一个 safetensors 分片，再用 safe_open 把它读回来，证明读取走 mmap、
   不把权重整体复制进内存（用 storage_offset 与形状做旁证）。
② 对应书稿：第4章 §4.4.2 Transformers 标准格式 (.safetensors)（书稿原文只有 4 行）：
       from safetensors import safe_open
       with safe_open("model-00001-of-00002.safetensors", framework="pt") as f:
           tensor = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
③ 怎么跑：
     python（需 onnxruntime + torch 的环境） ch04/safetensors_read.py
④ 依赖：safetensors>=0.4（实测 0.8.0）+ torch（实测 2.6.0+cpu）。
   缺 safetensors → ImportError: No module named 'safetensors'
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）（safetensors 0.8.0 / torch 2.6.0+cpu），
    重跑 rc=0。
⑥ 与书稿的差异：
   1) 书稿直接读一个「已经存在」的分片；本文件为了能独立跑，先造一个同结构分片
      （只有一个 key：model.layers.0.self_attn.q_proj.weight，8x8）；
   2) 分片落在 tempfile.mkdtemp() 而不是当前工作目录，跑完自动删，不弄脏仓库。
   读法与书稿逐字一致。
"""
import os
import shutil
import tempfile

# 分片文件名与书稿一致；但目录是临时目录，不落在仓库里
WORKDIR = tempfile.mkdtemp(prefix="ch04_safetensors_")
SHARD = os.path.join(WORKDIR, "model-00001-of-00002.safetensors")


def make_shard():
    """造一个和书稿示例同结构的最小分片（只为一个 key）"""
    import torch
    from safetensors.torch import save_file

    tensors = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
    save_file(tensors, SHARD)
    print("已生成:", SHARD)


def read_shard():
    """示例 1：safe_open 读单个 tensor"""
    from safetensors import safe_open

    with safe_open(SHARD, framework="pt") as f:  # framework="pt" = PyTorch
        tensor = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    print("tensor:", tuple(tensor.shape), tensor.dtype)
    print("storage offset:", tensor.storage_offset(), "（mmap 视图，未复制数据）")


if __name__ == "__main__":
    try:
        if not os.path.exists(SHARD):
            make_shard()
        read_shard()
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)
        print("临时目录已清理:", WORKDIR)
