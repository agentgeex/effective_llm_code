# -*- coding: utf-8 -*-
"""ch04/transformer_min.py —— 手写单头自注意力 + 多头注意力（最小 Transformer 核心）

① 这是什么：不依赖 transformers 库，用 torch 手写 Scaled Dot-Product Attention 的
   单头版（SimpleSelfAttention）与多头版（MultiHeadAttention），并附最小可跑 demo。
② 对应书稿：第4章 §4.2.1 Transformers 理论框架（单头）/（多头）；
   §4.3.1 基础框架 的 SimpleAttention 与本文件 MultiHeadAttention 是同一个计算
   （书稿只换了参数名 d_model/n_heads，结构一致），与书稿等价，不重复造第二份文件。
③ 怎么跑：
     python（需 onnxruntime + torch 的环境） ch04/transformer_min.py
   （系统 /usr/bin/python3 也有 torch 2.7.0，同样能跑；.venv / .venv-core 没有 torch）
④ 依赖：仅 torch（CPU 版即可）。无模型权重、无网络、无外部服务。
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu）， 重跑 rc=0。
⑥ 与书稿的差异：书稿只定义类、没有调用入口；这里补了 `__main__` 最小 demo（随机张量 +
   形状/有限值/softmax 行和检查）。两个类的实现与书稿逐行一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- 示例 1：单头自注意力 ----
class SimpleSelfAttention(nn.Module):
    """单头自注意力机制的简化实现（书稿 §4.2.1 第一段代码）"""

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)


# ---- 示例 2：多头注意力 ----
class MultiHeadAttention(nn.Module):
    """多头注意力（书稿 §4.2.1 第二段代码）"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(attn_output)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, H = 2, 5, 16, 4
    x = torch.randn(B, L, D)

    sa = SimpleSelfAttention(D)
    mha = MultiHeadAttention(D, H)

    out_sa = sa(x)
    out_mha = mha(x)

    print("输入          :", tuple(x.shape))
    print("单头注意力输出:", tuple(out_sa.shape), "| 有限值:", bool(torch.isfinite(out_sa).all()))
    print("多头注意力输出:", tuple(out_mha.shape), "| 有限值:", bool(torch.isfinite(out_mha).all()))

    # 注意力权重行和应为 1（softmax 性质）
    Q, K = sa.query(x), sa.key(x)
    w = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / sa.scale, dim=-1)
    print("注意力权重行和:", [round(v, 6) for v in w.sum(-1)[0].tolist()])
