# -*- coding: utf-8 -*-
"""ch06 §6.3.1 —— 用 PyTorch 手搭一个 MoE 解码器并训练

① 这是什么：从零实现一个 Transformer MoE 语言模型的完整训练脚本——
   jieba 分词 + torchtext 建词表 → 截断/填充 Dataset → 正弦位置编码 →
   自注意力 + MoE 路由（top-k 专家加权）→ 因果掩码 → 交叉熵训练循环 → 保存。

② 对应书稿：第 6 章 §6.3.1（书稿给了完整实现）。
   本文件与书稿逐行对照：模型/训练逻辑一字未改，差异仅在路径/超参数 env 化（见 ⑥）。

③ 怎么跑：
       # 小步验证（推荐）：1MB 模拟语料 + 1 epoch + 小 batch
       MOCK_MB=1 MAX_LINES=200 NUM_EPOCHS=1 BATCH_SIZE=2 MAX_SEQ_LEN=64 \\
         /tmp/ch06_torch/bin/python ch06/troch_train_c_moe_v1.py
   本机默认解释器不含 torchtext；需先建环境（见 ④）。

④ 依赖：torch、torchtext、jieba、numpy。
   `torchtext` 已停止维护，最后一版 0.18.0 要求 torch==2.3.0；
   本机验证环境：`/tmp/ch06_torch`（Python 3.11.15 + torch 2.3.0 + torchtext 0.18.0 + jieba）
   —— 详见 README §4.1。语料默认自动生成模拟数据，不需要外部下载。

⑤ 实测解释器与结果：见文件末尾「实测记录」注释块。

⑥ 与书稿的差异：
   1. 书稿写死 `/home/amdmz/motogpt/...` 路径；本文件改为环境变量可覆盖。
   2. 新增 `MOCK_MB` / `NUM_EPOCHS` / `BATCH_SIZE` / `MAX_SEQ_LEN` / `MAX_LINES` 环境变量，
      默认值与书稿一致（100MB / 3 / 32 / 512 / 全部），只为便于小步验证。
   3. 补建语料/模型输出目录（书稿假定目录已存在）。
   4. 模型结构、MoE 路由、因果掩码、损失计算与书稿**逐行一致**。
"""
# conda env  torch_final_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import jieba
import os
import math
import numpy as np
import random

# --- 0. 环境和超参数设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 全局超参数 (请根据你的硬件和数据量进行调整)
VOCAB_MIN_FREQ = 5           # 词汇表最小出现频率
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))       # 批次大小
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))        # 训练轮数
LEARNING_RATE = 0.0001       # 学习率
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))      # 最大序列长度
MAX_LINES = int(os.environ.get("MAX_LINES", "0"))          # 只取前 N 行（0=全部，便于小步验证）

# 模型超参数
MODEL_PARAMS = {
    'd_model': 512,          # 词嵌入/隐藏层维度 (d_model)
    'nhead': 8,              # 注意力头数 (nhead)
    'num_layers': 6,         # 层数 (num_layers)
    'dim_feedforward': 2048,  # 前馈网络维度
    'dropout': 0.1           # Dropout比率
}

# MoE 相关的超参数
MOE_PARAMS = {
    'num_experts': 8,       # 专家的总数量
    'top_k': 2              # 每个 token 选择的专家数量
}

# 文件路径（书稿为固定路径；这里改为环境变量可覆盖）
CORPUS_FILE = os.environ.get(
    "CORPUS_FILE", "/path/to/your/motogpt/datasource/test_data1/merged_file.txt"
)
MODEL_FILE = os.environ.get(
    "MODEL_FILE", "/path/to/your/motogpt/modle/motozolo_model/troch_train/transformer_model.pth"
)

# 特殊词元
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = [SPECIAL_SYMBOLS.index(s) for s in SPECIAL_SYMBOLS]


# --- 1. 数据集类：包含序列截断逻辑 ---
class TextFileDataset(Dataset):
    def __init__(self, file_path, vocab, tokenizer, max_len):
        self.file_path = file_path
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len  # 存储最大长度

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        if MAX_LINES > 0:
            self.lines = self.lines[:MAX_LINES]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        tokens = self.tokenizer(line.strip())
        ids = self.vocab(tokens)
        ids_with_specials = [BOS_IDX] + ids + [EOS_IDX]

        # 关键截断逻辑：确保序列长度不超过 MAX_SEQ_LEN
        if len(ids_with_specials) > self.max_len:
            # 截断时保留 BOS 和 EOS
            ids_with_specials = ids_with_specials[:self.max_len - 1] + [EOS_IDX]

        return torch.tensor(ids_with_specials, dtype=torch.long)

# --- 2. 数据处理封装类 ---


class CorpusProcessor:
    # 构造函数现在接收 max_len
    def __init__(self, file_path, vocab_min_freq, max_len):
        self.file_path = file_path
        self.vocab_min_freq = vocab_min_freq
        self.max_len = max_len
        self.vocab = None
        self.tokenizer = self._get_tokenizer()

    @staticmethod
    def _get_tokenizer():
        """返回一个中文分词器"""
        return lambda text: list(jieba.cut(text))

    def _yield_tokens(self):
        """逐行读取文件并生成分词后的词元"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield self.tokenizer(line.strip())

    def build_vocab(self):
        """构建并返回词汇表"""
        print("正在构建词汇表...")
        self.vocab = build_vocab_from_iterator(
            self._yield_tokens(),
            min_freq=self.vocab_min_freq,
            specials=SPECIAL_SYMBOLS
        )
        self.vocab.set_default_index(UNK_IDX)
        print(f"词汇表大小: {len(self.vocab)}")
        return self.vocab

    def get_dataloader(self, batch_size):
        """返回数据加载器"""
        if self.vocab is None:
            raise ValueError("请先调用 build_vocab 方法")

        dataset = self._get_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def _get_dataset(self):
        """返回数据集，并将 max_len 传入"""
        # 使用独立的 TextFileDataset 类
        return TextFileDataset(self.file_path, self.vocab, self.tokenizer, self.max_len)

    def _collate_fn(self, batch):
        """用于DataLoader的批次填充函数 (batch_first=False)"""
        padded_batch = nn.utils.rnn.pad_sequence(batch, padding_value=PAD_IDX, batch_first=False)
        return padded_batch


# --- 3. 模型构建 (Transformer MoE 解码器) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MoELayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gating_network = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x 形状: (S, B, E)
        seq_len, batch_size, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (S*B, E)

        # 1. 门控网络计算路由权重
        gate_logits = self.gating_network(x_flat)
        weights, selected_experts_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # 2. 对权重进行 softmax 归一化
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)

        # 3. 负载均衡损失 (简化，仅用于占位)
        load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 4. 初始化输出张量
        final_output = torch.zeros_like(x_flat)

        # 5. 遍历每个 token 进行路由和加权求和
        for i, token_weights in enumerate(weights):
            token = x_flat[i]
            token_output = torch.zeros_like(token)

            for j, expert_idx in enumerate(selected_experts_indices[i]):
                expert_output = self.experts[expert_idx](token)
                token_output += expert_output * token_weights[j]

            final_output[i] = token_output

        return final_output.view(seq_len, batch_size, d_model), load_balancing_loss


class TransformerMoEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_experts, top_k):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        # 修复：移除 batch_first=True，默认使用 batch_first=False
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        # 使用 MoE 层
        self.moe_layer = MoELayer(d_model, dim_feedforward, num_experts, top_k)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask, src_padding_mask):
        # src, x 形状: (S, B, E)

        # 自注意力层
        x = src
        norm_x = self.norm1(x)
        # 注意：src_mask 和 src_padding_mask 都是 (S, S) 和 (B, S) 形状，与 batch_first=False 兼容
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=src_mask, key_padding_mask=src_padding_mask)
        x = x + self.dropout1(attn_output)

        # MoE 层
        y = self.norm2(x)
        y, load_balancing_loss = self.moe_layer(y)
        x = x + self.dropout2(y)

        return x, load_balancing_loss


class TransformerMoEDecoderOnly(nn.Module):
    """模型层次设定"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, num_experts, top_k):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 构建 MoE 解码器层
        self.layers = nn.ModuleList([
            TransformerMoEDecoderLayer(d_model, nhead, dim_feedforward, dropout, num_experts, top_k)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask, src_padding_mask):
        src = self.pos_encoder(self.embedding(src))

        total_load_balancing_loss = 0
        for layer in self.layers:
            src, load_balancing_loss = layer(src, src_mask, src_padding_mask)
            total_load_balancing_loss += load_balancing_loss

        output = self.final_norm(src)
        return self.fc_out(output), total_load_balancing_loss


# --- 4. 训练管理类 ---
class ModelTrainer:
    def __init__(self, model, dataloader, criterion, optimizer, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def _create_causal_mask(self, size):
        # 修复：返回布尔类型掩码 (True 表示屏蔽)，解决 UserWarning
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def train(self, num_epochs):
        print("开始训练...")
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                batch = batch.to(self.device)  # (S, B)
                seq_len = batch.size(0)

                # 创建因果掩码 (Bool type)，并发送到设备
                mask = self._create_causal_mask(seq_len).to(self.device)

                # 创建填充掩码 (Bool type - True表示要忽略/屏蔽) (B, S)
                padding_mask = (batch == PAD_IDX).transpose(0, 1).to(self.device)

                # 前向传播：接收两个返回值
                output, load_balancing_loss = self.model(batch, mask, padding_mask)

                # 计算损失
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)  # 预测下一个词
                target = batch[1:].view(-1)

                # 总损失 = 交叉熵损失 + 负载均衡损失
                loss = self.criterion(output, target) + load_balancing_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f'纪元: {epoch+1}/{NUM_EPOCHS}, 平均损失: {avg_loss:.4f}')

        print("训练完成！")

# --- 主函数执行 ---


def main():
    # 生成模拟数据
    def generate_mock_data(file_path, size_mb=100):
        if os.path.exists(file_path):
            print(f"文件 '{file_path}' 已存在，跳过数据生成。")
            return

        print(f"正在生成 {size_mb}MB 的模拟中文数据...")
        sentences = [
            "今天天气真好，我们一起出去玩吧。",
            "深度学习是一项非常有趣的技术。",
            "这是一个非常棒的例子，它将帮助我学习。",
            "自然语言处理是人工智能的一个重要分支。",
            "我喜欢编程，尤其是使用 Python。",
        ]
        total_size = 0
        with open(file_path, 'w', encoding='utf-8') as f:
            while total_size < size_mb * 1024 * 1024:
                line = random.choice(sentences) + "\n"
                f.write(line)
                total_size += len(line.encode('utf-8'))
        print("数据生成完毕。")

    # 先建目录，再生成语料（书稿假定目录已存在）
    os.makedirs(os.path.dirname(os.path.abspath(CORPUS_FILE)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_FILE)), exist_ok=True)

    generate_mock_data(CORPUS_FILE, int(os.environ.get("MOCK_MB", "100")))

    # 1. 初始化数据处理器
    # 传入 MAX_SEQ_LEN 进行截断
    processor = CorpusProcessor(CORPUS_FILE, VOCAB_MIN_FREQ, MAX_SEQ_LEN)
    vocab = processor.build_vocab()

    # 2. 初始化模型
    model = TransformerMoEDecoderOnly(
        vocab_size=len(vocab),
        **MODEL_PARAMS,
        **MOE_PARAMS
    )

    # 3. 初始化数据加载器、损失函数和优化器
    dataloader = processor.get_dataloader(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 初始化训练器并开始训练
    trainer = ModelTrainer(model, dataloader, criterion, optimizer, device)
    trainer.train(NUM_EPOCHS)

    # 5. 保存模型和词汇表
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_hyperparameters': {**MODEL_PARAMS, **MOE_PARAMS, 'max_seq_len': MAX_SEQ_LEN}
    }, MODEL_FILE)

    print(f"模型和词汇表已成功保存到 {MODEL_FILE}")


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────
# 实测记录
#   环境：/tmp/ch06_torch（Python 3.11.15 / torch 2.3.0+cu121 / torchtext 0.18.0 / jieba 0.42.1）
#   关键：torchtext 0.18.0 必须配 torch 2.3.x。自由解析装到 torch 2.15 时
#         import torchtext 报 libtorchtext.so undefined symbol（ABI 不匹配）；
#         锁 torch==2.3.0 后正常。
#
#   $ CORPUS_FILE=/tmp/ch06_corpus/merged_file.txt \
#     MODEL_FILE=/tmp/ch06_out/transformer_model.pth \
#     MOCK_MB=1 MAX_LINES=200 NUM_EPOCHS=1 BATCH_SIZE=2 MAX_SEQ_LEN=64 \
#     /tmp/ch06_torch/bin/python ch06/troch_train_c_moe_v1.py
#   正在使用设备: cuda
#   词汇表大小: 40
#   纪元: 1/1, 平均损失: 0.1646
#   模型和词汇表已成功保存到 /tmp/ch06_out/transformer_model.pth
#   RC=0
#
#   这是小步验证（200 行 / 1 epoch / batch 2 / seq 64）：全链路能跑，
#   不代表书稿默认配置（100MB / 3 epochs / batch 32 / seq 512）的速度。
# ─────────────────────────────────────────────────────────
