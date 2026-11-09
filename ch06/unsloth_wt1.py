# -*- coding: utf-8 -*-
"""ch06 §6.4.1 —— unsloth 4bit LoRA 微调完整流程

① 这是什么：unsloth `FastLanguageModel` 的完整微调链路——
   4bit 加载 → `get_peft_model` 挂 LoRA → 数据集格式化 → `TrainingArguments`
   → `SFTTrainer.train()` → `save_pretrained`。

② 对应书稿：第 6 章 §6.4.1（书稿只给 4bit 加载：
   `unsloth/llama-3-8b-bnb-4bit` + `load_in_4bit=True`，下文一句「可直接用 SFTTrainer」；
   本文件是从原仓库搬来的完整版，续到了 `get_peft_model` / 数据集 / 训练 / 保存）。

③ 怎么跑（需 GPU + 模型 + 数据集）：
       export MODEL_DIR=/path/to/deepseek-coder-1.3b-instruct
       export DATASET_DIR=data/
       export OUTPUT_DIR=/path/to/output
       python ch06/unsloth_wt1.py
   三个路径均可被环境变量覆盖，默认值保留原仓库写法。

④ 依赖：unsloth / trl / datasets / peft / transformers / torch(CUDA) / bitsandbytes。
   需要：1.3B 级模型（示例路径 `models/deepseek-coder-1.3b-instruct/deepseek-coder-1.3b-instruct`，
   **本机不存在**）、instruction 格式数据集（`data/` 存在，
   内含 code_alpaca_20k.json）、CUDA GPU（4bit 量化必须）。

⑤ 实测解释器与结果（均真 import）：
   - venv-onnx / .venv / .venv-core → ModuleNotFoundError: No module named 'unsloth'
   - 系统 python3（用户站点已装 unsloth 2025.6.5）：
       ModuleNotFoundError: No module named 'psutil'
   - 临时环境补齐 psutil/regex/pandas/safetensors 并锁 transformers==4.52.3 后：
       ModuleNotFoundError: Could not import module 'DataCollatorForLanguageModeling'
       （链路：unsloth → unsloth_zoo.dataset_utils → trl.trainer.utils → transformers；
        未定位到单一缺包，属版本间不兼容）
   → 结论：**本轮未能跑通**。卡在依赖链，且模型目录不存在。不把未跑通的写成能跑。

⑥ 与书稿的差异：
   1. 书稿对应小节 只演示 4bit 加载（HF 仓库名 `unsloth/llama-3-8b-bnb-4bit`）；
      本文件是原仓库完整版：本地 1.3B 模型 + `get_peft_model` + 数据集 + 训练/保存。
   2. 模型/数据集/输出路径改为环境变量可覆盖，默认值保留原仓库硬编码路径。
   3. 书稿用 `### Instruction / ### Input / ### Response` 模板，本文件同；
      但数据集列名需为 instruction/input/output，实际数据集列名（code_alpaca_20k）可能不同，
      运行前需确认（本轮因依赖未跑通，未验证到这一步）。
   4. `from_pretrained(max_seq_length=2048)` 与 `SFTTrainer(max_seq_length=1024)` 不一致，
      这是原仓库写法；新版 TRL 已改为 `SFTConfig(max_seq_length=...)`，未验证。
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import os

MODEL_DIR = os.environ.get("MODEL_DIR", "models/deepseek-coder-1.3b-instruct/deepseek-coder-1.3b-instruct")
DATASET_DIR = os.environ.get("DATASET_DIR", "data/")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "models/deepseek-coder-1.3b-instruct/output/deepseek-coder-1.3b-instruct"
)



# 加载模型和分词器 (4bit量化节省显存)
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_DIR,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,  # 4bit量化
)

# 添加适配层（可选，提升微调效果）
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# 加载数据集
dataset = load_dataset(DATASET_DIR, split="train")
print(dataset[0])  # 检查第一条数据

# 格式化数据为对话模板（适配DeepSeek-instruct）
def format_instruction(sample):
    return {
        "text": f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
    }
dataset = dataset.map(format_instruction, remove_columns=["instruction", "input", "output"])

# 训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_8bit",
)

# 使用SFTTrainer训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=args,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("deepseek-coder-1.3b-finetuned")
tokenizer.save_pretrained("deepseek-coder-1.3b-finetuned")


