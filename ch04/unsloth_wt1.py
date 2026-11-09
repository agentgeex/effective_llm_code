"""ch04/unsloth_wt1.py —— Unsloth + LoRA 微调完整脚本

① 这是什么：用 Unsloth 的 FastLanguageModel 做 4bit 加载 → 挂 LoRA 适配层 → 把 Alpaca
   格式数据拼成对话模板 → SFTTrainer 训练 → 保存适配器与 tokenizer。
② 对应书稿：第4章 §4.3.2 大模型技术栈：模型预训练和微调。
   书稿只给两段：FastLanguageModel.from_pretrained + get_peft_model；本文件是仓库的完整训练版。
③ 怎么跑（需先装齐 unsloth 全家桶 + 有 GPU + 有模型/数据）：
     /usr/bin/python3 ch04/unsloth_wt1.py
   实际路径写死在脚本里：模型 models/deepseek-coder-1.3b-instruct/deepseek-coder-1.3b-instruct，
   数据集 data/，输出 ./deepseek-coder-1.3b-finetuned。
④ 依赖：unsloth、trl、datasets、transformers、torch(CUDA)、bitsandbytes；
   需 GPU（4bit 微调 1.3B 级）；需基座模型权重 + 指令数据集。
⑤ 实测状态（四个解释器逐个真 import，**未跑通**）：
   - venv-onnx / .venv：`import unsloth` → ModuleNotFoundError: No module named 'unsloth'
   - 系统 /usr/bin/python3：有 unsloth（能打印 🦥 Unsloth 横幅），但依赖链断了：
       import unsloth  → ModuleNotFoundError: No module named 'psutil'
       import trl / transformers / peft → ModuleNotFoundError: No module named 'regex'
       import datasets → ModuleNotFoundError: No module named 'pandas'
       import accelerate → ModuleNotFoundError: No module named 'safetensors'
   - 权重：models/deepseek-coder-1.3b-instruct 为空目录（书稿写死的 deepseek-coder-1.3b-instruct 不在）；
     数据集：data/ 存在（code_alpaca_20k.json，8MB）。
   结论：**依赖（缺 psutil/regex/pandas/safetensors）+ 基座权重两个前提都没有，无法真跑**。
   不改写成假可跑（不补 mock 训练），保持与书稿一致。
⑥ 与书稿的差异：书稿对应小节 只演示加载 + 挂 LoRA（约 20 行），本文件是完整训练脚本
   （含数据格式化、TrainingArguments、SFTTrainer、保存）；核心两段（from_pretrained /
   get_peft_model）与书稿一致，只是书稿用 unsloth/Qwen2.5-7B、本文件用本地 deepseek-coder-1.3b。
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch



# 加载模型和分词器 (4bit量化节省显存)
model, tokenizer = FastLanguageModel.from_pretrained(
    "models/deepseek-coder-1.3b-instruct/deepseek-coder-1.3b-instruct",
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
dataset = load_dataset("data/", split="train")
print(dataset[0])  # 检查第一条数据

# 格式化数据为对话模板（适配DeepSeek-instruct）
def format_instruction(sample):
    return {
        "text": f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
    }
dataset = dataset.map(format_instruction, remove_columns=["instruction", "input", "output"])

# 训练参数
args = TrainingArguments(
    output_dir="models/deepseek-coder-1.3b-instruct/output/deepseek-coder-1.3b-instruct",
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


