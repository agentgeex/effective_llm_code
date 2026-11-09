"""ch04/onnx_m.py —— 用 optimum 的 ORTModelForCausalLM 跑已导出的 ONNX 模型

① 这是什么：加载一个「已导出的 ONNX 目录」（model.onnx + model.onnx_data + config/tokenizer），
   用 chat template 组 prompt，贪婪解码生成一句话并打印耗时；自动选 CUDA/CPU 执行提供程序。
② 对应书稿：第4章 §4.4.5 格式转换路径图（书稿给的是 `export=True` 就地导出的写法）。
   与书稿写法不同，差异见 ⑥。
③ 怎么跑：
     # 先导出一次（with-past 版才有 KV cache）：
     venv-onnx/bin/optimum-cli export onnx \\
         --model models/Qwen2.5-0.5B-Instruct \\
         --task text-generation-with-past models/Qwen2.5-0.5B-onnx-past
     export ONNX_MODEL_DIR=models/Qwen2.5-0.5B-onnx-past
     python（需 onnxruntime + torch 的环境） ch04/onnx_m.py
④ 依赖：optimum[onnxruntime]（实测 optimum 2.1.0 + optimum-onnx 0.1.0）、transformers 4.57.6、
   onnxruntime 1.31.1、torch 2.6.0+cpu；需已导出的 ONNX 目录（实测 Qwen2.5-0.5B，2.4G）。
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）， 重跑（见文件末记录）。
⑥ 与书稿的差异：
   1) 书稿是「加载 HF 原模型 + export=True 就地导出 + save_pretrained」；本文件是「加载已导出的
      ONNX 目录」—— 两种形态都指向同一个 ORTModelForCausalLM，但前者多一步导出。
   2) 模型路径改读环境变量 ONNX_MODEL_DIR（书稿里是写死的相对路径），未设置时明确报错不猜。
   3) 其余逻辑与书稿一致：自动选 CUDA/CPU、chat template、贪婪解码、打印耗时。
   4) 必须用 with-past 导出的目录，否则报
      ValueError: use_cache=True was passed but the loaded model does not support pkv cache reuse。
"""

import os
import sys

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import torch
import time

# --- 配置 ---
# 指向「已导出的 ONNX 目录」（里面是 model.onnx + model.onnx_data + config/tokenizer）
model_path = os.environ.get("ONNX_MODEL_DIR", "")
if not model_path:
    print("缺少环境变量 ONNX_MODEL_DIR：请指向已导出的 ONNX 目录", file=sys.stderr)
    print("例如：export ONNX_MODEL_DIR=models/Qwen2.5-0.5B-onnx", file=sys.stderr)
    sys.exit(2)
if not os.path.isdir(model_path):
    print("ONNX_MODEL_DIR 不存在：%s" % model_path, file=sys.stderr)
    sys.exit(2)

# 自动检测并设置设备。如果您安装了 onnxruntime-gpu，它将使用 CUDA。
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if device == "cuda" else -1
print(f"Device set for ORT: {device}")

# 1. 加载 ONNX 模型 (使用 ORTModelForCausalLM) 和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 指定执行提供程序 (Execution Provider)
# 这确保 ONNX Runtime 知道在哪里执行计算（GPU 或 CPU）
provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"

# ORTModelForCausalLM 会自动加载所有 ONNX 文件 (decoder, decoder_with_past)
print("Loading ORTModel...")
ort_model = ORTModelForCausalLM.from_pretrained(
    model_path, 
    #device=device_id,
    provider=provider
) 
print("Model Loaded successfully.")

# 2. 准备输入 (使用 Chat Template 格式化输入，符合 Chat 模型要求)
messages = [
    {"role": "system", "content": "你是一位专注于简洁和精确回答的中文AI助手。"},
    {"role": "user", "content": "请用一句简短的话解释什么是量子计算。"}
]

# Qwen1.5 模型需要通过 Chat Template 格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 分词并发送到正确的设备
inputs = tokenizer([text], return_tensors="pt").to(device)

# 3. 配置生成参数
generation_kwargs = {
    "max_new_tokens": 50, # 限制生成长度
    "do_sample": False,   # 关闭采样，使用贪婪搜索确保可复现性
    "temperature": 0.0,
    "eos_token_id": tokenizer.eos_token_id, 
    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id 
}

# 4. 执行生成
start_time = time.time()
print("\n--- 开始生成 ---")
print(f"Prompt: {text}")

# ORTModelForCausalLM 的 generate 方法使用导出的 ONNX 图和 KV Cache
output_ids = ort_model.generate(
    **inputs, 
    **generation_kwargs
)

end_time = time.time()

# 5. 解码并打印结果
# 跳过输入的 Prompt 部分 (因为 output_ids 包含 prompt)
input_length = inputs.input_ids.shape[1]
output_text = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True) 

print("\n--- ONNX 模型生成结果 ---")
print(output_text)
print(f"\n生成耗时: {end_time - start_time:.2f} 秒")
# ---- 实测记录（本机 venv-onnx）----
# 解释器：python（需 onnxruntime + torch 的环境）（optimum 2.1.0 / onnxruntime 1.31.1）
# 命令：ONNX_MODEL_DIR=models/Qwen2.5-0.5B-onnx-past ch04/onnx_m.py
# rc=0，输出「量子计算利用量子比特（qubits）进行运算，比传统计算机更高效。」
# 生成耗时：1.92 秒
# 另：书稿的 export=True 写法在 optimum 2.1.0 上实测同样可用（返回 ORTModelForCausalLM）。
