"""ch04/model_parser_c.py —— 用 print(model) 看大模型的逻辑结构

① 这是什么：加载一个 CausalLM，直接 print(model)，看权重从磁盘读进内存后按层/维度组织的
   「脑回路」；下面保留一段未启用的注意力矩阵打印代码（书稿写作时用来逐层看注意力）。
   文件名带 _c 是仓库原来的命名，保留不改。
② 对应书稿：第4章 §4.5.2 大模型的逻辑结构：神经元的组织方式。
③ 怎么跑：
     MODEL_DIR=models/Qwen2.5-0.5B-Instruct \\
       python（需 onnxruntime + torch 的环境） ch04/model_parser_c.py
   不设 MODEL_DIR 时用书稿占位路径 /path/to/your/...，加载必然失败。
④ 依赖：torch + transformers（实测 venv-onnx：torch 2.6.0+cpu / transformers 4.57.6）；
   需模型权重。书稿用的是 Qwen3-14B，本机没下；本文件已用 0.5B 权重实测同一功能。
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）， 实跑（见文件末记录）。
⑥ 与书稿的差异：
   1) 模型路径改读环境变量 MODEL_DIR（默认保留书稿的 14B 占位路径）；
   2) 加载参数多了 output_attentions=True（书稿对应小节 没有，但下面备份的注意力矩阵代码需要它）；
   3) 书稿路径是 /home/amdmz/...，仓库版统一改成 /path/to/your/... 占位。print(model) 本体一致。
"""

from transformers import AutoModelForCausalLM,AutoModelForVision2Seq, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

import os

model = AutoModelForCausalLM.from_pretrained(
    os.environ.get("MODEL_DIR", "/path/to/your/motogpt/modle/tran_model/Qwen3-14B"), 
    low_cpu_mem_usage=True,
    output_attentions=True,
    attn_implementation="eager",
    )

print(model)






####暂时不用的代码备份，临时存放下

# tokenizer = AutoTokenizer.from_pretrained("/path/to/your/motogpt/modle/tran_model/Qwen1.5-4B-Chat")
# text = "人工智能正在快速发展，它对我们的生活产生了深远的影响。"
# # 使用分词器将文本转换为 ID
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model(**inputs)
# attentions = outputs.attentions

# tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
# np.set_printoptions(precision=2, suppress=True, linewidth=150)

# print("关注者 \\ 被关注者: ", end=" ")
# for token in tokens:
#     print(f"{token:<10}", end="")
# print()
# print("-" * 150)

# attention_matrix = attentions[0][0].detach().cpu().numpy()

# for i, row in enumerate(attention_matrix):
#     print(f"{tokens[i]:<15}", end="")
#     for val in row:
#         # 使用 [0] 索引来获取数组中的浮点数值
#         print(f"{val[0]:<10.2f}", end="")
#     print()

# print("\n--- 矩阵解读 ---")
# print("每一行代表一个词（'关注者'），它在关注其他词（'被关注者'）。")
# print("数值越大（越接近1），表示关注度越高。")
# print("例如，查看 '它' 这一行，哪个词的数值最高，就表示 '它' 这个词最关注哪个词。")


# #print(attentions)
# ---- 实测记录（本机 venv-onnx）----
# 解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / transformers 4.57.6）
# 命令：MODEL_DIR=models/Qwen2.5-0.5B-Instruct ch04/model_parser_c.py
# rc=0，打印出 Qwen2ForCausalLM 完整结构（24 层 / hidden 896 / vocab 151936）
# 告警原文：The following generation flags are not valid and may be ignored: ['output_attentions'].
#   —— transformers 4.57 的提示，print(model) 本体不受影响。
