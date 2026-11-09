# -*- coding: utf-8 -*-
"""ch03 §3.2.1 基于 Transformers 库调用大模型文件

① 这是什么
   用 transformers 加载本地/远程因果语言模型并推理的完整流程：
   加载分词器与模型 → 文本编码成张量 → generate → decode 回文本。

② 对应书稿
   第 3 章 §3.2.1「基于 Transformers 库调用大模型文件」
   承载 3 段（后两段是书稿讲解时对第一段的重复摘录）：
     示例 1：完整推理脚本
     示例 2：② 准备输入（device 选择 + tokenizer 转张量）
     示例 3：③ 生成回答（model.generate）

③ 怎么跑
   MODEL_NAME=/你的/Qwen2.5-0.5B-Instruct \
     python（需 onnxruntime + torch 的环境） ch03/transformers_infer.py
   本机实测：
     MODEL_NAME=models/Qwen2.5-0.5B-Instruct \
       python（需 onnxruntime + torch 的环境） ch03/transformers_infer.py

④ 依赖
   包：torch、transformers，以及 accelerate（device_map="auto" 必需，缺了报
       `ValueError: ... requires accelerate`）。实测 torch 2.6.0+cpu / transformers 4.57.6 /
       accelerate 1.15.0，全在 venv-onnx。
   权重：任意 HF 格式因果语言模型目录。本机用 models/Qwen2.5-0.5B-Instruct
        （约 1GB，CPU 可跑）。不设 MODEL_NAME 则回落到 HF 仓库名 Qwen/Qwen2.5-0.5B-Instruct，
        那需要能连 huggingface.co（本机直连超时，见 ENV_MATRIX §6）。
   服务：无

⑤ 实测解释器
   实测解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / transformers 4.57.6 /
              accelerate 1.15.0）
   实测结果：✅ 跑通（rc=0，CPU，0.5B，stderr 为空；输出原文，未改写）
       请解释什么是人工智能？它有哪些应用场景？

       人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，涉及使机器能够执行通常需要人类智能的任务。这些任务包括学习、推理、感知、语言处理和问题解决等。

       人工智能的应用场景非常广泛，涵盖了从语音识别到自动驾驶汽车，再到医疗诊断、金融分析、自然语言处理等领域。以下是几个具体的例子：

       1. 语音识别：通过使用深度学习技术，计算机可以理解并识别人类的语音指令，这
   （以上为一次运行的 stdout 原文，采样解码 → 逐次不同；`max_new_tokens=100` 截断，stderr 为空）

⑥ 与书稿的差异
   1. 模型来源改为环境变量 MODEL_NAME（默认 Qwen/Qwen2.5-0.5B-Instruct），可指向本地目录；
      书稿写死作者机器上的绝对路径 /home/effective_gpt/model/tran_model/qwen/...
      —— 该路径的「qwen/ 子目录」在 modelscope 1.40.2 上并不存在。
   2. 其余逻辑（tokenizer / from_pretrained(device_map="auto") / tokenizer(prompt,
      return_tensors="pt") / generate(max_new_tokens=100) / decode(skip_special_tokens=True)）
      与书稿完全一致。
"""
import os

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- 示例 1：① 加载分词器和模型 ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # ---- 示例 2：② 准备输入 ----
    prompt = "请解释什么是人工智能？"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # "pt" = PyTorch 张量

    # ---- 示例 3：③ 生成回答 ----
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
