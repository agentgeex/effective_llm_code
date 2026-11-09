# -*- coding: utf-8 -*-
"""ch04/model_architectures.py —— 同一条 Transformer 理论下的三种模型构型

① 这是什么：用 transformers 的 AutoClasses 分别加载 BERT（判别式）/ GPT（自回归）/ T5（序列到序列）
   三种构型，演示「同一套 Transformer 骨架，换 head 与 attention mask 就是不同模型」。
② 对应书稿：第4章 §4.2.2 基于 Transformer 理论框架的不同模型构型
   BERT 段（AutoModelForSequenceClassification）/ GPT 段（AutoModelForCausalLM）/
（T5，AutoModelForSeq2SeqLM）。书稿把三段分开放，这里合到一个文件用命令行选段。
③ 怎么跑：
     V=python（需 onnxruntime + torch 的环境）
     $V ch04/model_architectures.py bert
     $V ch04/model_architectures.py gpt
     $V ch04/model_architectures.py t5
   本机已缓存三个模型时加 HF_HUB_OFFLINE=1 可完全离线跑（否则首次会去 huggingface.co 下载）。
   模型名可用环境变量替换：BERT_MODEL / GPT_MODEL / T5_MODEL（默认值即书稿里的三个名字）。
④ 依赖：torch + transformers（实测 venv-onnx：torch 2.6.0+cpu / transformers 4.57.6）；
   需要模型权重：bert-base-chinese 约 400MB、Qwen2.5-0.5B 约 1GB、uer/t5-small-chinese 约 240MB。
   不静默降级：加载不到模型就抛原始异常，由调用方判断。
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / transformers 4.57.6，
   真 import OK）， 三个子命令重跑（见文件末实测记录）。
   ⚠️ 系统 /usr/bin/python3 **不行**，事实如下：装了 transformers 4.52.4 但缺 regex，
     `import transformers` 实测报
         File ".../transformers/utils/args_doc.py", line 22, in <module>
             import regex as re
         ModuleNotFoundError: No module named 'regex'
     （`find_spec("transformers")` 是 True、`metadata.version` 也是 4.52.4，
       看得见 ≠ 导得进去 —— 判能不能用只认真 import。）
⑥ 与书稿的差异：
   1) 三个模型名改为可用环境变量覆盖（默认值就是书稿里那三个，行为不变）；
   2) T5 段剔除了 tokenizer 返回的 token_type_ids（见示例 3 注释），否则报 ValueError；
   3) 书稿三段是散块，本文件合成一个可执行入口。其余与书稿一致。
"""
import os
import sys


def bert_demo():
    """示例 1：AutoModelForSequenceClassification（BERT 类，判别式）"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = os.environ.get("BERT_MODEL", "bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    text = "这部电影的叙事结构非常精妙，值得反复回味。"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    print(f"预测类别: {predicted_class}")  # 0=负面, 1=正面


def gpt_demo():
    """示例 2：AutoModelForCausalLM（GPT 类，自回归生成）"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = os.environ.get("GPT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "请用一句话解释注意力机制："
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def t5_demo():
    """示例 3：AutoModelForSeq2SeqLM（T5 类，序列到序列）"""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = os.environ.get("T5_MODEL", "uer/t5-small-chinese-cluecorpussmall")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text = """
2025年，大模型应用正式进入"工业化落地"阶段。越来越多的企业不再满足于简单的对话式AI，
而是开始将大模型深度嵌入核心业务流程：从智能客服到自动化代码审查，从金融风控到医疗影像分析。
这一趋势对模型的可靠性、可控性和部署效率提出了前所未有的要求。
"""

    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    # T5 的 forward/generate 不接受 token_type_ids（tokenizer 却会返回它）
    # 不剔除就报 ValueError: The following `model_kwargs` are not used by the model: ['token_type_ids']
    inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    outputs = model.generate(**inputs, max_new_tokens=80)
    print(f"摘要: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    table = {"bert": bert_demo, "gpt": gpt_demo, "t5": t5_demo}
    if which not in table:
        print(__doc__)
        print("可选: " + " / ".join(table))
        sys.exit(2)
    table[which]()

# ---- 实测记录（本机 venv-onnx）----
# 解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / transformers 4.57.6）
# 前置：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1（三个模型都在本机 HF 缓存里）
#   ch04/model_architectures.py bert  → rc=0  输出「预测类别: 0」（分类头随机初始化，每次可能不同）
#   ch04/model_architectures.py gpt   → rc=0  生成「注意力机制是一种在深度神经网络中……」
#   ch04/model_architectures.py t5    → rc=0  输出「摘要: extra0 extra1 extra2 extra2」
#       ↑ 无意义是对的：uer/t5-small-chinese 未做摘要微调，不是代码错（README §5 有说明）
