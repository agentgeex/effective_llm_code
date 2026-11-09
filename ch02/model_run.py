# -*- coding: utf-8 -*-
"""ch02 §2.4.2 网络阻力：消灭"几 KB/s 的等待"

书稿片段只给 from_pretrained 的两种写法（本地目录 vs HF 仓库名）；本文件是仓库里
同类的完整实现：AutoTokenizer + AutoModelForCausalLM + bf16 + device_map="auto"，
另含 chat_template / special token 检测与两种推理方法。

    MODEL_PATH=/你的/本地/模型目录 python（需 onnxruntime + torch 的环境） ch02/model_run.py

依赖：torch、transformers（实测 torch 2.6.0+cpu / transformers 4.57.6，venv-onnx）。
    模型权重：任意 HF 格式因果语言模型目录；本机实测用
    models/Qwen2.5-0.5B-Instruct（约 1GB，CPU 可跑）。
    实测解释器：python（需 onnxruntime + torch 的环境）

实测状态：✅ 跑通
    命令：MODEL_PATH=models/Qwen2.5-0.5B-Instruct \
          python（需 onnxruntime + torch 的环境） ch02/model_run.py
    关键输出（原文）：
        EOS Token: '<|im_end|>' (ID: 151645)
        --- 检测结束 ---
        我是来自阿里云的大规模语言模型，我叫通义千问。
    退出码：0；总耗时 5.57s（CPU，0.5B）。
    备注：首行告警 `torch_dtype is deprecated! Use dtype instead!` 来自 transformers 4.57.6，
          代码仍可运行；书稿未涉及该参数。

与书稿差异：
  1. `model_path` 默认值改为读环境变量 MODEL_PATH，未设置时保留原占位符；
     书稿用的是作者机器上的绝对路径 `/data/models/Qwen2.5-7B-Instruct`。
  2. 加载与生成逻辑与书稿的 `from_pretrained(..., device_map="auto")` 一致
     （另加 torch_dtype=torch.bfloat16）。
  3. 书稿还演示了「直接给 HF 仓库名让框架联网下载」，本文件未保留该写法；
     需要时把 MODEL_PATH 设为 `Qwen/Qwen2.5-7B-Instruct` 即可走同一条路径。
"""
import transformers
import torch
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 切换为你下载的模型文件目录, 这里的demo是Llama-3-8B-Instruct
# 如果是其他模型，比如qwen，chatglm，请使用其对应的官方demo
# model_path = "/path/to/your/motogpt/modle/tran_model/Meta-Llama-3-8B-Instruct"
# model_path = "/path/to/your/motogpt/modle/tran_model/Qwen1.5-4B-Chat"
# model_path = "/path/to/your/motogpt/modle/tran_model/DeepSeek-R1-Distill-Qwen-7B"
model_path = os.environ.get("MODEL_PATH", "/path/to/your/motogpt/modle/tran_model/DeepSeek-V2-Lite")  # 原占位符；本机实测用 MODEL_PATH 指向 Qwen2.5-0.5B-Instruct
# model_path = "/path/to/your/motogpt/modle/tran_model/DeepSeek-V2-Lite-AWQ"

# os.environ['HF_HOME'] = '/path/to/your/motogpt/modle/tran_model'

class Model_tl():

    def __init__(self,model_path=model_path):
        self.curr_model_path = model_path
        # self._init_model_llama_pipline()
        self._init_model_auto()
        self._check_model_properties() # 在这里调用新的检测方法

    def _init_model_llama_pipline(self):
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.curr_model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.curr_transformers_pipline = pipeline

    def _init_model_auto(self):
        # 使用 AutoTokenizer 和 AutoModelForCausalLM 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.curr_model_path,
            trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.curr_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False
        )



    def tl_transformers_autoclassses(self):

        """
        通过 AutoClasses 方式调用 LLaMA 格式的模型
        """
        messages = [
            {"role": "system", "content": "下面的问题请用中文回答!"},
            {"role": "user", "content": "你是谁"},
        ]

        # 同样使用 apply_chat_template 构建 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 这一步，不同的模型可能会有不一样的地方
        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        terminators = []

        if self.tokenizer.eos_token_id is not None:
            terminators.append(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            terminators.append(self.tokenizer.pad_token_id)

        if "<|eot_id|>" in self.tokenizer.vocab: # 这里可以使用 .vocab 来检查
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        if "<|endoftext|>" in self.tokenizer.vocab: # 这里可以使用 .vocab 来检查
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|endoftext|>"))

        if "<｜end▁of▁sentence｜>" in self.tokenizer.vocab: # 这里可以使用 .vocab 来检查
            terminators.append(self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>"))



        # 将 prompt 转换为 token ID
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 调用模型的 generate 方法进行推理
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        # 解码生成的 token
        response = outputs[0][len(inputs["input_ids"][0]):]
        print(self.tokenizer.decode(response, skip_special_tokens=True))

    def tl_transfromers_pipeline_llm(self):
        """通过transfromers的pipeline方式调用LLaMA格式的模型

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        # pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model_path,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device_map="auto",
        # )

        messages = [
            {"role": "system", "content": "下面的问题请用中文回答!"},
            {"role": "user", "content": "你是谁"},
        ]

        prompt = self.curr_transformers_pipline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 这一步，不同的模型可能会有不一样的地方
        # terminators = [
        #     self.curr_transformers_pipline.tokenizer.eos_token_id,
        #     self.curr_transformers_pipline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.vocab: # 这里可以使用 .vocab 来检查
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        outputs = self.curr_transformers_pipline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][len(prompt):])

    def _check_model_properties(self):
        """
        一个通用的检测方法，打印出模型的关键属性。
        """
        print("--- 模型属性检测 ---")

        # 打印 EOS token
        if self.tokenizer.eos_token:
            print(f"EOS Token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")

        # 检查是否有 chat_template
        if self.tokenizer.chat_template:
            print("\n发现自定义聊天模板（chat_template）！")
            print("这是模型建议的对话格式，请确保你的消息格式与之匹配。")

            # 打印一个示例
            messages = [{"role": "user", "content": "你好，请用中文回答。"}]
            example_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            print(f"示例 prompt:\n---\n{example_prompt}\n---")
        else:
            print("\n未发现自定义聊天模板。")

        # 检查是否有额外的 special tokens
        additional_special_tokens = self.tokenizer.all_special_tokens
        if additional_special_tokens:
            print(f"\n模型包含额外特殊 token: {additional_special_tokens}")
            print("这些 token 可能会影响模型行为或对话格式。")

        print("--- 检测结束 ---")



if __name__ == '__main__':

    d_start = datetime.now()
    print(f"[] ==== {d_start.strftime('%Y-%m-%d %H:%M:%S')} \n")

    model_tl_b = Model_tl()
    # model_tl_b.tl_transfromers_pipeline_llm()
    model_tl_b.tl_transformers_autoclassses()

    d_end = datetime.now()
    d_diff = d_end - d_start
    d_diff_second = d_diff.total_seconds()
    print("\n")
    print(f"[] ==== {d_end.strftime('%Y-%m-%d %H:%M:%S')} 执行耗时 {d_diff_second} \n")
    pass


