# -*- coding: utf-8 -*-
"""ch06 §6.2.4 —— 用 transformers 加载模型并生成

① 这是什么：AutoTokenizer + AutoModelForCausalLM 加载本地模型，用
   apply_chat_template 渲染对话模板，再 model.generate() 生成回答的完整脚本。
   类封装版：`Model_tl` 负责加载 + 属性自检，`tl_transformers_autoclassses()` 负责生成。

② 对应书稿：第 6 章 §6.2.4「加载 + 对话模板 + generate」。
   注意：同节讲 pipeline 文本分类那一段不是本文件；

③ 怎么跑：
       export MODEL_DIR=models/Qwen2.5-0.5B-Instruct
       python（需 onnxruntime + torch 的环境） ch06/model_run.py
   只验证加载、不跑生成：`MAX_NEW_TOKENS=0`（仍然会执行加载与 chat_template 检查）

④ 依赖：torch、transformers；一个本地模型目录。
   本机可用的最小模型：`models/Qwen2.5-0.5B-Instruct`（约 1GB）。
   不需要外网，不读 HF 缓存以外的任何服务。

⑤ 实测解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / transformers 4.57.6）
   实测日期：（见文件末尾「实测记录」）

⑥ 与书稿的差异：
   1. 书稿把 `MODEL_ID` 写死为 `Qwen/Qwen2.5-0.5B-Instruct`（仓库名，需要联网下载）；
      本文件改为读 `MODEL_DIR`（本地目录），未设置时用书稿同款仓库名兜底。
   2. 书稿是线性脚本；本文件是**类封装**（原仓库写法），多了 `_check_model_properties()`
      打印 EOS / chat_template / special tokens，便于排查模板问题。
   3. `max_new_tokens` 改为读 `MAX_NEW_TOKENS`（默认 256，与书稿一致），
      方便在 CPU 上做小步验证。
   4. 生成逻辑本身（apply_chat_template / 只看新 token / skip_special_tokens）与书稿一致。
"""
import transformers
import torch
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 模型目录：优先读环境变量，未设置时用书稿同款仓库名（需要联网）
# 本地已下载的示例：models/Qwen2.5-0.5B-Instruct
model_path = os.environ.get("MODEL_DIR", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

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
            max_new_tokens=MAX_NEW_TOKENS,
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
    if MAX_NEW_TOKENS > 0:
        model_tl_b.tl_transformers_autoclassses()
    else:
        print("[跳过生成] MAX_NEW_TOKENS=0，只验证加载与模板")

    d_end = datetime.now()
    d_diff = d_end - d_start
    d_diff_second = d_diff.total_seconds()
    print("\n")
    print(f"[] ==== {d_end.strftime('%Y-%m-%d %H:%M:%S')} 执行耗时 {d_diff_second} \n")
    pass




# ─────────────────────────────────────────────────────────
# 实测记录
#   解释器：python（需 onnxruntime + torch 的环境）
#           torch 2.6.0+cpu / transformers 4.57.6 / Python 3.11.15
#   模型：models/Qwen2.5-0.5B-Instruct
#
#   $ MODEL_DIR=... MAX_NEW_TOKENS=0 python ch06/model_run.py
#     → rc=0：EOS '<|im_end|>'、chat_template 检测通过、14 个 special token 列出
#   $ MODEL_DIR=... MAX_NEW_TOKENS=16 python ch06/model_run.py
#     → rc=0，耗时 1.1s，输出：我是阿里云研发的超大规模语言模型，我叫通义千问
#
#   说明：CPU 上跑 bfloat16，只验证「加载 + 模板 + 生成」链路；长文本速度不代表 GPU 表现。
# ─────────────────────────────────────────────────────────
