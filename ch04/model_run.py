"""ch04/model_run.py —— 用 AutoClasses 加载模型并对话生成的完整示例

① 这是什么：用 AutoTokenizer + AutoModelForCausalLM 加载本地模型，走 apply_chat_template
   构造 prompt，generate 生成；附带一份「模型属性检测」（EOS token / chat_template / 特殊 token）。
   文件里还保留一个 pipeline 写法，默认不启用。
② 对应书稿：第4章 §4.3.1 大模型技术栈：基础框架（书稿是 8 行精简片段：
   from_pretrained → tokenizer → generate → decode）。本文件是同一调用的完整实现，
   书稿只给了片段，仓库这份是完整实现。
③ 怎么跑：
     MODEL_DIR=models/Qwen2.5-0.5B-Instruct \\
       python（需 onnxruntime + torch 的环境） ch04/model_run.py
   不设 MODEL_DIR 时会用书稿占位符 /path/to/your/...，加载必然失败。
④ 依赖：torch、transformers、accelerate（`device_map="auto"` 缺它报
   ValueError: ... requires accelerate）；需模型权重目录；无外部服务。
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）
   （torch 2.6.0+cpu / transformers 4.57.6 / accelerate 1.15.0）， 实跑（见文件末记录）。
⑥ 与书稿的差异：
   1) 模型路径改为读环境变量 MODEL_DIR（默认保留书稿占位符，不猜路径）；
   2) 书稿片段只用 generate 默认参数，本文件额外做了 chat template、终止符探测、
      模型属性打印；
   3) 多了一个未启用的 pipeline 实现（书稿未涉及）。生成主逻辑与书稿一致。
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
model_path = "/path/to/your/motogpt/modle/tran_model/DeepSeek-V2-Lite"  # 这个模型通过本类的方法没有加载成功
# model_path = "/path/to/your/motogpt/modle/tran_model/DeepSeek-V2-Lite-AWQ"

# os.environ['HF_HOME'] = '/path/to/your/motogpt/modle/tran_model'

# 模型目录：优先读环境变量 MODEL_DIR；未设置时保留书稿占位符（照抄必然加载失败，属预期）
model_path = os.environ.get("MODEL_DIR", model_path)

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



# ---- 实测记录（本机 venv-onnx）----
# 解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / transformers 4.57.6 / accelerate 1.15.0）
# 命令：MODEL_DIR=models/Qwen2.5-0.5B-Instruct ch04/model_run.py
# rc=0，生成「我是来自阿里云的大规模语言模型，我叫通义千问。」
# 执行耗时：1.13 秒（CPU，bfloat16）
