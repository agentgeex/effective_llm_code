# -*- coding: utf-8 -*-
"""ch06 §6.5.1 —— vLLM 原生 Python API 用法合集（书稿对应小节 的互补实现）

① 这是什么：vLLM 的**原生 Python API** 四种用法——
   离线同步 `LLM.generate()`、`LLM.chat()`、`AsyncLLMEngine` 流式、异步 chat。
   不是 OpenAI 兼容 HTTP 客户端。

② 对应书稿：第 6 章 §6.5.1：书稿是用 OpenAI SDK 调
   已启动的 vLLM 服务（HTTP）；本文件是同一能力的另一条路径（进程内原生 API），
   两者互补，不是逐行对应。

③ 怎么跑（需自行安装 vLLM 与本地模型）：
       export MODEL_DIR=/path/to/your/Qwen3-14B
       python ch06/vllm_t.py
   文件里的 `local_model_path` 默认取 MODEL_DIR，未设置时用书稿同款占位路径。

④ 依赖：vllm、torch（CUDA 版）、transformers、一个本地模型目录。
   书稿示例是 Qwen3-14B（bf16 约 28GB）。

⑤ 实测解释器：**python（需 vllm 的环境）**（vllm 0.8.5 / torch 2.6.0+cu124 /
   transformers 4.51.3 / CUDA True）。
    真跑：
       cd <仓库根>
       MODEL_DIR=models/Qwen2.5-0.5B-Instruct \
         venv-vllm/bin/python ch06/vllm_t.py        # 注意：**不要**设 PYTHONPATH（仓库 vllm/ 会遮蔽）
   结果：rc=0，`Processed prompts: ... output: 346.39 toks/s`，真生成一段 ND 模式说明。
   ⚠️ 模型换成 0.5B 是因为 10G 卡放得下 14B bf16（28GB）不现实，不是代码问题。
   ⚠️ vllm 必须独环境 + 退版本：最新的 0.32.0 要 torch 2.13+cu130（本机驱动只到 CUDA 12.4），
      0.10.2 要 torch 2.8（无 cu124 轮子）。`0.8.5 + --torch-backend=cu124` 才能跑，
      且其 transformers 必须 `4.51.3`（4.52+ 删了 `all_special_tokens_extended`）。详见 ENV_MATRIX §8.8。

⑥ 与书稿的差异：
   1. 书稿用 OpenAI SDK 调 HTTP 服务；本文件是 vLLM 原生 API（见②）。
   2. `local_model_path` 改为读 `MODEL_DIR`，未设置时用书稿占位路径。
   3. 文件内 `AsyncLLMEngine.from_vllm_config(model=...)` / `await engine.stop()`
      两段带「假设与旧版兼容」注释，**未在真实 vLLM 上验证**（版本间 API 变动大）。
   4. `os.environ["VLLM_USE_V1"]="0"` 是原仓库为关 V1 引擎加的，书稿没有。
"""
import asyncio
from vllm import LLM,AsyncLLMEngine,SamplingParams
import time
import os
from transformers import AutoTokenizer

os.environ["VLLM_USE_V1"] = "0"

# 本地模型目录：优先读环境变量，未设置时用书稿占位路径
local_model_path = os.environ.get(
    "MODEL_DIR",
    "/path/to/your/motogpt/modle/tran_model/Qwen3-14B",
)


def  test_model_jr():

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model=local_model_path, task="generate")  # Name or path of your model
    output = llm.generate("Hello, my name is",sampling_params)
    print(output)



def test_sample_guanfang():
    prompts = [
        "你是谁？请用中文回答",
        "你会做什么？请用中文回答",
        "法国在哪里？请用中文回答",
        "人工智能的未来是什么？请用中文回答",
    ]

    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model=local_model_path)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


def test_sample():

    # prompts = [
    #     "你好，你是谁？",
    #     "法国的首都在哪里？",
    # ]

    prompts = [
        "你好，你叫什么名字",
        "The president of the United States is",
        "法国的首都在哪里",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

    # 初始化LLM，vLLM会自动加载分词器
    llm = LLM(model=local_model_path, trust_remote_code=True, max_model_len=4096)

    # 获取分词器
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

    # 使用分词器的 chat_template 属性来格式化 prompt
    formatted_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    print("Formatted prompts:", formatted_prompts)

    # 使用格式化后的 prompts 进行推理
    outputs = llm.generate(formatted_prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


def run_vllm_interface_local_model():
    """离线同步推理版本"""

    # 确保路径存在
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(f"当面目录没有发现模型文件: {local_model_path}")

    # --- 其他参数 ---
    gpu_memory_utilization = 0.8
    max_model_len = 2048
    trust_remote_code = True # 同样，请谨慎使用
    max_num_seqs = 32 # 这个参数更多是服务端的配置，LLM 直接推理不直接体现

    # 1. 初始化 LLM 对象，传入本地模型路径
    llm = LLM(
        model=local_model_path, # 直接使用本地路径
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        # tensor_parallel_size=... # 如果有多卡
    )

    # 采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

    # 准备 prompts
    #prompts = ["The capital of China is"]
    prompts = [
                    "你是谁？请用中文回答",
                    "你会做什么？请用中文回答",
                    "法国在哪里？请用中文回答",
                    "人工智能的未来是什么？请用中文回答",
                ]

    # 2. 执行推理
    print(f"Starting inference with local model from: {local_model_path}")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # 3. 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 60)

    print(f"Inference completed in {end_time - start_time:.2f} seconds.")


def run_vllm_interface_local_model_chat():
    """处理对话形式的提示词"""
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(f"Local model directory not found: {local_model_path}")

    # --- 其他参数 ---
    gpu_memory_utilization = 0.8
    max_model_len = 2048
    trust_remote_code = True # 同样，请谨慎使用
    max_num_seqs = 32 # 这个参数更多是服务端的配置，LLM 直接推理不直接体现

    # 1. 初始化 LLM 对象，传入本地模型路径
    llm = LLM(
        model=local_model_path, # 直接使用本地路径
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        # tensor_parallel_size=... # 如果有多卡
    )

    # 采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4000)

    # 准备 prompts
    conversation = [
        {"role": "system", "content": "你是资深的大模型技术顾问，熟悉 Transformer 架构、训练与推理的各种细节，回答要具体、可操作。请用中文回答。"},
        # {"role": "user", "content": "Transformer 的编码器和解码器在结构上有什么不同？各自适合什么任务？"},
        # {"role": "user", "content": "多头注意力里的「多头」到底解决了什么问题？和单头比强在哪？"},
        {"role": "user", "content": "位置编码为什么必须加？只用自注意力、不加位置信息会出什么问题？"},

    ]

    print("Starting multi-turn chat inference with LLM.chat()...")

    start_time = time.time()

    # LLM.chat() 方法返回一个包含 RequestOutput 对象的列表
    outputs = llm.chat(
        messages=conversation,
        sampling_params=sampling_params,
        # use_tqdm=True # 可以选择显示进度条
    )

    end_time = time.time()

    if outputs:
        # 模型的回答是 conversation 列表中的最后一个 assistant 消息
        # vLLM 的 output.outputs[0].text 包含了模型的最新回答
        # 注意：LLM.chat() 返回的 output.prompt 会是格式化后的完整 prompt
        # output.outputs[0].text 是模型生成的回答
        print(f"Full formatted prompt:\n{outputs[0].prompt}\n")
        print(f"Model's latest response: {outputs[0].outputs[0].text!r}")
    else:
        print("No output received.")




##############################  async ####################################
##########################################################################



async def run_async_inference_local_model():
    # --- 模型路径设置 ---
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(f"Local model directory not found: {local_model_path}")

    # --- 其他参数 ---
    gpu_memory_utilization = 0.8
    max_model_len = 2048
    trust_remote_code = True
    max_num_seqs = 32

    # 1. 使用 V1 API 的方式初始化引擎
    # AsyncLLMEngine.from_vllm_config() 是推荐的初始化方法
    engine = AsyncLLMEngine.from_vllm_config(
        model=local_model_path,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        worker_use_ray=False,
        engine_use_ray=False,
    )

    # 注意：engine.generate() 的参数格式可能也发生了变化，这里使用新的方式
    # 在 V1 中，generate() 方法的参数通常与旧版本不同
    # 最新的API用法可以参考vLLM官方文档
    # 这里我们假设新的generate方法与旧版兼容

    # 2. 定义 prompt 和采样参数
    prompt = "Explain the concept of vLLM's PagedAttention."
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=150)
    request_id = "req-local-async"

    # 3. 提交请求并流式获取结果
    print(f"Starting async inference with local model from: {local_model_path}")
    generated_text = ""
    try:
        # 使用engine.generate()方法
        async for output_chunk in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            generated_text += output_chunk.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Full Generated text: {generated_text!r}")
    except Exception as e:
        print(f"An error occurred during async inference: {e}")
    finally:
        # 确保在程序结束时停止引擎
        await engine.stop()


async def run_async_chat():
    # --- 模型路径 ---
    #model_path = "/path/to/your/motogpt/modle/tran_model/DeepSeek-V2-Lite"
    # 或者 model_path = "facebook/opt-125m"

    # --- 初始化 AsyncLLMEngine ---
    engine = AsyncLLMEngine(
        model=local_model_path,
        worker_use_ray=False,
        engine_use_ray=False,
    )

    # --- 定义采样参数 ---
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

    # --- 构建多轮对话历史 ---
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is the capital of Japan?"},
        {"role": "assistant", "content": "The capital of Japan is Tokyo."},
        {"role": "user", "content": "What is its population?"}
    ]
    request_id = "chat-req-1"

    # --- 调用 chat() 方法进行异步推理 ---
    print("Starting async multi-turn chat inference with AsyncLLMEngine.chat()...")
    generated_text = ""
    try:
        # engine.chat() 返回一个异步迭代器，可以流式获取结果
        async for output_chunk in engine.chat(
            messages=conversation,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            generated_text += output_chunk.outputs[0].text
            # print(f"Received chunk: {output_chunk.outputs[0].text!r}") # 流式打印
        print(f"Full formatted prompt:\n{output_chunk.prompt}\n") # prompt 属性在最后一个 chunk 中
        print(f"Model's latest response: {generated_text!r}")
    except Exception as e:
        print(f"An error occurred during async chat inference: {e}")
    finally:
        await engine.stop()



if __name__ == "__main__":


    # test_model_jr()
    # test_sample_guanfang()
    # test_sample()
    # run_vllm_interface_local_model()
    run_vllm_interface_local_model_chat()
    # asyncio.run(run_async_inference_local_model())