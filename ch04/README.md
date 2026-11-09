# 第 4 章 大模型原理和技术实现

## 这章讲什么

拆开看：注意力机制到底在算什么、同一个 Transformer 理论怎么长出三种构型（encoder-only / decoder-only / encoder-decoder）、模型文件在磁盘上是什么格式（safetensors / ONNX）、以及一次完整训练长什么样。

这一章的代码量是全书中段最重的一章 —— **每一段都能在 CPU 上真跑**，不需要 GPU、不需要 API key。目的是让你用手摸到原理，而不是只看公式。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`transformer_min.py`](transformer_min.py) | 手写单头自注意力 + 多头注意力，最小可运行的 Transformer 核心 | **A 可真跑** |
| [`model_architectures.py`](model_architectures.py) | 同样的理论，三种构型各跑一遍：BERT 做分类、GPT 做生成、T5 做 seq2seq | **A 可真跑**（三个子命令都 rc=0；见下方「输出不对不是 bug」） |
| [`model_run.py`](model_run.py) | 用 AutoClasses 加载模型并对话生成的完整示例 | **A 可真跑** |
| [`model_parser_c.py`](model_parser_c.py) | 用 `print(model)` 直视模型的逻辑结构：层是怎么堆起来的 | **A 可真跑** |
| [`safetensors_read.py`](safetensors_read.py) | safetensors 零拷贝（mmap）读取最小示例 | **A 可真跑**（临时文件自动清理，不残留） |
| [`onnx_export_min.py`](onnx_export_min.py) | PyTorch → ONNX → ONNX Runtime 全链路最小示例 | **A 可真跑** |
| [`onnx_m.py`](onnx_m.py) | 用 optimum 的 `ORTModelForCausalLM` 加载已导出的 ONNX 模型 | **B 要条件**（需要一个已导出的 ONNX 目录，约 2.4GB） |
| [`unsloth_wt1.py`](unsloth_wt1.py) | Unsloth + LoRA 微调完整脚本 | **B 要条件**（要 GPU、要 1.3B 基座权重、要一整套训练依赖，缺一不可） |

## 怎么跑

全章不需要网络（模型权重除外），在仓库根执行：

```bash
# 1. 先看注意力到底在算什么 —— 零依赖外的东西，最快
python3 ch04/transformer_min.py

# 2. 三种模型构型
python3 ch04/model_architectures.py bert
python3 ch04/model_architectures.py gpt
python3 ch04/model_architectures.py t5

# 3. 看清模型内部结构
python3 ch04/model_parser_c.py

# 4. 加载模型并生成
MODEL_DIR=/你的/模型目录 python3 ch04/model_run.py

# 5. 模型文件格式
python3 ch04/safetensors_read.py
python3 ch04/onnx_export_min.py
python3 ch04/onnx_m.py          # 需要先有一次 ONNX 导出产物
```

模型名不是写死的：`model_architectures.py` 认 `BERT_MODEL` / `GPT_MODEL` / `T5_MODEL` 三个环境变量，`model_run.py` 认 `MODEL_DIR`。默认值是书里用的那几个模型。

## 依赖与前置

| 依赖 | 谁需要 |
|---|---|
| `torch`（CPU 版就够） | 除 `safetensors_read.py` 外的全部 |
| `transformers` | `model_architectures.py` / `model_run.py` / `model_parser_c.py` / `onnx_m.py` |
| `accelerate` | `model_run.py`（`device_map="auto"` 要它） |
| `safetensors` | `safetensors_read.py` |
| `onnx` + `onnxruntime` | `onnx_export_min.py` |
| `optimum` + `optimum-onnx` | `onnx_m.py` |
| 模型权重 | `bert-base-chinese`（约 400MB）、`Qwen2.5-0.5B-Instruct`（约 1GB）、`t5-small-chinese`（约 240MB）|

**不需要 GPU，不需要 API key。** 只有 `unsloth_wt1.py` 要 GPU。

## 常见报错

**`ModuleNotFoundError: No module named 'regex'`（`transformers` 导入失败）**
系统自带的旧版 `transformers` 可能缺依赖。用虚拟环境并 `pip install "transformers>=4.40"`，不要依赖系统包。

**`ValueError: ... requires accelerate`**
`device_map="auto"` 需要 `accelerate`，只装 torch + transformers 不够。

**`t5` 的输出是一串 `extra0 extra1 extra2`**
**这不是 bug。** `uer/t5-small-chinese` 没做过摘要微调，它只是被拿来证明「encoder-decoder 这条路能跑通」。想看到有意义的摘要，要么换成微调过的模型，要么自己训 —— 这正是第 11 章「微调」那一节的事。

**`unsloth_wt1.py` 一上来就 `ModuleNotFoundError`**
这个脚本的依赖链最长（`unsloth` / `trl` / `datasets` / `psutil` / `regex` / `pandas`），而且需要 1.3B 基座权重和显存。少一样都跑不了 —— 书里这一节的重点是**看懂微调脚本的骨架**，不是让你在本机训出一个模型。

**`onnx_m.py` 报找不到 ONNX 模型**
要先用 `onnx_export_min.py` 导出一份，并把 `onnx_m.py` 里的路径指过去。导出产物 2GB 以上，仓库里不提供。
