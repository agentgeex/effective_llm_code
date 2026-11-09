# 第 2 章 大模型实战环境准备

## 这章讲什么

把「跑大模型」需要的东西装齐：Python 环境、PyTorch、CUDA、模型文件、在线 API。这一章的重点不是学会什么新概念，而是**让你的机器具备跑后面所有章节的能力**，以及认识到「下载模型」这件事为什么会卡、卡在哪。

装完这一章，后面 13 章才有条件跑。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`model_run.py`](model_run.py) | 用 `transformers` 从本地目录或 HF 仓库名加载一个因果语言模型，检测 chat 模板，然后对话生成 | **A 可真跑**（本机 Ubuntu + RTX 3080 实跑 rc=0，CPU 也能跑） |
| [`openai_test.py`](openai_test.py) | 用 OpenAI SDK 调在线推理服务：类封装 + function calling 两段 | **B 要条件**（缺 `OPENAI_API_KEY` 时脚本直接报缺哪个变量并退出，不会假跑） |

## 怎么跑

先跑能离线那个，它验证「模型下下来了、框架认得、能生成」：

```bash
# 需要先备好一个 HF 格式的模型目录，或者设成 HF 仓库名
MODEL_PATH=/你的/本地/模型目录 python3 ch02/model_run.py
```

手上没有模型？最小的选择是 `Qwen/Qwen2.5-0.5B-Instruct`（约 1GB，CPU 能跑）：

```bash
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct python3 ch02/model_run.py
```

再跑要 key 那个：

```bash
export OPENAI_API_KEY=你的key
# 如果用的不是默认端点，再设一个：
# export OPENAI_BASE_URL=https://api.deepseek.com/v1
python3 ch02/openai_test.py
```

## 依赖与前置

| 跑哪个 | 需要什么 |
|---|---|
| `model_run.py` | `torch` + `transformers` + `accelerate`；一个 HF 格式的因果语言模型目录或仓库名 |
| `openai_test.py` | `openai` SDK（本仓库实测 3.3.0）；一个可用的 API key |

显存：`model_run.py` 用 `device_map="auto"`，有卡就上卡、没卡就上 CPU。0.5B 模型 1GB 内存够用；更大的模型按「参数量 × 2 字节」粗估。

## 常见报错

**`ValueError: ... requires accelerate`**
`device_map="auto"` 是 `accelerate` 提供的功能，只装 `torch` + `transformers` 不够。`pip install accelerate` 即可。

**`缺少环境变量 OPENAI_API_KEY`（退出码 2）**
不是代码坏了，是设计如此：宁可明确告诉你缺什么，也不假装调通。设上环境变量再跑。

**`ImportError: No module named 'transformers'`**
装的是别的解释器的包。确认 `which python3` 和你 `pip install` 用的是同一个 —— 这是新手最常踩的一个坑。

**在线调用连不上，或报 `model_not_found`（404）**
两步走。先确认 `OPENAI_BASE_URL` 指对了 —— 脚本默认走官方 `https://api.openai.com/v1`。
再确认 `OPENAI_MODEL` 和端点对得上：每个服务的模型名都不一样，官方端点不认 `gemini-2.5-flash` 这类网关模型名，反之亦然。两个变量要一起改：

```bash
# 官方
export OPENAI_API_KEY=sk-...
python3 ch02/openai_test.py                      # 默认 https://api.openai.com/v1 + gpt-4o-mini

# DeepSeek 官方（同一份代码，只换两个变量）
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export OPENAI_MODEL=deepseek-chat

# 本地 vLLM / Ollama 的兼容层
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=<你起服务时 --served-model-name 写的名字>
```

**模型下载卡住或超时**
不是代码问题。国内网络访问 HuggingFace 通常要走镜像（`HF_ENDPOINT`）或事先手动下载。模型文件放在哪、目录结构长什么样，见本书 §2.3。
