# 第 6 章 大模型技术生态

## 这章讲什么

生态就是「除了模型本身，你还要装什么」。这一章一口气过完训练侧（PyTorch / Accelerate / unsloth / DeepSpeed）、推理侧（vLLM）、编排侧（DSPy）、检索侧（RAG 的 HyDE 与 RRF）、以及多模态和云端 SDK。

章节长、组件多，所以本目录**只放能独立跑的最小示例** —— 每个文件都在回答一个问题：「这个框架到底解决什么，最少要写几行」。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`torch_env_check.py`](torch_env_check.py) | PyTorch 环境自检：版本、CUDA 可用性、显卡型号与显存 | **A 可真跑**（两种环境都验过：有卡报型号显存，CPU 版给提示） |
| [`transformers_config.py`](transformers_config.py) | 只读 `config.json`、按 config 搭骨架，不加载权重 | **A 可真跑** |
| [`model_run.py`](model_run.py) | 用 transformers 加载模型并生成 | **A 可真跑** |
| [`accelerate_min.py`](accelerate_min.py) | Accelerate 的最小可跑版：**同一份训练脚本，改 5 行就从单卡到多卡** | **A 可真跑**（单卡 / `accelerate launch` / CPU / fp16 四种跑法全 rc=0） |
| [`rag_hyde_rrf.py`](rag_hyde_rrf.py) | RAG 2.0 的两台手术：HyDE（用假设答案去检索）+ RRF（多路结果融合） | **A 可真跑**（零第三方依赖，内置 mock，不需要网络和模型） |
| [`troch_train_c_moe_v1.py`](troch_train_c_moe_v1.py) | 用 PyTorch 手搭一个 MoE 解码器并训练 | **A 可真跑**（小步验证：200 行语料 / 1 epoch / loss 0.16；要 `torchtext`） |
| [`dspy_min.py`](dspy_min.py) | DSPy 最小可跑版：**不写 Prompt，只声明输入输出** | **A 可真跑**（离线桩 LM；接真模型要 key，未验证） |
| [`openai_test.py`](openai_test.py) | OpenAI SDK 的两种最小用法：普通对话 + tools 两轮往返 | **B 要条件**（缺 key 时按设计退出） |
| [`gemini_live_api.py`](gemini_live_api.py) | Gemini Live 的双向流式接口，重点是「打断」语义 | **C 示意**（自检能跑，真连服务要 key + 音频设备） |
| [`unsloth_wt1.py`](unsloth_wt1.py) | unsloth 4bit LoRA 微调完整流程 | **B 要条件**（依赖链 + 基座权重 + 显存，三样缺一不可） |
| [`vllm_t.py`](vllm_t.py) | vLLM 原生 Python API 用法合集 | **B 要条件**（要装 vLLM + 一个能放进显存的模型） |

## 怎么跑

建议按「零依赖 → 要框架 → 要 key」的顺序：

```bash
# 第一组：零第三方依赖，最先跑，先确认思路对不对
python3 ch06/rag_hyde_rrf.py

# 第二组：要 torch / transformers
python3 ch06/torch_env_check.py
python3 ch06/transformers_config.py
MODEL_DIR=/你的/模型目录 python3 ch06/model_run.py

# 第三组：训练与编排
python3 ch06/accelerate_min.py                 # 直接跑 = 单进程
accelerate launch ch06/accelerate_min.py       # 多进程（cwd 必须是仓库根！）
python3 ch06/dspy_min.py

# 第四组：要 key / 要显存
export OPENAI_API_KEY=你的key
python3 ch06/openai_test.py
```

`accelerate launch` 那一条有个坑，见下面「常见报错」。

## 依赖与前置

| 依赖 | 谁需要 | 说明 |
|---|---|---|
| 无 | `rag_hyde_rrf.py` | 只用标准库，任何 Python 3.8+ 都能跑 |
| `torch` | 环境自检、config、model_run、accelerate、MoE | CPU 版即可，`torch_env_check` 会告诉你有没有卡 |
| `transformers` | config / model_run | 建议 ≥4.40 |
| `accelerate` | `accelerate_min.py` | 版本 1.x |
| `torchtext` | `troch_train_c_moe_v1.py` | **已停止维护**，只支持到较老的 torch；这是它跑起来要先搭专用环境的原因 |
| `dspy` | `dspy_min.py` | 本仓库实测 2.6.x |
| `openai` | `openai_test.py` | 本仓库实测 3.3.0 |
| 模型权重 | model_run / vllm_t / unsloth | 0.5B 级模型约 1GB |
| GPU | unsloth / vllm | 10GB 显存能跑 0.5B~1.3B 级 |

## 常见报错

**`accelerate launch` 报 `ModuleNotFoundError` 找不到同目录的模块**
`accelerate launch` 会用子进程重新 import 你的脚本，而 `sys.path` 里没有脚本所在目录。**cwd 必须是仓库根**（`cd` 到仓库根再执行上面的命令）。在 `/tmp` 或别的目录下跑必失败，这不是代码问题。

**`import vllm` 成功了，但一用就 `ModuleNotFoundError`**
陷阱：仓库里如果有个叫 `vllm` 的**空目录**，Python 3 会把它当成「命名空间包」导入成功 —— 假象。换到别的目录（比如 `/tmp`）再 `import vllm` 就会如实报错。判断标准：`python3 -c "import vllm; print(vllm.__file__)"`，看路径到底指向哪。

**`ModuleNotFoundError: No module named 'regex'`（导入 transformers 时）**
系统自带的旧版 transformers 缺依赖。用虚拟环境装新版，别用系统包。

**`ModuleNotFoundError: No module named 'psutil'`（`import unsloth` 时）**
unsloth 的间接依赖没装全。它的依赖链很长（`psutil` / `regex` / `pandas` / `trl` / `datasets`），少一个就断。

**`ModuleNotFoundError: No module named 'torchtext'` 或版本不兼容**
`torchtext` 已停止维护，只和特定版本的 `torch` 搭配。要么照它的兼容表锁版本，要么跳过这个文件 —— 它演示的是 MoE 的结构，不是训练效果。

**DSPy 的解析错误看起来「跟你写的不一样」**
`dspy_min.py` 里如果字段名和断言不一致，报错信息可能被适配器层「兜」了一层：先试结构化适配器，失败就退回 JSON 适配器，最后报出来的错可能指向后一层。排查顺序是：先看你的 Signature 字段名，再看具体抛错的那个适配器。这是本书 §6.7.1 专门讲的一个坑。

**`缺少环境变量 OPENAI_API_KEY`（退出码 2）**
设计如此 —— 没有 key 就不发请求，也不会假装跑通。默认端点已经是官方 `https://api.openai.com/v1`、默认模型 `gpt-4o-mini`；要接别的服务，`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 必须成对改（见 `ch02/README.md` 的那段例子，同一套写法）。
