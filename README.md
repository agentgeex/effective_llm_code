# 《有效的大模型》配套代码

[English](README.en.md) | **简体中文**

书里每一章讲到的技术，这里都有一份能跑（或明确告诉你为什么跑不了）的代码。
代码按书中章节分目录：`ch01/` – `ch14/`。
**章节 ↔ 文件对照表在 [INDEX.md](INDEX.md)** —— 按章节查、按文件查，都看那一份。

> 这里没有演示视频，没有亮眼的跑分，也没有「一键运行」。有的是：**每条结论都对应一份能复现的代码，跑不通就写跑不通。**

---

## 一、这是什么

这是《有效的大模型》一书的示例代码仓库。写作时的一个基本判断是：

> 读者卡住的地方，几乎从来不是「不知道 API 长什么样」，而是「按书上抄下来就是不work」。

所以这个仓库的取舍是**宁可少，不骗人**：

- **多数示例文件头部有六行固定结构的注释**（82 个 `.py` 里 50 个带这套 ①–⑥ 编号；`common/` 下那 21 个是公共模块，本来就不算示例。还有 11 个示例没补齐，列在第 6 节）：这是干什么的、对应书哪一节、怎么跑、依赖什么、用什么解释器实测过、和书里有什么差异。
- 跑不了的示例不会被删掉，而是标清楚**缺什么**（显存、API key、外部服务、模型权重），并给出降级路径。
- 示例代码里出现的参数、阈值、数据都是**教学示意**，不是生产配置，更不是操作规程。

## 二、怎么开始（三步跑通第一个例子）

**不需要 API key、不需要 GPU** 的第一个例子：

```bash
# 1) 第一个例子：不装任何包、不用 key、不用 GPU（跑约 30 秒，它真的在等 30 秒）
python3 ch10/concurrency_vs_serial.py

# 2) 一条命令装完全套环境（首次约 11 GB / 516 个包）
uv sync --all-extras

# 3) 手写自注意力（CPU 就能跑，大环境里已经有 torch）
.venv/bin/python ch04/transformer_min.py
```

三步各自在干什么：

1. **第 1 步零依赖。** 它只用标准库 `asyncio` / `time`，所以不用等任何安装 —— 先确认「这个仓库的代码在我机器上真的跑得动」，再谈装环境。它会把 100 张工单分别用串行、并发 10、全并发跑一遍，直接把「单任务延迟」和「整体吞吐量」的差别量化出来。
2. **第 2 步是一条命令，不是一张表。** `uv sync --all-extras` 把 `pyproject.toml` 里所有分组一次装完 —— torch / transformers / vllm / chromadb / fastmcp / dspy / crewai / langgraph / 火山 SDK ……共 516 个包、约 11 GB。**你不需要判断「这个例子要哪一组」。**
   代价先说清：首次下载量大；而且 `vllm` 会把 torch 定到 **CUDA 13** 构建（本机实测 `2.11.0+cu130`），**GPU 需要 NVIDIA 驱动 ≥ 580**。驱动更老时 `torch.cuda.is_available()` 是 `False`，GPU 示例跑不了，纯 CPU 的示例不受影响。
3. **第 3 步不用再装任何东西。** 大环境里已经有 torch，直接跑。（以前那个「为了一个 CPU 示例单独 `uv pip install torch`」的步骤已经没有必要。）

想知道某个文件是干什么、怎么跑：

```bash
python3 -c "print(open('INDEX.md').read())"   # 或者直接打开 INDEX.md
```

想跑需要 key 的例子（比如 `ch03/api_call_deepseek.py`）：

```bash
export DEEPSEEK_API_KEY=...     # 见 .env.example，里面只有变量名
python ch03/api_call_deepseek.py
```

**没有 key 时，这类示例会明确报错并退出（退出码 2），不会静默假装成功。** 这是刻意设计的：静默失败比报错难查得多。

## 三、目录结构

```
.
├── ch01/ … ch14/          按书中章节分的示例（第 11 章按案例再分一层）
│   └── ch11/cases/        五个完整案例，每个自带 README
│       ├── video_probe/       §11.1 视频识别游泳者溺水
│       ├── novel_gen/         §11.2 长篇小说生成
│       ├── finetune/          §11.3 对一个大模型进行微调
│       ├── knowledge_graph/   §11.4 构建大模型知识图谱
│       └── mcp_system/        §11.5 手搓一个完整的 MCP 体系
├── common/                多章复用的公共模块（数据库、日志、GPT 封装等）
├── config/                配置读取（真值文件不入库，只有 example）
├── ecosystem/             生态示例（DSPy、Gemini Live）
├── tools/                手动安装脚本（spaCy 中文模型，见 ENV_MATRIX §9）
├── INDEX.md               章节 ↔ 文件对照（最有用的一张表）
├── EXAMPLES_STATUS.md     82 个文件逐个的真实退出码（哪些能跑、哪些缺什么）
├── ENV_MATRIX.md          依赖分组说明 + 作者机器上的原始取证
└── pyproject.toml         依赖声明
```

## 四、章节 ↔ 代码怎么查

两步：

1. 打开 [INDEX.md](INDEX.md)，按章找文件，每个文件一行「这个文件干什么」。
2. 跑之前看一眼 [EXAMPLES_STATUS.md](EXAMPLES_STATUS.md) 里这个文件的真实退出码 —— 免得你花半小时排自己的环境，结果是它本来就缺 key。

多数 `.py` 文件头有这套六行注释，直接看文件也行：

```
① 这是什么   ② 对应章节   ③ 怎么跑
④ 依赖       ⑤ 实测环境   ⑥ 和书里的差异
```

## 五、环境就是一条命令

```bash
uv sync --all-extras
```

装完 516 个包、约 11 GB，`pyproject.toml` 里所有分组都在里面。你不需要读一张「哪个文件要哪组」的表 —— 那是把装箱单的活派给读者。

版本冲突在 `pyproject.toml` 里就用 `constraint-dependencies` 钉死了（`transformers<5`、`huggingface-hub<1.0`、`onnxruntime<1.24`），不需要你在装的时候想办法。

**已知的一处例外**：`unsloth` / `trl` / `torchtext` 没进这套环境，因为它们要求的 torch 与 `vllm` 要求的那份直接打架。用它们的三个示例（`ch04/unsloth_wt1.py`、`ch06/unsloth_wt1.py`、`ch06/troch_train_c_moe_v1.py`）要另开环境，按各自文件头装。[ENV_MATRIX.md](ENV_MATRIX.md) §10 记了当时的取证。

装完之后每个文件到底能不能跑，见 **[EXAMPLES_STATUS.md](EXAMPLES_STATUS.md)** —— 82 个文件逐个的真实退出码，不是推断。

## 六、已知限制

说实话的部分：

1. **不是所有示例都能无条件跑通。** 需要 GPU / API key / 大模型权重 / 外部服务（MySQL、Redis、Milvus）的，都在文件头写清了前置条件。没写「能跑」的，就是没验证过。
2. **微调案例的数据是模板合成的**，不是真实手册语料。它证明的是数据管道的形状，不是模型效果。
3. **知识图谱案例用的是三篇短文**，是为了让读者十分钟内看到建图结果，不是规模演示。
4. **视频案例需要样本视频**，仓库里只带合成测试片，不带真实拍摄素材。
5. **示例里的参数、阈值、领域数据都是教学示意**，不能当生产配置或操作规程用。
6. **`outputs/` 不入库**：自己跑一遍就有了。仓库里没有「已经跑好的结果」，因为结果应该由你在你的机器上产生。
7. **还有 11 个示例没补齐六行文件头**：`ch02/model_run.py`、`ch02/openai_test.py`、`ch06/rag_hyde_rrf.py`、`ch06/torch_env_check.py`、`ch06/transformers_config.py`、`ch07/function_calling_demo.py`、`ch07/stock_server.py`、`ch07/task_server.py`、`ch07/weather_server.py`、`config/cfg.py`、`ecosystem/gemini_demo/gemini_live_api.py`。它们能跑，但头里缺 ⑤ 实测解释器 —— 这个不能靠猜，等逐个真跑过再补。
8. **spaCy 中文模型要手动装，而且会被 `uv sync` 删掉。** `zh_core_web_sm` / `zh_core_web_lg` 是 GitHub Release 的 wheel（不在 PyPI）。写进 `pyproject.toml` 的话，因为 `uv lock` 会把所有分组一起解析，**任何**一次 `uv sync`（包括 `--all-extras`）都变成「先得能访问 GitHub」。所以它们被移到了 `tools/install_spacy_models.sh`。顺序是「先 `uv sync`，再装模型」；之后再 sync 又没了，重装一次即可。
9. **82 个文件里，只有 32 个能在这台机器上真跑出结果。** 这个数字不含辩解：24 个直接跑通、7 个用文件头写的参数（`--mock` / `--stub` / 子命令）跑通、1 个常驻服务探活通过。其余 50 个：25 个缺 key / 缺权重 / 缺外部服务 / 缺包 / 驱动太老，1 个是代码配的是旧版 torch 的导出 API，2 个本来就是示意代码，22 个是 `common/` 和 `config/` 下的库模块（没有 `__main__`，不是可执行脚本）。逐条见 [EXAMPLES_STATUS.md](EXAMPLES_STATUS.md)。
10. **`unsloth` / `trl` / `torchtext` 没进 extras，是故意的。** 代码里确实 import 了它们（`ch04/unsloth_wt1.py`、`ch06/unsloth_wt1.py`、`ch06/troch_train_c_moe_v1.py`），但它们与 `gpu` 组里钉住的 torch 版本互相打架 —— 加进去只会让每一个 `uv sync --all-extras` 的人都卡在装不上。要跑这几个示例，按各自文件头手动装。

## 七、许可

本仓库代码采用 **Apache License 2.0**，全文见 [LICENSE](LICENSE)。
书中的文字与示例代码采用同一许可。

## 八、安全

发现密钥或安全问题，请走私信，别开公开 issue —— 见 [SECURITY.md](SECURITY.md)。
