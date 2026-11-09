# ENV_MATRIX — 哪个解释器跑哪些示例

> 一句话：**本仓库没有一个环境能跑全部示例，这是有意的分层，不是没装好。**
> ONNX 环境只装推理链（torch/onnx/transformers），MCP 环境只装协议链（mcp/fastmcp）。
> 别试图把它们合并成一个大环境 —— torch 与 transformers 的版本要求和 mcp 2.x 的依赖树会互相拉扯，
> 合并的代价远大于收益。

本文所有数字都是 **在本机实测** 的，命令与输出见文末「§5 取证」。

---

## 1. 四个解释器实测对照（**真 import，不看 metadata**）

> 🔴 **口径**：下表每一格都是 `importlib.import_module(...)` **真跑过**的结果，
> 不是 `find_spec` 也不是 `metadata.version`。
> 起因：系统 `python3` 装了 transformers 4.52.4、`find_spec` 是 True、`metadata` 也有版本号，
> 但 `import transformers` 实测报 `ModuleNotFoundError: No module named 'regex'` —— **看得见 ≠ 导得进去**。
> 前一版本表就是被这个骗了（曾写「系统 python3 的 transformers 可用」，是错的）。

| 解释器 | Python | torch | onnx | onnxruntime | transformers | fastmcp | mcp | litellm |
|---|---|:--:|---|:--:|---|:--:|:--:|:--:|
| `.venv` | 3.10.12 | ✗ | ✗ | 1.23.2 | ✗ | **4.1.4** | **2.3.0** | ✓ |
| `.venv-core` | 3.10.12 | ✗ | ✗ | ✗ | ✗ | **4.1.4** | **2.3.0** | ✗ |
| `venv-onnx` | 3.11.15 | **2.6.0+cpu** | **1.23.2** | 1.31.1 | **4.57.6** | ✗ | ✗ | ✓ |
| 系统 `/usr/bin/python3` | 3.10.12 | 2.7.0+cu126 | ✗ | ✗ | **✗（缺 regex）** | ✗ | ✗ | ✗ |

其余常用包（同样真 import）：

| 解释器 | accelerate | safetensors | optimum | openai | numpy | chromadb | langgraph | crewai |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `.venv` | ✗ | ✗ | ✗ | 3.3.0 | 2.2.6 | **1.1.1** | ✓ | 1.6.1 |
| `.venv-core` | ✗ | ✗ | ✗ | 3.40.0 | 2.2.6 | ✗ | ✗ | ✗ |
| `venv-onnx` | 1.15.0 | 0.8.0 | ✓ | 2.54.0 | 2.4.6 | ✗ | ✗ | ✗ |
| 系统 `python3` | **✗（缺 safetensors）** | ✗ | ✗ | **✗（缺 sniffio）** | 1.26.4 | ✗ | ✗ | ✗ |

**系统 `python3` 的三个坑（全部实测）**：
- `transformers` → 缺 `regex`
- `openai` / `fastapi` → 缺 `sniffio`
- `accelerate` → 缺 `safetensors`
- `tiktoken` → 缺 `regex`
- `pandas` / `jieba` / `pypinyin` / `pytest` / `redis`（redis 能 import）/ 视包而定

## 2. 🔴 仓库根的目录名会假装成 PyPI 包（不止 `mcp/`）

仓库里有 **26 个顶层目录**，其中 7 个与常见包/标准库同名：`asyncio` `openai` `fastapi`
`langchain` `milvus` `vllm` `config`。它们都没有 `__init__.py`，Python 会把它们当成
**namespace package**（`__file__` 为 `None`）。
（原本还有第 8 个 `mcp` —— 已搬入 `ch07/`，见本节末尾。）

**触发条件（实测，不是推测）**：只有当「该名字在该解释器里没有装成 regular package」时，
本地目录才会被命中。装了真包 → 真包赢（regular package 优先于 namespace 目录）。

在仓库根逐个 `find_spec`（`.venv` 解释器）：

```
asyncio     /usr/lib/python3.10/asyncio/__init__.py                  ← 标准库赢
openai      .../.venv/lib/python3.10/site-packages/openai/__init__.py ← 真包赢
mcp         .../.venv/lib/python3.10/site-packages/mcp/__init__.py    ← 真包赢
fastapi     .../.venv/lib/python3.10/site-packages/fastapi/__init__.py← 真包赢
langchain   NAMESPACE->['<仓库根>/langchain']     ← 假象！
milvus      NAMESPACE->['<仓库根>/milvus']        ← 假象！
vllm        NAMESPACE->['<仓库根>/vllm']          ← 假象！
config      NAMESPACE->['<仓库根>/config']        ← 假象（仓库自己的模块）
```

### 这个假象坑过两次（都留了证）

| 现象 | 真相 |
|---|---|
| 仓库根 `import mcp` → `__file__=None`、`__path__=['…/mcp']`；`import mcp.server` → `ModuleNotFoundError: No module named 'mcp.server'` | 该解释器**没装 mcp**。`venv-onnx` / 系统 `python3` 都这样 |
| 仓库根 `import langchain` 成功 → 于是写入「langchain 1.4.7 实测可导入」 | **langchain 根本没装**。在 `/tmp` 跑：`ModuleNotFoundError: No module named 'langchain'`（三个解释器全一样）。那句已从 README 删掉 |

### 自检一行

```bash
python -c "import mcp; print(mcp.__file__)"      # 或把 mcp 换成 openai/langchain/milvus/vllm
```

- 打印 `.../site-packages/...` → 真包
- 打印 `None` → **被仓库里的同名目录遮蔽了**（此时该包其实没装）

### 正确姿势

```bash
cd ch07 && python weather_server.py     # 推荐：在章节目录里跑
python ch07/weather_server.py           # 等价：sys.path[0] 是脚本所在目录
cd /tmp && python -c "import mcp; ..."  # 探测某个包在不在，就去仓库外
```

**别做**：仓库根 `python -c "import X"` 探测、仓库根直接 `pytest`。
**也不要**为此重命名目录或加 `__init__.py` —— 目录名是书稿里的名字，
加了 `__init__.py` 会把它变成真包，反而把「装着 SDK 的环境」也顶掉。文档说明足够。

#### 根目录 `mcp/` 已搬入 `ch07/`，遮蔽问题消除

ch07 定稿时，仓库根 `mcp/` 的 5 个脚本整体 `git mv` 进 `ch07/`（`mcp_server.py` /
`mcp_server1.py` / `mcp_client.py` / `mcp_server_mysql.py` / `mcp_count_qw.py`），
根目录不再存在名为 `mcp` 的目录。

搬后实测（`.venv` 解释器）：

```
cd <仓库根> && python -c "import mcp; print(mcp.__file__)"
→ <仓库根>/.venv/lib/python3.10/site-packages/mcp/__init__.py   ← 真包，无遮蔽
```

- `ch07/` 下没有 `mcp/` 子目录，只有 `mcp_server.py` 这类**脚本文件名**；文件名不构成包名，不会遮蔽 `mcp`。
- **cwd 仍然重要**：从仓库根执行 `ch07/mcp_server.py` 时 `sys.path[0]` 是 `ch07/`，
  从章节目录执行时是 `ch07/` 本身；只有当某个 `sys.path` 项下真的存在 `mcp/` 目录时遮蔽才会重现
  —— 所以别在 `ch07/` 里新建名为 `mcp` 的目录，也别在仓库根做包探测。

> ⚠️ **计数口径说明（重要）**：本节的三张表是**发布前那棵树上 92 个 `.py` 的实测结果**。
> 公开发布版移除了与本书无关的归档目录（历史技术栈）和两个书中已删除的案例，
> 所以当前仓库是 **82 个 `.py`**。**表格按文件列的逐行结论仍然有效**（那些文件都还在），
> 但「合计 / 各解释器覆盖数」这类汇总数字未按新树重算 —— 以文件行为准，别看合计。

### 3.1 关键章节（逐块对过的示例，真 import 判）

| 文件 | 书稿位置 | 用哪个解释器（真 import） | 外部依赖 |
|---|---|---|---|
| `ch03/api_call_deepseek.py` | §3.x | `venv-onnx` / `.venv` / `.venv-core` | openai（系统 python3 的 openai 缺 sniffio，**不行**） |
| `ch03/transformers_infer.py` | §3.x | **仅 `venv-onnx`** | torch, transformers |
| `ch04/transformer_min.py` | §4.x | **`venv-onnx`** | torch（系统 python3 也行） |
| `ch04/model_architectures.py` | §4.x | **仅 `venv-onnx`** | torch, transformers（系统 python3 装了 4.52.4 但缺 regex，真 import 失败） |
| `ch04/onnx_export_min.py` | §4.x | **仅 `venv-onnx`** | numpy, onnxruntime, torch |
| `ch04/safetensors_read.py` | §4.x | **仅 `venv-onnx`** | safetensors, torch |
| `ch05/openai_protocol_http.py` | §5.x | 任意（四个都行） | requests |
| `ch06/torch_env_check.py` | §6.x | **`venv-onnx`**（系统 python3 也行） | torch |
| `ch06/transformers_config.py` | §6.x | **仅 `venv-onnx`** | transformers |
| `ch06/rag_hyde_rrf.py` | §6.x | 任意（零第三方依赖） | — |
| `ch07/weather_server.py` | §7.2.2 | **`.venv` / `.venv-core`** | mcp |
| `ch07/stock_server.py` | §7.3.2 | **`.venv` / `.venv-core`** | fastmcp |
| `ch07/task_server.py` | §7.3.3 | **`.venv` / `.venv-core`** | fastmcp |
| `ch07/function_calling_demo.py` | §7.1.4 | **`venv-onnx` / `.venv`** | litellm |
| `ch08/openai_cloud_stream.py` | §8.3.1 | **`venv-onnx` / `.venv` / `.venv-core`** | openai |
| `ch08/vllm_openai_compat.py` | §8.3.1 | **`venv-onnx` / `.venv` / `.venv-core`** | openai |
| `ch08/rag_chroma_demo.py` | §8.3.3 | **仅 `.venv`** | chromadb 1.1.1 + openai |
| `ch09/react_agent_min.py` | §9.4.1 | **`venv-onnx` / `.venv` / `.venv-core`** | openai；`--mock` 模式零依赖 |
| `ch07/mcp_server.py` `ch07/mcp_server1.py` `ch07/mcp_client.py` | §7.3.5 等 | **`.venv` / `.venv-core`** | fastmcp |
| `ch10/concurrency_vs_serial.py` | §10.1.5 | 任意（零第三方依赖） | — |
| `ch10/structured_output.py` | §10.3.4 / §10.4.2.2 | 任意（零第三方依赖） | — |
| `ch10/tool_param_adapter.py` | §10.4.3.2 / §10.4.3.3 | 任意（零第三方依赖） | — |
| `ch13/arithmetic_trap_demo.py` | §13.3.1 | `.venv` / `.venv-core` / `venv-onnx`（离线零依赖；`--live` 需 openai） | openai |
| `ch13/tool_dispatch_demo.py` | §13.3.4 | 同上 | openai |
| `ch07/mcp_server_mysql.py` | §7.3.5 | **`.venv` / `.venv-core`** | fastmcp, mcp, pymysql |
| `ch07/mcp_count_qw.py` | 仓库自有 | **`.venv` / `.venv-core`** | fastmcp, openai |

> ⚠️ `ch08/rag_chroma_demo.py` 只能 `.venv` —— 我第一次复核时用 `venv-onnx` 跑它报
> `No module named 'chromadb'`，那不是代码问题，是拿错解释器。文件头写的就是 `.venv/bin/python`。

### 3.2 全量分布（脚本算，实测时 92 个 `.py` 全覆盖）

方法与 §5.3 一致，但**判据从 `find_spec` 换成真 `import`**（仓库内 `common`/`config`/`cases` 视为本地模块）。
外部第三方模块共 36 个。逐个解释器能覆盖的文件数：

| 解释器 | 可覆盖 |
|---|---:|
| `.venv` | **62 / 92** |
| `.venv-core` | 57 / 92 |
| `venv-onnx` | 52 / 92 |
| 系统 `python3` | **40 / 92**（旧口径用 `find_spec` 虚报为 56/83） |

| 可满足的解释器组合 | 文件数 |
|---|---:|
| 四个都行 | 27 |
| `.venv` + `.venv-core`（MCP 系） | 13 |
| 仅 `venv-onnx`（torch/onnx 系） | 12 |
| `.venv` + `.venv-core` + 系统 `python3` | 8 |
| `venv-onnx` + `.venv` + `.venv-core` | 9 |
| 仅 `.venv` | 3 |
| 仅系统 `python3` | 2 |
| `venv-onnx` + 系统 `python3` | 2 |
| 其他组合 | 2 |
| **四个都不行** | **14** |
| 合计 | **92** |

**四个都不行的 14 个（比旧口径的 8 个多 6 个 —— 旧表用 `find_spec` 漏判）**：

| 文件 | 缺什么 |
|---|---|
| `ch11/cases/finetune/train_lora.py`、`infer_compare.py` | `peft` + `torch` + `transformers` |
| `ch11/cases/video_probe/ocr_frames.py` | `opencv-python` + `pytesseract` + 系统 tesseract |
| `common/gpt/volcengine.py` | `volcenginesdkarkruntime` |

> 但**跑通用的环境不是这四个**（当时是带 `peft`/`sklearn` 的专用环境，已不存在）。
> 现在按「当前四个解释器」判，它们是「四个都不行」。这两件事不矛盾，但读者要知道差别 ——
> 详见 §6。

### 3.3 各章汇总

| 章/目录 | 文件数 | 用什么 |
|---|---:|---|
| `ch03` | 2 | `venv-onnx`；`api_call_deepseek.py` 还可用 `.venv`/`.venv-core`（缺 key 才跑不通） |
| `ch04` | 4 | **`venv-onnx`**（3 个仅此一个；`transformer_min.py` 系统 python3 也行） |
| `ch05` | 1 | 任意（仅 requests） |
| `ch06` | 3 | `venv-onnx`；`rag_hyde_rrf.py` 零依赖任意 |
| `ch07` | 9 | `.venv`/`.venv-core`（`function_calling_demo.py` 用 `venv-onnx`/`.venv`） |
| `ch08` | 3 | `.venv`（chromadb / openai）；`vllm_openai_compat.py` 三个都行 |
| `ch09` | 1 | 任意（openai 需 `.venv` 系或 venv-onnx；系统 python3 的 openai 缺 sniffio） |
| `model/` | 12 | `venv-onnx`（6 个）；另 6 个四环境都不行（torchtext 等） |
| `common/` | 21 | 多数任意；`volcengine.py` 不行 |
| `ch11/cases/` | 24 | 见 §3.2 与 §6 |
| `ecosystem/` | 3 | `.venv`（crewai/langgraph） |
| `openai/` | 1 | `venv-onnx`/`.venv`/`.venv-core` |
| `vllm/` `evalscope_1/` | 2 | **不行**（vllm / evalscope 未装） |

> 原根目录 `mcp/`（5 个文件）已 `git mv` 进 `ch07/`，故 `ch07` 从 4 增至 9，`mcp/` 这一行取消。
> 原根目录 `cases/`（7 个案例、123 个文件、24 个 `.py`）已整体 `git mv` 进 `ch11/cases/`；本表路径与计数同步为 24。

## 4. 每个示例文件头都有一行 `实测解释器`

取回/新增的示例在文件头「依赖」段里都写了：

```
实测解释器：python（需 onnxruntime + torch 的环境）   （或 .venv/bin/python）
```

看到这一行，就别换解释器硬试 —— 换环境跑不通不代表代码有问题。

---

## 5. 取证（可复现）

### 5.1 四解释器包版本（真 import 取 `__version__`）

```bash
cd /tmp     # ← 必须离开仓库根，否则 mcp 会被 §2 的目录遮蔽
for P in python（需 onnxruntime + torch 的环境） \
         python \
         <仓库根>/.venv-core/bin/python /usr/bin/python3; do
  $P -c "
import importlib, json
o={}
for m in ['torch','onnx','onnxruntime','transformers','fastmcp','mcp','litellm','accelerate',
          'safetensors','optimum','openai','numpy','chromadb','langgraph','crewai']:
    try:
        mod=importlib.import_module(m); o[m]=getattr(mod,'__version__','OK')
    except BaseException: o[m]='-'
print(json.dumps(o,ensure_ascii=False))"
done
```

实测输出（`-` = 真 import 失败）：

```
venv-onnx    torch 2.6.0+cpu  onnx 1.23.2  onnxruntime 1.31.1  transformers 4.57.6
             fastmcp -  mcp -  litellm OK  accelerate 1.15.0  safetensors 0.8.0
             optimum OK  openai 2.54.0  numpy 2.4.6  chromadb -  langgraph -  crewai -
.venv        torch -  onnx -  onnxruntime 1.23.2  transformers -
             fastmcp 4.1.4  mcp OK  litellm OK  accelerate -  safetensors -  optimum -
             openai 3.3.0  numpy 2.2.6  chromadb 1.1.1  langgraph OK  crewai 1.6.1
.venv-core   torch -  onnx -  onnxruntime -  transformers -
             fastmcp 4.1.4  mcp OK  litellm -  accelerate -  safetensors -  optimum -
             openai 3.40.0  numpy 2.2.6  chromadb -  langgraph -  crewai -
python3      torch 2.7.0+cu126  onnx -  onnxruntime -  transformers -
             fastmcp -  mcp -  litellm -  accelerate -  safetensors -  optimum -
             openai -  numpy 1.26.4  chromadb -  langgraph -  crewai -
```

**`find_spec` / `metadata.version` 会骗人 —— 同一个解释器三种说法**：

| 模块 | `metadata.version` | `find_spec` | 真 `import` |
|---|---|---|---|
| 系统 python3 · transformers | 4.52.4 | True | **✗ `No module named 'regex'`** |
| 系统 python3 · openai | 1.71.0 | True | **✗ `No module named 'sniffio'`** |
| 系统 python3 · accelerate | 1.6.0 | True | **✗ `No module named 'safetensors'`** |

### 5.2 遮蔽取证

```bash
# 仓库根，未装 mcp 的解释器
cd <仓库根>
python（需 onnxruntime + torch 的环境） -c "import mcp;print(mcp.__file__)"
# → None
/usr/bin/python3 -c "import mcp.server"
# → ModuleNotFoundError: No module named 'mcp.server'

# 仓库外，同一个解释器
cd /tmp && /usr/bin/python3 -c "import importlib.util as u;print(u.find_spec('mcp'))"
# → None   （真相：没装）

# 已装 mcp 的解释器在仓库根 → 不受影响
cd <仓库根>
.venv/bin/python -c "import mcp;print(mcp.__file__)"
# → <仓库根>/.venv/lib/python3.10/site-packages/mcp/__init__.py
```

### 5.3 逐文件依赖可满足性（实测时 92 个文件，真 import）

```bash
git ls-files -z                     # 取全量（-z 处理中文路径）
  → ast 抽顶层 import（含函数内 import）
  → 排除 stdlib 与仓库内本地模块（common / config / cases / dataset + 各文件名）
  → 四个解释器各跑一次 importlib.import_module（cwd 固定 /tmp）
  → 全通过才算「该解释器可满足」
```

实测结果：外部第三方模块 36 个；`venv-onnx` 覆盖 52/92、`.venv` 62/92、
`.venv-core` 57/92、系统 `python3` **40/92**（旧口径用 `find_spec` 虚报为 56/83）。

> 这个分母每加一个 `.py` 就会变。**别手写死** —— 加完文件重跑一遍 §5.3 的脚本。

> ⚠️ 注释里的「可见性」再强调一次：这几个数字是**依赖可满足性**，
> 不等于「跑得通」。`ch04/model_architectures.py` 在 `venv-onnx` 里依赖全满足，
> 但首次运行仍会去 Hugging Face 下模型 —— 见 §6。

## 6. 依赖够了 ≠ 跑得通：三个已知的「下一步还会卡」

| 文件 | 依赖满足后还缺什么 | 实测 |
|---|---|---|
| `ch04/model_architectures.py` | 模型权重。直连 `huggingface.co` 超时：`OSError: We couldn't connect to 'https://huggingface.co' ...` | 加 `HF_ENDPOINT=https://hf-mirror.com` 后 ✅ 跑通：`预测类别: 0` |
| `ch03/transformers_infer.py` | 同上（Qwen2.5-0.5B），本机已有本地权重可绕开 | ✅ 已用本地权重跑通 |
| `ch08/rag_chroma_demo.py` | 第 4 步要 `OPENAI_API_KEY` | 检索链路 ✅，生成步骤跳过 |

`INDEX.md` 里它们是 A 级（真跑通过），但当时用的是带 `peft` / `scikit-learn` 的专用环境，
**那个环境现在不在**。按当前四个解释器判，它们「四个都不行」。
两个说法都对，区别是「曾经能跑」与「现在这台机器上能跑」。读者要跑得自己补 `peft`/`sklearn`。


---

## 7. 拿一把真实 key 逐个实跑：16 个 API 示例通了 15 个

把 `DEEPSEEK_API_KEY` 放进 `<仓库根>/.env`（`.env` 已在 `.gitignore` 里，不进版本库），然后**逐个真跑** ——
不看 `--dry-run`、不看「预计可跑」，只认退出码。

### 7.1 结果表（rc 为当次真实退出码）

| 脚本 | rc | 靠什么打通 |
|---|---|---|
| `ch02/openai_test.py` | **0** | 加 `OPENAI_BASE_URL` / `OPENAI_MODEL` 覆盖（默认值不变） |
| `ch03/api_call_deepseek.py` | **0** | 原生 DeepSeek 端点，无需改 |
| `ch05/openai_protocol_http.py` | **0** | 原生；`MODEL_ID=deepseek-flash` |
| `ch06/openai_test.py` | **0** | 加 `OPENAI_MODEL` 覆盖（`base_url` 本来就能覆盖） |
| `ch06/gemini_live_api.py` | **1** | ❌ **只有这个不行**：`google-genai` 的 Live 流式协议，与 OpenAI 协议不通，换 key 无效；须自备 `GEMINI_API_KEY` |
| `ch07/function_calling_demo.py` | **0** | `FC_MODEL` 必须带 provider 前缀：`deepseek/deepseek-chat`（走 litellm，不带前缀报 `LLM Provider NOT provided`） |
| `ch07/mcp_count_qw.py` | **0** | ① 先把 `ch07/mcp_server.py --sse` 起在 127.0.0.1:8011（默认 stdio，**必须加 `--sse`**）② 客户端是交互式 `input()`，要管道喂问句 + `退出` |
| `ch08/openai_cloud_stream.py` | **0** | `CLOUD_BASE_URL` + `CLOUD_MODEL` 已支持 |
| `ch08/rag_chroma_demo.py` | **0** | ① embedding 步遇 404 自动降级为本地哈希向量（本轮新增降级逻辑）② 聊天模型改 `OPENAI_MODEL` 可覆盖 |
| `ch09/crewai_demo/crewai_quickstart.py` | **0** | `OPENAI_MODEL_NAME=deepseek-flash`（CrewAI 默认 `gpt-4.1-mini`，会被端点拒） |
| `ch09/react_agent_min.py` | **0** | 原生，已内置 `DEEPSEEK_API_KEY` 优先 |
| `ch11/cases/novel_gen/outline_anchor.py` | **0** | **必须 `PYTHONPATH=<仓库根>`**，否则 `ModuleNotFoundError: No module named 'common'` |
| `ch11/cases/novel_gen/section_writer.py` | **0** | 同上 |
| `ch11/cases/novel_gen/hard_reference_edit.py` | **0** | 同上 |
| `ch13/arithmetic_trap_demo.py --live` | **2** | ✅ **rc=2 是设计信号**不是失败：`return 0 if bad == 0 else 2`，本轮 3 次全算错 → 正好印证「算术陷阱」这一节的论点 |
| `ch13/tool_dispatch_demo.py --live` | **0** | 原生；并演示了「书稿版序列 3 处违规、修正版合法」 |

**净结果：15 个真跑通，1 个（Gemini Live）必须自备 Gemini key。**

### 7.2 一条命令跑全覆盖（recipe）

```bash
cd <仓库根>
set -a; . ./.env; set +a
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL="deepseek-flash"
export OPENAI_MODEL_NAME="deepseek-flash"
export CLOUD_BASE_URL="https://api.deepseek.com/v1"
export CLOUD_MODEL="deepseek-flash"
export PYTHONPATH="$PWD"
```

> ⚠️ 代码**不会自动加载 `.env`**（`.env.example` 顶部已说明），必须 `set -a; . ./.env; set +a` 手动导入。

### 7.3 DeepSeek 侧实测到的模型名单（`GET /models`）

| id | 名称 | 上下文 | 输入模态 |
|---|---|---|---|
| `deepseek-flash` | DeepSeek-V4.1-Flash | 1,048,576 | text, **image** |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | 1,048,576 | text |

- `deepseek-chat` / `deepseek-reasoner` / `deepseek-v4-flash` 调用也能返回 200（旧别名仍兼容），但 `/models` 未列出。
- **`deepseek-flash` 支持图像输入** —— 书稿第 11 章多模态那一节（§11.1）若只是「理解图片」，不一定需要本地 mmdet/mmpose；但「目标检测/姿态估计」仍要专门模型，别混淆。

### 7.4 跑完必须还原产物

`novel_gen` 三个脚本会重写 `outputs/`（12 个文件被改动、2 个新增），`rag_chroma_demo` 会写 `.chroma_demo/`（已 gitignore）。
按「产物不刷新」的既定口径，脚本文档里要写明：**跑完 `git checkout -- <outputs 目录>`**。

---

## 8. GPU 环境 `venv-gpu`（**RTX 3080 真跑**）

前四节讲的都是 CPU 解释器。本节是第五个：**能上 GPU 的那个**。

### 8.1 硬件与驱动（当次实测）

```text
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
NVIDIA GeForce RTX 3080, 10240 MiB, 550.90.07

$ python（需 torch 的环境） -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
2.6.0+cu124 12.4 True NVIDIA GeForce RTX 3080 (8, 6)
```

> ⚠️ `venv-onnx` 里的 torch 是 **2.6.0+cpu**，`torch.cuda.is_available()` 为 `False`。
> 别拿它跑任何 GPU 活 —— 它只负责 ONNX/CPU 推理。

### 8.2 `venv-gpu` 装了什么（真 import 取 `__version__`）

| 包 | 版本 | 用途 |
|---|---|---|
| torch | **2.6.0+cu124** | CUDA 12.4 轮子，`is_available()=True` |
| transformers | **4.57.6** | 见 §8.5 版本冲突说明 |
| accelerate | 1.15.0 | `device_map="auto"` |
| peft | 0.21.2 | LoRA（ch11 §11.3） |
| bitsandbytes | 0.50.2 | 4bit/8bit 量化（书 2.4 节） |
| datasets | 5.1.0 | 训练数据 |
| sentence-transformers | 6.2.0 | bge 向量 |
| opencv-python-headless | 5.0.0 | ch11 video_probe 读帧 |
| pytesseract | 0.3.13 | OCR 的 Python 壳 |
| numpy | 2.4.6 | — |
| huggingface-hub | 0.36.2 | **不能升到 1.x/2.x**，见 §8.5 |
| openai | 3.42.1 | — |

```text
# 建法（可复现；本机没有 pip，一律走 uv）
export UV_CACHE_DIR=wheels/uv-cache
uv venv --python 3.11 venv-gpu
uv pip install --python python（需 torch 的环境） \
    wheels/torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl
uv pip install --python python（需 torch 的环境） \
    numpy transformers accelerate peft bitsandbytes datasets sentence-transformers \
    scikit-learn opencv-python-headless pytesseract python-dotenv openai jieba pypinyin
```

> 🔴 **装到 `<数据盘>`，不要装到 `/` 下的 `.venv`。**
> 实测：`/` 只剩 52G，`<数据盘>` 有 124G。`venv-gpu` 本体 7.4G，装了 torch CUDA 依赖后体积翻倍。

### 8.3 环境变量（写进 shell 片段，别写死进代码）

```bash
# ~/.bashrc 或每次跑之前 set -a; . ./gpu_env.sh; set +a
export HF_HOME=models/hf
export MODELSCOPE_CACHE=models/modelscope
export UV_CACHE_DIR=wheels/uv-cache
export HF_ENDPOINT=https://hf-mirror.com   # 直连 huggingface.co 本机不通，见 §8.6
export PYTHONPATH=<仓库根>   # 否则 import common/config 失败
```

### 8.4 系统依赖与模型（当次实测清单）

```text
$ tesseract --version | head -1
tesseract 4.1.1
$ tesseract --list-langs
List of available languages (3): chi_sim  eng  osd
# 装法：sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim  （≈15MB）
```

| 模型 | 来源 | 体积 | 本地路径 |
|---|---|---|---|
| `Qwen3-8B` | **hf-mirror**（`hf-mirror.com/Qwen/Qwen3-8B`，逐文件 curl） | **16G** | `models/Qwen3-8B` |
| `BAAI/bge-small-zh-v1.5` | hf-mirror | 92M | `models/bge-small-zh-v1.5` |
| `Qwen2.5-0.5B-Instruct` | 已有（10-29） | 954M | `models/Qwen2.5-0.5B-Instruct` |

- **为什么下 Qwen3-8B 而不是回退 Qwen2.5-7B**：hf-mirror 上 `Qwen/Qwen3-8B` 存在且完整（5 个 safetensors，合计 16.4G），
  书稿 ch10 用的就是它，没必要退版本。
- **LoRA 训练仍用 0.5B**：10G 卡上 8B 全参/LoRA 训练会 OOM；书上写的也是「小模型验证链路」。
  8B 只用于「加载得起来、能生成」的验证。
- 下载总量 **16.1G**，预算 60G 内。

### 8.5 🔴 这个环境里三个**真实的版本冲突**（不是配置问题）

1. **`transformers` 必须 `4.x`**。`transformers 5.22.1` 删了 `AutoModelForVision2Seq`，
   `ImportError: cannot import name 'AutoModelForVision2Seq'`。锁 `4.57.6` 后恢复正常
   （改成占位路径后它们仍跑不了，但那是**缺权重**，不是版本）。
2. **`huggingface-hub` 必须 `<1.0`**。升到 2.4.0 → `transformers` 直接
   `ImportError: huggingface-hub>=0.34.0,<1.0 is required ... but found 2.4.0`。
3. **`diffusers` 与上面两条互斥**。`diffusers 0.42.1` 需要 `huggingface-hub>=1.0`
   （要 `resolve_revision`），而 `transformers 4.57.6` 要 `<1.0`。
   **二选一，本环境选 transformers** —— 代价：`diffusers` 相关示例仍是 B（`ImportError: cannot import name 'resolve_revision'`）。
   要跑 diffusers 就再建一个独立 venv，别在这里改 hub。

另有一条**不是版本冲突但长得像**：`fastmcp` 与本环境的 `mcp` 版本不匹配
（`ImportError: cannot import name 'AuthorizationCodeResult' from 'mcp.shared.auth'`）。
**MCP 示例继续用 `.venv` / `.venv-core`**（那里 fastmcp 4.1.4 + mcp 2.3.0 是配套的），别在 `venv-gpu` 里跑 MCP。

### 8.6 网络实测（与「都能通」的传闻不符）

| 目标 | 结果 |
|---|---|
| `pypi.org` | 200，但**单请求 6.7s**，`uv pip install` 会卡在解析阶段十几分钟 |
| `mirrors.aliyun.com/pypi` | **200，0.1s** ← 实际用的就是这个 |
| `download.pytorch.org` | 200，torch cu124 轮子 768MB / 1.8MB/s |
| `huggingface.co` | ❌ **超时**（curl 无响应） |
| `hf-mirror.com` | ✅ 200，8–10MB/s |
| `modelscope.cn` | ✅ 200 |

```bash
# 装包用阿里云镜像（pypi 官方源在本机会卡住）
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
```

### 8.7 这个环境里跑通了什么

| 脚本 | 之前 | 现在 |
|---|---|---|
| `ch11/cases/finetune/train_lora.py` | 无解释器有 torch+peft | **rc=0**，GPU 训练，峰值 3680 MiB，探针 loss 3.0027→1.0011 |
| `ch11/cases/finetune/infer_compare.py` | 同上 | **rc=0**，base vs LoRA 并排原文 |
| `ch11/cases/video_probe/ocr_frames.py --engine pytesseract` | 无 OCR 引擎 | **rc=0**，真识别出 `SCENE 1 CH-01 FRAMES 000-049` |
| `ch11/cases/video_probe/ocr_frames.py --engine tesseract-cli` | 代码里没有这个分支（死分支） | **rc=0**，5/5 帧相似度 1.0。已补的分支 + 探测对齐，详见 ch11 的 video_probe README |
| `ch06/vllm_t.py` | 四个解释器都没 vllm | **rc=0**（`venv-vllm`），346 tok/s，真生成一段 ND 模式说明 |

### 8.8 `venv-vllm`：vllm 只能单独一个环境，而且要退版本

```text
$ python（需 vllm 的环境） -c "import vllm,torch;print(vllm.__version__,torch.__version__,torch.cuda.is_available())"
0.8.5 2.6.0+cu124 True
```

三次失败才成功，过程照实记：

| 尝试 | 结果 |
|---|---|
| `pip install vllm`（最新 0.32.0） | ✅ 装上，但拉的是 `torch 2.13.0+cu130` → 起引擎报 **`The NVIDIA driver on your system is too old (found version 12040)`**（本机驱动 550.90.07 = CUDA 12.4） |
| `vllm==0.10.2 --torch-backend=cu124` | ❌ 该版本硬依赖 `torch==2.8.0`，而 cu124 轮子最高只到 2.6.0 → 无解 |
| **`vllm==0.8.5 --torch-backend=cu124`** | ✅ 装上（torch 2.6.0+cu124），但起服务报 `Qwen2Tokenizer has no attribute all_special_tokens_extended` |
| **再加 `transformers==4.51.3`** | ✅ **真跑通**：`ch06/vllm_t.py` rc=0，346 tok/s |

**结论**：要跑 vLLM，必须 `vllm 0.8.5 + torch 2.6.0+cu124 + transformers 4.51.3` 这三件套一起锁；
`venv-gpu` 里的 transformers 4.57.6 不能动（vllm 要 4.51.x，二者不能共存）。
这也是为什么 **vllm 必须独立 venv**，不是洁癖。

---

## 9. spaCy 中文模型：手动装，不在 `pyproject.toml` 里

`zh_core_web_sm`（约 12MB）和 `zh_core_web_lg`（约 575MB）是 **GitHub Release 的 wheel**，不在 PyPI。

它们原本声明在 `pyproject.toml` 的 `nlp` / `nlp-model` 两个 extra 里。问题出在：
**`uv lock` 会把所有 `optional-dependencies` 分组一起解析**，不管你实际要装哪一组。
于是读者只想 `uv sync --extra core`（numpy / pandas / openai，全在 PyPI），
也必须先能访问 GitHub 才拿得到这两个 wheel 的元数据 —— 国内网络这一步常常直接断，
`uv sync` 只会给你一句 70 秒左右超时，看不出问题出在一个你根本没要装的包上。

所以它们被移出 `pyproject.toml`，改成手动装：

```bash
bash tools/install_spacy_models.sh sm      # zh_core_web_sm，约 12MB
bash tools/install_spacy_models.sh lg      # zh_core_web_lg，约 575MB
bash tools/install_spacy_models.sh both    # 两个都装
```

**顺序很重要**（不粉饰）：`uv sync` 会把不在 `pyproject.toml` 里的包当成「多余」删掉。
所以是「**先** `uv sync --extra nlp`，**再**跑上面的脚本」。之后任何一次 `uv sync` 都会把模型删掉，
重跑一次脚本即可。

校验装没装上：

```bash
.venv/bin/python -c "import spacy; print(spacy.load('zh_core_web_sm').meta['version'])"
```

> **现状**：扫全仓 82 个 `.py`，**没有任何示例 `import spacy`**。
> 所以这一步现在谁都不必跑 —— 保留它，是因为 `nlp` 分组本身还在，这两个模型是旧依赖声明的遗留。
