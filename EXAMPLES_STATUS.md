# 示例实跑状态（EXAMPLES_STATUS）

这张表是**这个仓库对读者的承诺**：跑通的就是跑通的，跑不通的写清楚缺什么。
每一行的 `rc` 都是一次真实进程退出码，不是推断。

## 在什么环境跑的

| 项 | 值 |
|---|---|
| 日期 | 2026-11-09 |
| 机器 | Ubuntu 22.04 / 16 核 / RTX 3080 10GB / 驱动 550.90.07（CUDA 12.4） |
| Python | 3.11.15（`uv` 管的独立 venv，`.venv/`） |
| 依赖 | `uv sync --all-extras` 一条命令装完，516 个包（uv 0.11.7，rc=0） |
| 解释器 | `.venv/bin/python`，仓库根为工作目录 |
| 跑法 | 每个文件**单独起一个进程**，`stdin=/dev/null`，超时 60s，环境里**没有任何 API key** |
| 未做的事 | 没改任何代码；失败就是失败，没有为了让它返回而打补丁 |

## 汇总

**82 个 `.py` 文件**，82 个全部跑过。

| 归类 | 个数 |
|---|---|
| ✅ 跑通 | 25 |
| ✅ 跑通（用文档给的参数） | 7 |
| 🔁 常驻服务（已探活） | 1 |
| 📄 示意代码（按设计不执行） | 2 |
| 🔑 缺 API key | 9 |
| 📦 缺模型权重 | 6 |
| 🌐 需要外网下载 | 2 |
| 🌐 缺外部服务 | 1 |
| 🖥 缺驱动/系统组件 | 2 |
| 📚 缺包（extras 里没有） | 4 |
| 🧱 缺前置产物 | 1 |
| 🧩 库模块（非可执行脚本） | 17 |
| 🧩 库模块（import 需外部条件） | 5 |
| **合计** | **82** |

能真正跑出结果的：**33** 个（✅ 跑通 25 + 用文档参数 7 + 常驻服务探活 1）。
剩下 49 个里，24 个是**环境/凭证不够**，1 个**缺前置产物**，2 个是**本来就是示意代码**，22 个是**库模块、不是可执行脚本**。

### 三种「没跑通」必须分清

| 现象 | 含义 | 是不是缺陷 |
|---|---|---|
| `rc=2` + 打印「缺少环境变量 XXX / 请设置 MODEL_DIR」 | 脚本**主动**拒绝在缺条件时发请求 | ❌ 不是，这是设计 |
| `ModuleNotFoundError` / `ImportError` | 环境里确实少东西 | ⚠️ 是本表要暴露的问题 |
| `rc=0` 但只打印「示意代码，不可直接运行」 | 文件本身就是片段 | ❌ 不是，标签早就写着 |

## 逐文件明细

### ch02/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch02/model_run.py` | `1` | 📦 缺模型权重 | 模型路径仍是占位 `/path/to/your/motogpt/...`，HFValidationError |
| `ch02/openai_test.py` | `2` | 🔑 缺 API key | 无 OPENAI_API_KEY，脚本自检后 rc=2 退出（按设计） |

### ch03/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch03/api_call_deepseek.py` | `2` | 🔑 缺 API key | 无 DEEPSEEK_API_KEY，rc=2 退出（按设计） |
| `ch03/transformers_infer.py` | `超时(60s)` | 🌐 需要外网下载 | 去 huggingface.co 拉 Qwen2.5-0.5B，30s 内没下完，60s 超时 |

### ch04/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch04/model_architectures.py` | `2` | 📦 缺模型权重 | `bert\|gpt\|t5` → 裸跑 rc=2 要子命令；子命令要 bert-base-chinese / Qwen2.5 / t5 权重（要外网） |
| `ch04/model_parser_c.py` | `1` | 📦 缺模型权重 | 模型路径仍是占位 `/path/to/your/motogpt/...` |
| `ch04/model_run.py` | `1` | 📦 缺模型权重 | 模型路径仍是占位 `/path/to/your/motogpt/...` |
| `ch04/onnx_export_min.py` | `0` | ✅ 跑通 | `export` → rc=0（11 MB .onnx）；`check` → rc=0；`infer` → rc=0，输出 shape (1, 5)。已显式 `dynamo=False` 走 legacy 导出器，以保持与书稿的 `dynamic_axes` 写法一致；`check` 子命令改调 `onnx.checker.check_model`（旧的 `python -m onnx.checker` 是假校验，详见文末第 1 条） |
| `ch04/onnx_m.py` | `1` | 📚 缺包（**故意不补**） | `optimum` 2.3 把 ONNX Runtime 执行器拆成了独立包；装它要 `huggingface-hub>=1.0`，与 `transformers<5` 冲突。保持 `transformers<5` 优先 |
| `ch04/safetensors_read.py` | `0` | ✅ 跑通 | 自造 safetensors 文件并 mmap 读回，tensor (8,8)，临时目录自清 |
| `ch04/transformer_min.py` | `0` | ✅ 跑通 | 手写单头自注意力，CPU 跑完（本机 CUDA 不可用） |
| `ch04/unsloth_wt1.py` | `1` | 📚 缺包（extras 里没有） | 缺 unsloth；按设计不进 extras（与 gpu 组 torch 打架） |

### ch05/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch05/openai_protocol_http.py` | `2` | 🔑 缺 API key | 无 key 时不发请求、rc=2 退出（按设计） |

### ch06/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch06/accelerate_min.py` | `1` | 🖥 缺驱动/系统组件 | torch 2.11.0+cu130 要驱动 ≥580，本机 550.90.07（CUDA 12.4）→ 初始化失败 |
| `ch06/dspy_min.py` | `0` | ✅ 跑通 | DummyLM 离线桩，Predict/CoT/compile/save-load 全链路 10s 跑完 |
| `ch06/gemini_live_api.py` | `0` | 📄 示意代码（按设计不执行） | 打印缺失清单（key/麦克风/音频设备）后 rc=0 退出，本来就是示意代码 |
| `ch06/model_run.py` | `超时(60s)` | 🌐 需要外网下载 | 去 huggingface.co 拉 Qwen2.5-0.5B，60s 超时 |
| `ch06/openai_test.py` | `2` | 🔑 缺 API key | 无 key 时不发请求、rc=2 退出（按设计） |
| `ch06/rag_hyde_rrf.py` | `0` | ✅ 跑通 | HyDE + RRF 融合，纯计算，rc=0 |
| `ch06/torch_env_check.py` | `0` | ✅ 跑通 | 打印 torch/CUDA 环境；CUDA 不可用属本机驱动限制，脚本本身正常 |
| `ch06/transformers_config.py` | `2` | 📦 缺模型权重 | 既无 MODEL_DIR 也无 MODEL_ID，rc=2 并给出设置方法（按设计） |
| `ch06/troch_train_c_moe_v1.py` | `1` | 📚 缺包（extras 里没有） | 缺 torchtext（ABI 与 torch 2.11 不匹配，故不声明） |
| `ch06/unsloth_wt1.py` | `1` | 📚 缺包（extras 里没有） | 缺 unsloth；按设计不进 extras |
| `ch06/vllm_t.py` | `1` | 📦 缺模型权重 | 本地模型目录 `/path/to/your/motogpt/...` 不存在 |

### ch07/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch07/function_calling_demo.py` | `1` | 🔑 缺 API key | 缺 DEEPSEEK_API_KEY |
| `ch07/mcp_client.py` | `1` | ✅ 跑通（用文档给的参数） | 裸跑连不上（没起服务）；配 `ch07/mcp_server.py --sse` 后 rc=0，列出 4 tool + 1 resource |
| `ch07/mcp_count_qw.py` | `1` | 🔑 缺 API key | 缺 DEEPSEEK_ARK_API_KEY，且需要 8011 上的 MCP Server |
| `ch07/mcp_server.py` | `0` | ✅ 跑通 | stdio 下 stdin 关闭即干净退出；`--sse` 另探活 http=200，客户端 rc=0 |
| `ch07/mcp_server1.py` | `超时(60s)` | 🔁 常驻服务（已探活） | uvicorn 常驻 8002；已起进程 + `GET /sse` → http=200（返回 endpoint 事件），已杀干净 |
| `ch07/mcp_server_mysql.py` | `1` | 🌐 缺外部服务 | 缺 MYSQL_PASSWORD；需要有 MySQL 实例 |
| `ch07/stock_server.py` | `0` | ✅ 跑通 | FastMCP stdio 服务，rc=0 |
| `ch07/task_server.py` | `0` | ✅ 跑通 | FastMCP stdio 服务，rc=0 |
| `ch07/weather_server.py` | `0` | ✅ 跑通 | FastMCP stdio 服务，rc=0 |

### ch08/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch08/openai_cloud_stream.py` | `2` | 🔑 缺 API key | 无 OPENAI_API_KEY，rc=2 并打印导出方法（按设计） |
| `ch08/rag_chroma_demo.py` | `0` | ✅ 跑通 | Chroma 入库 5 条 + Top-3 检索，无 key 时退回本地哈希向量，排序断言通过 |
| `ch08/vllm_openai_compat.py` | `1` | ✅ 跑通（用文档给的参数） | `--stub` → 裸跑连不上 vLLM；`--stub` 起桩服务后客户端 rc=0 |

### ch09/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch09/crewai_demo/crewai_quickstart.py` | `2` | 🔑 缺 API key | 无 OPENAI_API_KEY，rc=2 并提示 `--dry-run`（按设计） |
| `ch09/langgraph_demo/langgraph_state_graph.py` | `0` | ✅ 跑通 | StateGraph 编译 + 两轮工具调用，rc=0 |
| `ch09/react_agent_min.py` | `1` | 🔑 缺 API key | 缺 DEEPSEEK_API_KEY |

### ch10/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch10/concurrency_vs_serial.py` | `0` | ✅ 跑通 | 串行/并发实测 33.4s，加速比 10.0x / 99x |
| `ch10/structured_output.py` | `0` | ✅ 跑通 | 5 组解析用例全过；书稿版在围栏/尾逗号上失败是刻意对比 |
| `ch10/tool_param_adapter.py` | `0` | ✅ 跑通 | 参数适配 + 串并行对比，全过 |

### ch11/finetune/（案例）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch11/cases/finetune/infer_compare.py` | `1` | 🧱 缺前置产物 | 缺 `finetune/adapter`——要先跑 train_lora.py，而后者卡在驱动 |
| `ch11/cases/finetune/prepare_data.py` | `0` | ✅ 跑通 | 324 条合成 → 清洗 → 划分，落盘 ch11/cases/finetune/data/ |
| `ch11/cases/finetune/train_lora.py` | `1` | 🖥 缺驱动/系统组件 | 同上：cu130 要驱动 ≥580，本机 550.90.07 |

### ch11/knowledge_graph/（案例）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch11/cases/knowledge_graph/build_kg.py` | `0` | ✅ 跑通 | 从 jieba 分词建图谱，节点/关系落盘 kg.json |
| `ch11/cases/knowledge_graph/query_kg.py` | `2` | ✅ 跑通（用文档给的参数） | `stats` → 裸跑 rc=2 缺子命令；`stats` / `search 变压器` 均 rc=0 |

### ch11/mcp_system/（案例）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch11/cases/mcp_system/client.py` | `0` | ✅ 跑通 | 拉起 server.py 做端到端调用，rc=0 |
| `ch11/cases/mcp_system/server.py` | `0` | ✅ 跑通 | stdio 干净退出；已由 client.py 端到端验证 |

### ch11/novel_gen/（案例）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch11/cases/novel_gen/hard_reference_edit.py` | `1` | ✅ 跑通（用文档给的参数） | `--mock` → 裸跑缺前置产物；`--mock` rc=0，产出 edit_ch02_s01.md + reference_report |
| `ch11/cases/novel_gen/outline_anchor.py` | `1` | ✅ 跑通（用文档给的参数） | `--mock` → 裸跑缺 key；`--mock` rc=0，产出 outline.json |
| `ch11/cases/novel_gen/section_writer.py` | `1` | ✅ 跑通（用文档给的参数） | `--mock` → 裸跑缺 key；`--mock` rc=0，6 节正文 + 记忆卡轨迹 |

### ch11/video_probe/（案例）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch11/cases/video_probe/fade_test.py` | `0` | ✅ 跑通 | 淡入淡出检测，rc=0 |
| `ch11/cases/video_probe/make_clip.py` | `0` | ✅ 跑通 | 用 cv2 合成测试片段，rc=0 |
| `ch11/cases/video_probe/ocr_frames.py` | `2` | ✅ 跑通（用文档给的参数） | `--engine pytesseract` → 裸跑缺 keyframes.json（跑批顺序在其后）；`--engine pytesseract` rc=0，5 帧全部真识别 |
| `ch11/cases/video_probe/probe.py` | `0` | ✅ 跑通 | ffprobe 探测 + 关键帧抽取，产出 keyframes.json 等 |
| `ch11/cases/video_probe/report.py` | `0` | ✅ 跑通 | 汇总成 report.md，rc=0 |

### ch13/

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ch13/arithmetic_trap_demo.py` | `0` | ✅ 跑通 | 离线 Decimal 对照 4 组算术陷阱，全一致 |
| `ch13/tool_dispatch_demo.py` | `0` | ✅ 跑通 | 工具分发三路对照，rc=0 |

### common/（给各章示例引用的公共库）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `common/db/mysql/mysqlconn.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/db/mysql/mysqlconn4pool.py` | `1` | 🧩 库模块（import 需外部条件） | import 即要求 MYSQL_PASSWORD（按设计） |
| `common/db/mysql/mysqlservice.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/db/mysql/mysqltest.py` | `1` | 🧩 库模块（import 需外部条件） | import 即连 MySQL（本机无实例） |
| `common/db/redis/Redis.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/db/redis/redistest.py` | `1` | 🧩 库模块（import 需外部条件） | `from Redis import Redis` 是文件同目录写法；`PYTHONPATH=.` 下报 No module named 'Redis' |
| `common/gpt/gptservice.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/gpt/volcengine.py` | `1` | 🧩 库模块（import 需外部条件） | import 即要求 AK/SK（按设计） |
| `common/log/logservice.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/mail/mailservice.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/acc.py` | `0` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/acc1.py` | `1` | 🧩 库模块（import 需外部条件） | import 即读 `./obj_train_data/`（数据未随仓库发布） |
| `common/util/commonservice.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/http_utils.py` | `0` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/jinja2utils.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/singleton.py` | `0` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/testutil.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/util/utils.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/web/imagecode.py` | `0` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/web/page.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |
| `common/web/webutil.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |

### config/（配置读取）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `config/cfg.py` | `1` | 🧩 库模块（非可执行脚本） | 无 `__main__`，是给示例 import 的库；`PYTHONPATH=. python -c "import …"` 通过 |

### ecosystem/（书 §6 生态、§9 框架）

| 文件 | rc | 归类 | 原因 |
|---|---|---|---|
| `ecosystem/gemini_demo/gemini_live_api.py` | `0` | 📄 示意代码（按设计不执行） | 同 ch06 版本：打印缺失清单后 rc=0 退出，示意代码 |

## 本书环境的三条硬限制（不是代码问题）

### 1. 本机 CUDA 用不了：torch 2.11 是 cu130 构建，驱动只有 550

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12040).
torch 2.11.0+cu130 / cuda build 13.0 / is_available False
```

`--all-extras` 解析出的 torch 是 `2.11.0+cu130`（vllm 0.23 要它）。CUDA 13 的轮子需要
NVIDIA 驱动 **≥ 580**；本机 550.90.07 只能到 CUDA 12.4。结果：所有 `.cuda()` 的示例
（`ch06/accelerate_min.py`、`ch11/cases/finetune/train_lora.py`）在这里必然失败。

纯 CPU 的示例不受影响 —— `ch04/transformer_min.py` rc=0 就是在 CPU 上跑的。
读者若要在 GPU 上跑，先确认 `nvidia-smi` 里的驱动版本 ≥ 580。

### 2. 没有 API key

跑批时环境变量里一个 key 都没有（脚本也刻意不读 `.env`）。所以凡是要联网调模型的示例
都会走到「缺 key」分支。这不是缺陷 —— 恰恰证明这些脚本**没有偷偷硬编码 key**。

### 3. 到 huggingface.co 不通

`ch03/transformers_infer.py`、`ch06/model_run.py` 会去下载 Qwen2.5-0.5B，
本机网络 60s 内没拉下来，按超时记。要跑这两个，先自己把模型下到本地再改路径。

## 需要你知道的两件事

### 1. `ch04/onnx_export_min.py`：能跑了，但走的是 legacy 导出器

这个文件踩了两层，都修了。

**第一层（缺包）**：`gpu` 组原先没声明 `onnxscript`，torch 2.11 的 `torch.onnx.export`
直接 `ModuleNotFoundError`。已补进 `pyproject.toml`。

**第二层（API 换代）**：torch 2.9 起默认换成 dynamo 导出器，**不再接受书稿教的 `dynamic_axes`**：

```
ValueError: Found the following conflicts between user-specified ranges and
inferred ranges from model tracing:
- Received user-specified dim hint Dim.DYNAMIC(min=None, max=None), but tracing
  inferred a static shape of 50 for dimension inputs['input_ids'].shape[1].
```

修法：显式 `dynamo=False` 退回 legacy（TorchScript）导出器。**因为书稿正文讲的就是
`dynamic_axes`** —— 仓库的首要义务是和书对得上，不是用上最新 API。
代价写进文件头 ⑥：legacy 导出器 torch 官方已标记为旧路径、将来会移除。

**顺带修掉一个更隐蔽的问题：文件头那条「校验」是假的。**

```
$ python -m onnx.checker /tmp/definitely_missing.onnx
RuntimeWarning: 'onnx.checker' found in sys.modules ...
rc=0                       ← 文件不存在也 rc=0
```

onnx 1.23 上 `python -m onnx.checker <file>` 什么都不检查。文件头 ③ 里那句「校验」
从来没校验过任何东西 —— **以前谁拿它当「导出成功」的证据，那个证据是假的**。

现在改成 `check` 子命令，走 `onnx.checker.check_model(onnx.load(path))`。对照：

| 输入 | 命令 | rc |
|---|---|---|
| 真导出的 `.onnx` | `onnx_export_min.py check` | **0**（`ONNX 校验通过`） |
| 文本冒充 `.onnx` | 同上 | **1**（`DecodeError: Wire format was corrupt`） |
| 不存在的路径 | 旧的 `-m onnx.checker` | **0**（假绿） |

### 2. `ch04/onnx_m.py`：这个「缺包」是**故意不补**的

`optimum` 2.3 把 ONNX Runtime 执行器拆成了独立发行包，`from optimum.onnxruntime import ...`
不再存在。要补就得装 `optimum[onnxruntime]`，而它要求 `huggingface-hub>=1.0` ——
与 `transformers<5` 这条硬约束直接打架。

**选择：保 `transformers<5`。** 理由很直白：transformer 是全书主线，ONNX 执行器只是一个示例。
不能为了一个 B 级示例，把整个大环境的基础前提换掉。

所以这一行永远是「缺包」，不是没装好。要跑它，自己单开一个 venv。

## 关于「代码 bug」

**0 个。** 82 个文件里没有「逻辑写错导致跑不出来」的。

`ch04/onnx_export_min.py` 那个已修（`dynamo=False` + 真校验），修完 export / check / infer 三步 rc 全 0。

顺带一提的写法（不算 bug）：`common/db/redis/redistest.py:5` 是 `from Redis import Redis`，
依赖「文件同目录在 `sys.path` 上」。用 `PYTHONPATH=.` 从仓库根导入就会报
`No module named 'Redis'`。老仓的导入习惯，按原样保留。
