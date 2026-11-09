# 章节 ↔ 代码对照

> 按章节查、按文件查，都看这一张表。
>
> **说在前面**：能跑 ≠ 一定能跑。这张表只告诉你「这个文件对应书里哪一段、它是干什么的」。
> 真正跑之前，先看 [README](README.md) 的「环境是分层的」一节，再看 [ENV_MATRIX.md](ENV_MATRIX.md) 里对应那一行。


## 第 2 章 大模型实战环境准备

| 文件 | 这个文件干什么 |
|---|---|
| [`ch02/model_run.py`](ch02/model_run.py) | ch02 §2.4.2 网络阻力：消灭 |
| [`ch02/openai_test.py`](ch02/openai_test.py) | ch02 §2.2.4 在线推理服务：零部署直接调用 |

## 第 3 章 大模型快速体验

| 文件 | 这个文件干什么 |
|---|---|
| [`ch03/api_call_deepseek.py`](ch03/api_call_deepseek.py) | ch03 §3.1 方法1：直接调用现有大模型服务 / 「API 调用：开发者的正确打开方式」 |
| [`ch03/transformers_infer.py`](ch03/transformers_infer.py) | ch03 §3.2.1 基于 Transformers 库调用大模型文件 |

## 第 4 章 大模型原理和技术实现

| 文件 | 这个文件干什么 |
|---|---|
| [`ch04/model_architectures.py`](ch04/model_architectures.py) | 同一条 Transformer 理论下的三种模型构型 |
| [`ch04/model_parser_c.py`](ch04/model_parser_c.py) | 用 print(model) 看大模型的逻辑结构 |
| [`ch04/model_run.py`](ch04/model_run.py) | 用 AutoClasses 加载模型并对话生成的完整示例 |
| [`ch04/onnx_export_min.py`](ch04/onnx_export_min.py) | PyTorch → ONNX → ONNX Runtime 全链路最小示例 |
| [`ch04/onnx_m.py`](ch04/onnx_m.py) | 用 optimum 的 ORTModelForCausalLM 跑已导出的 ONNX 模型 |
| [`ch04/safetensors_read.py`](ch04/safetensors_read.py) | safetensors 零拷贝（mmap）读取最小示例 |
| [`ch04/transformer_min.py`](ch04/transformer_min.py) | 手写单头自注意力 + 多头注意力（最小 Transformer 核心） |
| [`ch04/unsloth_wt1.py`](ch04/unsloth_wt1.py) | Unsloth + LoRA 微调完整脚本 |

## 第 5 章 如何让大模型「耳聪目明」

| 文件 | 这个文件干什么 |
|---|---|
| [`ch05/openai_protocol_http.py`](ch05/openai_protocol_http.py) | 裸 HTTP 调 OpenAI 兼容协议（requests，不用 SDK） |

## 第 6 章 大模型技术生态

| 文件 | 这个文件干什么 |
|---|---|
| [`ch06/accelerate_min.py`](ch06/accelerate_min.py) | ch06 §6.2.3 Accelerate 的最小可跑版：**同一份训练脚本，改 5 行就到手**。 |
| [`ch06/dspy_min.py`](ch06/dspy_min.py) | ch06 §6.7 DSPy 的最小可跑版：**不写 Prompt，只声明输入输出**，让编译器去挑示例与措辞。 |
| [`ch06/gemini_live_api.py`](ch06/gemini_live_api.py) | 书里这段要教的是「打断（interruption）语义」：服务端检测到用户开口就停止生成， |
| [`ch06/model_run.py`](ch06/model_run.py) | ch06 §6.2.4 —— 用 transformers 加载模型并生成 |
| [`ch06/openai_test.py`](ch06/openai_test.py) | ch06 §6.2.6 —— OpenAI SDK 两种最小用法 |
| [`ch06/rag_hyde_rrf.py`](ch06/rag_hyde_rrf.py) | ch06 §6.6.1.2 —— RAG 2.0 的两台手术：HyDE 与 RRF 融合 |
| [`ch06/torch_env_check.py`](ch06/torch_env_check.py) | ch06 §6.2.1 —— PyTorch 环境自检 |
| [`ch06/transformers_config.py`](ch06/transformers_config.py) | ch06 §6.2.4 —— 只读 config、按 config 搭骨架 |
| [`ch06/troch_train_c_moe_v1.py`](ch06/troch_train_c_moe_v1.py) | ch06 §6.3.1 —— 用 PyTorch 手搭一个 MoE 解码器并训练 |
| [`ch06/unsloth_wt1.py`](ch06/unsloth_wt1.py) | ch06 §6.4.1 —— unsloth 4bit LoRA 微调完整流程 |
| [`ch06/vllm_t.py`](ch06/vllm_t.py) | ch06 §6.5.1 —— vLLM 原生 Python API 用法合集 |

## 第 7 章 MCP 协议：大模型的「神经突触」

| 文件 | 这个文件干什么 |
|---|---|
| [`ch07/function_calling_demo.py`](ch07/function_calling_demo.py) | 对应章节：ch07 §7.1.4 一个 Function Calling 的完整示例 |
| [`ch07/mcp_client.py`](ch07/mcp_client.py) | MCP 客户端最小闭环：连上 SSE 服务端 → 列工具 → 读资源 |
| [`ch07/mcp_count_qw.py`](ch07/mcp_count_qw.py) | MCP + 大模型：模型自己决定调哪个 MCP 工具（靠系统提示，不是原生 function calling） |
| [`ch07/mcp_server.py`](ch07/mcp_server.py) | 完整 MCP Server：resource / resource template / prompt 三种原语都演示 |
| [`ch07/mcp_server1.py`](ch07/mcp_server1.py) | 专演示 prompt 原语的服务端（资源 + 提示词两件） |
| [`ch07/mcp_server_mysql.py`](ch07/mcp_server_mysql.py) | 把 MySQL 包成 MCP 工具：query_database / list_tables / list_databases |
| [`ch07/stock_server.py`](ch07/stock_server.py) | 对应章节：ch07 §7.3.2 MCP开发神器：FastMCP |
| [`ch07/task_server.py`](ch07/task_server.py) | 对应章节：ch07 §7.3.3 开发MCP服务器端 |
| [`ch07/weather_server.py`](ch07/weather_server.py) | 对应章节：ch07 §7.2.2 自己动手写一个 MCP Server |

## 第 8 章 大模型应用技术选型

| 文件 | 这个文件干什么 |
|---|---|
| [`ch08/openai_cloud_stream.py`](ch08/openai_cloud_stream.py) | ch08 §8.3.1 路径A 的可运行版：用 OpenAI SDK 流式调用云端推理服务， |
| [`ch08/rag_chroma_demo.py`](ch08/rag_chroma_demo.py) | ch08 §8.3.3 RAG 四步的最小可跑版：向量化 → Chroma 检索 → 拼 Prompt → 问模型。 |
| [`ch08/vllm_openai_compat.py`](ch08/vllm_openai_compat.py) | ch08 §8.3.1 路径B 的可运行版：连 vLLM 暴露的 OpenAI 兼容端点做一次 chat 调用。 |

## 第 9 章 大模型和智能体（AGENT）

| 文件 | 这个文件干什么 |
|---|---|
| [`ch09/react_agent_min.py`](ch09/react_agent_min.py) | 手写最小 ReAct Agent：想 → 调工具 → 看结果 → 再想，直到给出答案 |
| [`ch09/crewai_demo/crewai_quickstart.py`](ch09/crewai_demo/crewai_quickstart.py) | 多角色协作：每个 Agent 一个角色一个任务，串起来跑 |
| [`ch09/langgraph_demo/langgraph_state_graph.py`](ch09/langgraph_demo/langgraph_state_graph.py) | 用 LangGraph 状态图表达 think→act 循环 |

## 第 10 章 大模型应用架构设计

| 文件 | 这个文件干什么 |
|---|---|
| [`ch10/concurrency_vs_serial.py`](ch10/concurrency_vs_serial.py) | 串行 vs 并发的耗时对比（同一批任务两种跑法） |
| [`ch10/structured_output.py`](ch10/structured_output.py) | 让模型稳定吐 JSON：从解析自由文本到约束输出格式 |
| [`ch10/tool_param_adapter.py`](ch10/tool_param_adapter.py) | 工具参数适配器 + 失败隔离（模型给的参数和工具要的对不上时怎么兜） |

## 第 11 章 大模型应用案例

| 文件 | 这个文件干什么 |
|---|---|
| [`ch11/cases/mcp_system/client.py`](ch11/cases/mcp_system/client.py) | 连上 server.py，走完一整圈 |
| [`ch11/cases/mcp_system/server.py`](ch11/cases/mcp_system/server.py) | MCP 服务端：3 个工具 + 1 个 Resource + 1 个 Prompt |
| [`ch11/cases/knowledge_graph/build_kg.py`](ch11/cases/knowledge_graph/build_kg.py) | 对应章节：第 11 章 §11.4「如何构建大模型知识图谱」。 |
| [`ch11/cases/knowledge_graph/query_kg.py`](ch11/cases/knowledge_graph/query_kg.py) | 在落盘的图谱上查询 |
| [`ch11/cases/novel_gen/hard_reference_edit.py`](ch11/cases/novel_gen/hard_reference_edit.py) | 硬引用修改：改一段时把原文贴进 prompt，并列出引用过它的其它节 |
| [`ch11/cases/novel_gen/outline_anchor.py`](ch11/cases/novel_gen/outline_anchor.py) | 一句话：先把「大纲 + 角色设定 + 章节摘要」做成一张记忆卡，每次生成都把它当 prompt 前缀。 |
| [`ch11/cases/novel_gen/section_writer.py`](ch11/cases/novel_gen/section_writer.py) | 分节推进：一次写一节，写完更新记忆卡并落盘，长篇因此可续写 |
| [`ch11/cases/video_probe/fade_test.py`](ch11/cases/video_probe/fade_test.py) | 渐变过渡检测：为什么「多切出一段」不是误检 |
| [`ch11/cases/video_probe/make_clip.py`](ch11/cases/video_probe/make_clip.py) | 用 ffmpeg 合成一段带已知切点的测试视频 |
| [`ch11/cases/video_probe/ocr_frames.py`](ch11/cases/video_probe/ocr_frames.py) | 关键帧 OCR：把画面上的文字抽出来，三种引擎可切 |
| [`ch11/cases/video_probe/probe.py`](ch11/cases/video_probe/probe.py) | 视频镜头切分：按画面差异找出场景边界 |
| [`ch11/cases/video_probe/report.py`](ch11/cases/video_probe/report.py) | 原则：报告里出现的每个数字都从 JSON 里读出来，**不手写、不估算**。 |
| [`ch11/cases/finetune/infer_compare.py`](ch11/cases/finetune/infer_compare.py) | 书稿 §11.3.4 给了两种验证：LLM-as-Judge 盲评 + 关键路径回测。 |
| [`ch11/cases/finetune/prepare_data.py`](ch11/cases/finetune/prepare_data.py) | 对应第 11 章 §11.3.1 / §11.3.2 / §11.3.3。 |
| [`ch11/cases/finetune/train_lora.py`](ch11/cases/finetune/train_lora.py) | LoRA 训练：拿第 1 步造的数据微调 Qwen2.5-0.5B（书里 §11.3 没给训练代码，本仓补） |

## 第 13 章 边界与克制：大模型不该做的「深水区」

| 文件 | 这个文件干什么 |
|---|---|
| [`ch13/arithmetic_trap_demo.py`](ch13/arithmetic_trap_demo.py) | 「确定性算术」陷阱的三组 prompt，用 Decimal 给出精确答案 |
| [`ch13/tool_dispatch_demo.py`](ch13/tool_dispatch_demo.py) | 「LLM 只当指令调度员」：把算术交给代码，模型只决定调什么 |

## 公共模块（多章复用）

| 文件 | 这个文件干什么 |
|---|---|
| [`common/db/mysql/mysqlconn.py`](common/db/mysql/mysqlconn.py) | MySQL 单连接封装 |
| [`common/db/mysql/mysqlconn4pool.py`](common/db/mysql/mysqlconn4pool.py) | MySQL 连接池封装 |
| [`common/db/mysql/mysqlservice.py`](common/db/mysql/mysqlservice.py) | MySQL 增删改查的薄封装（saveRow / 查询） |
| [`common/db/mysql/mysqltest.py`](common/db/mysql/mysqltest.py) | MySQL 连接与查询的调用示例 |
| [`common/db/redis/Redis.py`](common/db/redis/Redis.py) | Redis 连接池与常用读写封装 |
| [`common/db/redis/redistest.py`](common/db/redis/redistest.py) | Redis 封装的调用示例 |
| [`common/util/acc.py`](common/util/acc.py) | 检测框 IOU 等评估小工具 |
| [`common/util/acc1.py`](common/util/acc1.py) | 评估工具的早期版本 |
| [`common/util/commonservice.py`](common/util/commonservice.py) | 通用响应结构（code/message/data） |
| [`common/util/http_utils.py`](common/util/http_utils.py) | HTTP 请求/JSON POST/文件上传的薄封装 |
| [`common/util/jinja2utils.py`](common/util/jinja2utils.py) | Jinja2 模板渲染小工具 |
| [`common/util/singleton.py`](common/util/singleton.py) | 单例装饰器 |
| [`common/util/testutil.py`](common/util/testutil.py) | 测试/调试辅助（含一个可配置目标的 HTTP 探测示例） |
| [`common/util/utils.py`](common/util/utils.py) | 通用小工具（时间、字符串、重试等） |
| [`common/gpt/gptservice.py`](common/gpt/gptservice.py) | OpenAI 兼容 GPT 客户端（示例统一用它读 `DEEPSEEK_API_KEY`） |
| [`common/gpt/volcengine.py`](common/gpt/volcengine.py) | 火山方舟（Ark）SDK 的最小调用 |
| [`common/web/imagecode.py`](common/web/imagecode.py) | 图形验证码生成 |
| [`common/web/page.py`](common/web/page.py) | 分页 HTML 片段工具 |
| [`common/web/webutil.py`](common/web/webutil.py) | Web 层通用工具 |
| [`common/mail/mailservice.py`](common/mail/mailservice.py) | 发邮件封装（SMTP） |
| [`common/log/logservice.py`](common/log/logservice.py) | 日志初始化（按名字取 logger） |

## 配置

| 文件 | 这个文件干什么 |
|---|---|
| [`config/cfg.py`](config/cfg.py) | 配置加载（仓库根定位 + 配置项读取） |

## 生态示例

| 文件 | 这个文件干什么 |
|---|---|
| [`ecosystem/gemini_demo/gemini_live_api.py`](ecosystem/gemini_demo/gemini_live_api.py) | 书里这段要教的是「打断（interruption）语义」：服务端检测到用户开口就停止生成， |

## 第 11 章的五个案例

第 11 章按案例分目录，每个目录自带 README（怎么跑、依赖什么、数据从哪来）：

| 案例 | 目录 | 书中位置 |
|---|---|---|
| 视频识别游泳者溺水 | [`ch11/cases/video_probe/`](ch11/cases/video_probe/) | §11.1 |
| 长篇小说生成 | [`ch11/cases/novel_gen/`](ch11/cases/novel_gen/) | §11.2 |
| 对一个大模型进行微调 | [`ch11/cases/finetune/`](ch11/cases/finetune/) | §11.3 |
| 构建大模型知识图谱 | [`ch11/cases/knowledge_graph/`](ch11/cases/knowledge_graph/) | §11.4 |
| 手搓一个完整的 MCP 体系 | [`ch11/cases/mcp_system/`](ch11/cases/mcp_system/) | §11.5 |

## 没有列进来的东西

- 与本书内容无关的历史脚本：已从本仓移除（本仓只保留与书对应的代码）。
- 各案例的 `outputs/` 运行产物：不入库。自己跑一遍就有了。
- 模型权重、数据集、虚拟环境：一律不入库，见 `.gitignore`。

