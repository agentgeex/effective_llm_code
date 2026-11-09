# 第 11 章 大模型应用案例

## 这章讲什么

前面十章都是零件，这一章是五个完整的东西：**视频结构化**、**长篇小说生成**、**模型微调**、**知识图谱**、**MCP 体系**。

每个案例都按「能独立跑起来」的标准写 —— 有自己的 `README.md`、自己的数据、自己的跑法。你可以只挑感兴趣的那个看，不必按顺序。

## 本目录的文件

五个案例，各自一个子目录：

| 案例 | 干什么 | 需要什么 |
|---|---|---|
| [`cases/video_probe/`](cases/video_probe/) | 视频镜头切分 + 关键帧 OCR + 结构化报告：把一段视频变成可检索的片段表 | ffmpeg、OpenCV；OCR 还要 tesseract |
| [`cases/novel_gen/`](cases/novel_gen/) | 长篇生成：提纲锚定 + 分节推进 + 硬引用修改 | 无（`--mock`）；真生成要 API key |
| [`cases/finetune/`](cases/finetune/) | 微调全流程：造数据 → 清洗/划分 → LoRA 训练 → 前后对比 | 第 1 步只要 `scikit-learn`；训练要 GPU + 权重 |
| [`cases/knowledge_graph/`](cases/knowledge_graph/) | 从中文语料抽实体关系建图，再在图上查询 | 只要 `jieba` |
| [`cases/mcp_system/`](cases/mcp_system/) | 手搓一个完整的 MCP 服务端 + 客户端，跑通整圈协议 | 只要 `mcp` 2.x |

每个子目录的 `README.md` 里有：这个案例干什么、对应书哪一节、怎么跑、依赖什么、数据从哪来。

## 怎么跑

**建议从最简单的两个开始**，它们不需要 GPU、不需要 key：

```bash
pip install jieba
python3 ch11/cases/knowledge_graph/build_kg.py       # 建图，几十秒
python3 ch11/cases/knowledge_graph/query_kg.py stats # 看图上有什么

pip install "mcp>=2.0"
python3 ch11/cases/mcp_system/client.py              # 端到端跑完一整圈协议
```

再往下按兴趣挑：

```bash
# 视频（要先装 ffmpeg 和 opencv）
python3 ch11/cases/video_probe/make_clip.py
python3 ch11/cases/video_probe/probe.py

# 长篇小说（--mock 不需要 key）
python3 ch11/cases/novel_gen/outline_anchor.py --mock

# 微调（第 1 步不需要 GPU）
pip install scikit-learn
python3 ch11/cases/finetune/prepare_data.py
python3 ch11/cases/finetune/train_lora.py     # 这一步要 GPU + 基座权重
```

## 依赖与前置

| 案例 | 依赖 |
|---|---|
| `video_probe` | 系统 `ffmpeg` / `ffprobe`；`opencv-python`、`numpy`；OCR 另需 `pytesseract` + 系统 `tesseract`（含中文语言包） |
| `novel_gen` | 标准库；真生成要 OpenAI 兼容 key（默认 `DEEPSEEK_API_KEY`） |
| `finetune` | `scikit-learn`（造数据）；`torch` / `transformers` / `peft` / `datasets`（训练）；基座权重默认 `Qwen2.5-0.5B-Instruct` |
| `knowledge_graph` | `jieba` |
| `mcp_system` | `mcp` 2.x |

**五个案例都不依赖任何在线服务**（除 `novel_gen` 的真生成那条路）。也不需要 GPU，除非你真要训微调。

## 常见报错

**`ModuleNotFoundError: No module named 'cv2'`**
装的是 `opencv-python`，但导入名是 `cv2`。`pip install opencv-python opencv-python-headless`，后者适合无显示器的服务器。

**`ModuleNotFoundError: No module named 'sklearn'`**
包名是 `scikit-learn`，导入名是 `sklearn`。名字不一样是历史原因，不是装错了。

**`ModuleNotFoundError: No module named 'peft'`**
微调三个包要一起装：`torch` / `transformers` / `peft`。少哪个它就报哪个，按报错补即可。

**`mcp` 导入失败 / 说找不到 `FastMCP`**
`mcp_system` 是按 mcp **2.x** 写的（`MCPServer`）。装成 1.x 会失败；要么升 2.x，要么把导入改回 `mcp.server.fastmcp`。

**OCR 报找不到 `tesseract`**
`pytesseract` 只是个壳，真正的 OCR 引擎是系统包：`apt install tesseract-ocr tesseract-ocr-chi-sim`。只装 Python 包不够。

**跑完发现 `data/`、`outputs/` 变了**
那是运行产物，本来就不入库（见 `.gitignore`）。要留着就自己另存，别提交回仓库。
