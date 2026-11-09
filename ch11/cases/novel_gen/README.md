# 长篇小说生成（§11.2）

## 这个案例干什么

长篇生成的难点不是「写得好」，是**写到第 40 节还记得第 3 节埋的伏笔**。这个案例拆出三个可单独验证的机制：

| 文件 | 机制 | 解决什么 |
|---|---|---|
| `outline_anchor.py` | 提纲锚定 | 每次生成都把提纲和「记忆卡」放进 prompt 前缀，模型不会跑题 |
| `section_writer.py` | 分节推进 | 一次只写一节，写完更新记忆卡并落盘，长篇因此可续写 |
| `hard_reference_edit.py` | 硬引用修改 | 改某一段时，把目标段原文贴进 prompt，并列出引用了它的其它节，避免改出前后矛盾 |

三个机制都**不依赖特定模型**，换成任何 OpenAI 兼容接口都能用。

## 对应书哪一节

第 11 章 §11.2「长篇小说生成」。

## 怎么跑

**离线（不需要任何 API key）**：三个脚本都支持 `--mock`，用内置的假模型验证整个流程的骨架。

```bash
python ch11/cases/novel_gen/outline_anchor.py --mock
python ch11/cases/novel_gen/section_writer.py --mock
python ch11/cases/novel_gen/hard_reference_edit.py --mock
```

**接真模型**：

```bash
export DEEPSEEK_API_KEY=...          # 或其他 OpenAI 兼容 key，见 .env.example
python ch11/cases/novel_gen/outline_anchor.py
python ch11/cases/novel_gen/section_writer.py
```

模型名可以用环境变量 `NOVEL_MODEL` 覆盖（默认 `deepseek-chat`）。

## 依赖什么

- Python 标准库（`argparse` / `json` / `re` / `pathlib`），**没有第三方依赖**。
- 接真模型时需要一个 OpenAI 兼容的 API key；走的是仓库里的 `common/gpt/gptservice.py`（读 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`）。
- **没有 key 时**：`--mock` 模式完全可跑；不带 `--mock` 会明确报错说缺哪个环境变量，不会静默返回空结果。

## 数据从哪来

- 提纲 `outline.json`、记忆卡：`outline_anchor.py` 首次运行会**生成一份示例提纲**，之后读写同一份文件。
- 正文：写在 `outputs/`（不入库，自己跑就有了）。
- 没有任何外部语料或模型权重。

## 提醒

三个机制证明的是「工程结构」，不是「文学生成质量」。`--mock` 模式下的文本是占位内容，别拿它评价效果。
