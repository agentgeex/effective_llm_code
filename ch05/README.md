# 第 5 章 如何让大模型「耳聪目明」

## 这章讲什么

大模型本身只会「接话」。要让它真正干活，得给它装上三样东西：**能调工具**（函数调用）、**能看能听**（多模态输入）、**能被评测**（知道它到底行不行）。

这一章的代码不多，但它是从「聊天」走向「系统」的转折点 —— 后面第 7 章的 MCP、第 8 章的 RAG、第 9 章的 Agent，都是从这一章的协议开始的。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`openai_protocol_http.py`](openai_protocol_http.py) | 不用任何 SDK，用 `requests` 裸调 OpenAI 兼容协议 —— 把 HTTP 请求体摊开给你看 | **B 要条件**（要 key；端点可达性已实测） |

书里这一章还有几段 JSON 样例、SFT 数据格式、流水线 ASCII 图和一串外部 CLI 命令，它们不是独立脚本，不需要仓库单独提供文件 —— 内容已经收在这个脚本的 `build_payload()` 和各段注释里。

## 怎么跑

```bash
export DEEPSEEK_API_KEY=你的key
python3 ch05/openai_protocol_http.py
```

建议**打开这个文件边看边跑**：它的价值不在输出，在于让你看清楚「一次大模型调用」在最底层就是一个 HTTP POST，请求体里就是 `model` / `messages` / `tools` 这几个字段。看懂了这一层，后面所有 SDK 都只是包装。

## 依赖与前置

- `requests`（不是 `openai`！这一章刻意不用 SDK）
- 一个可用的 API key（`DEEPSEEK_API_KEY`）
- 能访问 `api.deepseek.com`

## 常见报错

**`401 authentication_error`**
key 缺失或不正确。这一章实测过：用无效 key 请求，回来的是明确的 401 JSON，而不是超时 —— 说明端点通、协议对，只差凭证。

**`ImportError: No module named 'requests'`**
`pip install requests`。注意别装成 `request`（少个 s，是个完全不同的包）。

**跟着书里的评测命令跑，报 `unrecognized arguments: --output-dir`**
书里 §5.3.4 那两行评测命令有两处笔误，照抄会失败：
- `--output-dir` 不是 evalscope 的选项，正确的是 `--work-dir`；
- `--datasets gsm8k,ceval,general_qa` 这种逗号串会被当成**一个**数据集名，要拆成多次传参或空格分隔。

这两处已在书稿勘误里记录（E-10 / E-11）。评测本身还需要模型权重和数据集，本章不建议真跑。
