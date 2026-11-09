# 第 3 章 大模型快速体验

## 这章讲什么

两条最直接的路：一是**直接调 API**，不用管模型跑在哪；二是**在自己机器上加载模型**，用 `transformers` 跑推理。这一章让你在十分钟内看到大模型的输出，并且看清这两条路各自适合什么场景。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`api_call_deepseek.py`](api_call_deepseek.py) | 走 DeepSeek 的 OpenAI 兼容接口，最小的一次对话调用 | **B 要条件**（缺 `DEEPSEEK_API_KEY`；端点可达性已实测：无效 key 会收到 `401 authentication_error`，不是网络问题） |
| [`transformers_infer.py`](transformers_infer.py) | 用 `transformers` 在本地加载模型并生成，不经过任何在线服务 | **A 可真跑**（本机实跑 rc=0） |

## 怎么跑

先跑 API 那条，最快看到结果：

```bash
export DEEPSEEK_API_KEY=你的key
python3 ch03/api_call_deepseek.py
```

再跑本地推理那条（模型要先备好）：

```bash
python3 ch03/transformers_infer.py
```

两个脚本都在仓库根目录执行。换模型/换模型路径的入口在脚本开头的常量里，不用改中间逻辑。

## 依赖与前置

| 跑哪个 | 需要什么 |
|---|---|
| `api_call_deepseek.py` | `openai` SDK；`DEEPSEEK_API_KEY`；能访问 `api.deepseek.com` |
| `transformers_infer.py` | `torch` + `transformers`；一个本地模型目录（约 1GB 起） |

## 常见报错

**`401 authentication_error`**
key 不对或没设。这个报错其实是**好消息** —— 说明网络和端点都通，只差凭证。如果换成连接超时或 DNS 失败，那才是网络问题。

**`ModuleNotFoundError: No module named 'openai'`**
装错解释器了。`python3 -m pip show openai` 看装在哪，或者直接用 `python3 -m pip install openai`。

**本地推理报找不到模型 / 目录不存在**
`transformers_infer.py` 里的模型路径是占位符，要改成你自己的。相对路径是相对**当前工作目录**，不是相对脚本文件 —— 所以务必在仓库根执行。
