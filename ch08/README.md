# 第 8 章 大模型应用技术选型

## 这章讲什么

同一个需求，可以云上调、可以本地跑、可以加检索、可以自己微调 —— 这章讲的是**怎么选**，以及每种选法在工程上意味着什么代价。

代码部分只给三条主线的最小实现：云端流式、本地推理服务、RAG 检索增强。选型判断本身在正文里。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`openai_cloud_stream.py`](openai_cloud_stream.py) | 云端流式输出：一边生成一边打印，不等整段返回 | **B 要条件**（缺 key 时按设计退出，退出码 2；本机到公网不通，真实云端流式未验证） |
| [`vllm_openai_compat.py`](vllm_openai_compat.py) | 本地推理服务 + OpenAI 兼容客户端，两边都在这一个文件里 | **A 可真跑（`--stub` 模式）**（桩服务 + 书稿原样客户端，1 次请求闭环；真 vLLM 要装 vllm + 权重） |
| [`rag_chroma_demo.py`](rag_chroma_demo.py) | RAG 最小闭环：chromadb 存向量 → 检索 → 拼上下文 → 生成 | **A 可真跑（检索段）**（5 条入库、Top-1 命中；最后一步真生成要 key，无 key 时跳过） |

## 怎么跑

```bash
# 1. 先跑不依赖任何服务的那个：本地推理服务的桩模式
python3 ch08/vllm_openai_compat.py --stub

# 2. RAG 检索闭环（不需要 key，最后一步自动跳过）
python3 ch08/rag_chroma_demo.py

# 3. 云端流式（要 key）
export OPENAI_API_KEY=你的key
# export OPENAI_BASE_URL=https://api.deepseek.com/v1
python3 ch08/openai_cloud_stream.py

# 4. 真连 vLLM（要先起服务）
vllm serve /你的/模型目录 --port 8000
python3 ch08/vllm_openai_compat.py --base-url http://localhost:8000/v1
```

## 依赖与前置

| 依赖 | 谁需要 | 说明 |
|---|---|---|
| `openai` | 三个文件都 import | 本仓库实测 3.3.0 |
| `chromadb` | `rag_chroma_demo.py` | 本仓库实测 1.1.1；**只能用装了它的那个解释器跑** |
| `OPENAI_API_KEY` | 云端流式、RAG 的生成步骤 | 无 key 时两个脚本都按设计退出/降级，不会假跑 |
| `vllm` + 模型权重 | 真连模式 | 10GB 显存能放 0.5B~1.3B 级模型 |

## 常见报错

**`ModuleNotFoundError: No module named 'chromadb'`**
装错解释器了 —— 这个包通常只装在某个专门的虚拟环境里。用 `python3 -c "import chromadb"` 确认，再用同一个解释器跑脚本。

**`APIConnectionError` / `Network is unreachable`**
本机到公网不通，或者 `OPENAI_BASE_URL` 指错了地址。这与代码无关：换网络、换端点，或改用 `--stub` 模式先验证逻辑。

**`--stub` 模式跑通了，真 vLLM 连不上**
`--stub` 只证明「客户端 + 协议 + 报文」这段是对的，不证明远端服务可用。真连模式报连接错误，先看 `vllm serve` 有没有起来、端口对不对。

**chromadb 提示 telemetry / 收集信息**
它默认会尝试上报使用统计。离线环境会看到一条警告，不影响功能；需要的话设 `ANONYMIZED_TELEMETRY=False`。
