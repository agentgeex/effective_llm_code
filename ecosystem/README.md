# ecosystem/ —— 生态示例（补充目录）

这一层放的是「书里讲了、但按章节归档不太顺」的生态示例。**主入口仍然是按章目录**，这个目录只做补充。

## 这里有什么

| 文件 | 对应书 | 能跑吗 | 需要什么 |
|---|---|---|---|
| [`gemini_demo/gemini_live_api.py`](gemini_demo/gemini_live_api.py) | 第 6 章 §6.2.6 Gemini Live API | **C 示意** | `GEMINI_API_KEY` + 麦克风/播放设备；无 key 时只做自检（打印缺什么） |
| `dspy_demo/` | —— | —— | 说明见该目录 README：DSPy 的示例统一放在 [`../ch06/dspy_min.py`](../ch06/dspy_min.py) |

其余生态示例都在对应章节目录里：

- Accelerate → [`../ch06/accelerate_min.py`](../ch06/accelerate_min.py)
- DSPy → [`../ch06/dspy_min.py`](../ch06/dspy_min.py)
- vLLM → [`../ch06/vllm_t.py`](../ch06/vllm_t.py)
- LangGraph / CrewAI → [`../ch09/`](../ch09/)
- chromadb RAG → [`../ch08/rag_chroma_demo.py`](../ch08/rag_chroma_demo.py)

## 为什么目录名都带 `_demo` 后缀

因为 Python 会把「当前目录下的同名目录」当成包，遮住真正要 import 的那个 —— 这叫**同名遮蔽**。

| 目录名 | 会不会真撞 |
|---|---|
| `accelerate_demo/` | 会 —— `accelerate` 是 pip 包名 |
| `dspy_demo/` | 会 —— `dspy` 是 pip 包名 |
| `gemini_demo/` | 不会（真正的包是 `google-genai`，模块名 `google.genai`），但为了让命名统一也加了后缀 |

判断某个名字有没有被遮蔽，一行就够：

```bash
python3 -c "import mcp; print(mcp.__file__)"
```

如果打印出来的是**仓库里的目录**而不是 `site-packages`，就是被遮蔽了 —— 这时换个目录再跑（在 `/tmp` 下跑一次对照），或者干脆改名。

## 怎么跑

```bash
# Gemini（无 key 时只做自检）
export GEMINI_API_KEY=你的key
python3 ecosystem/gemini_demo/gemini_live_api.py

# 不带 key 跑，会打印缺失清单并以 0 退出
python3 ecosystem/gemini_demo/gemini_live_api.py
```

## 定级口径（与全仓一致）

- **A**：依赖可装、本机亲手跑通、有输出
- **B**：代码完整，但要 key / 外部服务 / 大显存；都给了无凭证的降级路径
- **C**：需要设备或特定服务，只能示意，不可直接跑通

## 与书稿的差异

每个文件顶部都有「对应书稿小节 + 本文件相对书稿的改动」注释，写清「书稿原写法 → 会报什么错 → 改成了什么」。
按文件查的完整清单见 [`../INDEX.md`](../INDEX.md)。
