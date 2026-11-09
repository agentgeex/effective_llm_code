# 第 9 章 大模型和智能体（AGENT）

## 这章讲什么

从「一次问答」到「自己决定下一步做什么」—— 这就是 Agent。这一章讲三件事：ReAct 循环（想 → 做 → 看结果 → 再想）、状态图编排（LangGraph）、以及多角色协作（CrewAI）。

三个框架写法差别很大，但内核是同一个循环。看懂一个，其余都是语法。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`react_agent_min.py`](react_agent_min.py) | 手写最小 ReAct：模型选工具 → 本地执行 → 结果回填 → 直到给答案 | **A 可真跑（`--mock`）**（2 次工具调用、3 轮出最终答案；真调模型要 key） |
| [`langgraph_demo/langgraph_state_graph.py`](langgraph_demo/langgraph_state_graph.py) | 用 LangGraph 的状态图表达 think→act 循环 | **A 可真跑**（纯本地，无需 key） |
| [`crewai_demo/crewai_quickstart.py`](crewai_demo/crewai_quickstart.py) | 多角色协作：给每个 Agent 一个角色和任务，串起来跑 | **B 要条件**（`--dry-run` 能跑通并打印 crew 结构；真 `kickoff()` 要 `OPENAI_API_KEY`） |

## 怎么跑

```bash
# 1. 先手写版，最能看清循环
python3 ch09/react_agent_min.py --mock

# 2. 状态图版
python3 ch09/langgraph_demo/langgraph_state_graph.py

# 3. 多角色版：先看结构
python3 ch09/crewai_demo/crewai_quickstart.py --dry-run

# 4. 真跑多角色（要 key）
export OPENAI_API_KEY=你的key
python3 ch09/crewai_demo/crewai_quickstart.py
```

`react_agent_min.py` 去掉 `--mock` 就会去调真模型，需要 `DEEPSEEK_API_KEY`。

## 依赖与前置

| 依赖 | 谁需要 | 说明 |
|---|---|---|
| 无 | `react_agent_min.py --mock` | 只用标准库 |
| `langgraph` | `langgraph_demo/` | 本仓库实测 1.2.18 |
| `crewai` | `crewai_demo/` | 本仓库实测 1.6.1 |
| `OPENAI_API_KEY` | crewai 真跑 | 无 key 时构造 `Agent()` 就会报错 |
| `DEEPSEEK_API_KEY` | `react_agent_min.py` 真跑 | 同上 |

## 常见报错

**`ImportError: Error importing native provider: OPENAI_API_KEY is required`**
CrewAI 在**构造对象时**就要 key，不是等到调用时。所以没有 key 连 `--dry-run` 之外的路径都进不去。要么设 key，要么只用 `--dry-run` 看结构。

**`--mock` 和真跑的差别在哪**
`--mock` 用内置的假模型代替真实调用，走的是**完全相同**的循环逻辑：选工具、执行、回填、再判断。它验证的是「循环对不对」，不验证「模型选得好不好」。这个区分很重要 —— 很多人以为 mock 跑通了就等于接上模型也能跑。

**Agent 陷入死循环 / 一直调工具**
ReAct 没有天然终止条件，必须有「最大轮数」兜底。仓库里这份脚本带了轮数上限；自己改的时候别把这个保护删掉。

**`ModuleNotFoundError: No module named 'langgraph'`**
LangGraph 的包名是 `langgraph`，但导入路径和类名跟文档版本有关。先 `pip show langgraph` 看一眼版本，再对照官方文档的版本说明。
