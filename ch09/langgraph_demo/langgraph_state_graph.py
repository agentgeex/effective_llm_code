# -*- coding: utf-8 -*-
"""LangGraph 状态图最小示例（无 LLM、无外部服务，纯本地跑）

① 这是什么
   书稿 §9.3.1 的 LangGraph StateGraph 示例：节点(think/act) + 条件边(should_continue)
   + 累加状态(Annotated[list, operator.add]) 组成的最小有状态图。

② 对应书稿：第 9 章 §9.3.1，书稿对应小节（代码块 `from langgraph.graph import StateGraph, END`）

③ 怎么跑（仓库根目录执行）
   .venv/bin/python ch09/langgraph_demo/langgraph_state_graph.py

④ 依赖
   - 包：langgraph（本机实测可 import，版本号未打印）、langchain-core（langgraph 的传递依赖）
   - 无外部服务、无 API key、无权重、无 GPU

⑤ 实测解释器
   python（Python 3.10.12）
   实测：rc=0，think→act 循环 2 轮，最终消息 6 条

⑥ 与书稿的差异
   (1) 书稿块末是 `app = graph.compile()`，**到这就结束了** → 编译成功但脚本零输出（能编译 ≠ 能跑）。
       本文件把建图收进 build_graph()，末尾补 `app.invoke({"messages": [], "next_action": ""})`
       并打印最终状态 —— 这 3 行是书稿缺的「真正跑起来」那一步。
   (2) 其余原样保留：StateGraph / add_node / set_entry_point / add_edge /
       add_conditional_edges / END / Annotated[list, operator.add] 累加语义，一行未改。
   (3) 只补了 output 打印，没加 LLM、没加 checkpointer（书里本来就没有）。

★ 书稿强调的点在代码里体现为：compile() 只做图校验，不驱动执行；必须 invoke() 才会真正走节点。
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str  # 记录当前决策结果："act" 或 "end"


def think(state: AgentState) -> AgentState:
    """Agent 的思考节点：决定下一步做什么"""
    # 在实际应用中，这里会调用 LLM 分析当前状态
    last_msg = state["messages"][-1] if state["messages"] else "开始思考"
    print(f"[思考] 当前上下文: {last_msg}")
    return {"next_action": "act", "messages": state["messages"] + ["已分析完毕"]}


def act(state: AgentState) -> AgentState:
    """Agent 的执行节点：调用工具"""
    print(f"[执行] 工具调用中...")
    return {"next_action": state.get("next_action", ""), "messages": ["工具执行完成"]}


def should_continue(state: AgentState) -> str:
    """条件边：判断是继续思考还是结束"""
    if len(state["messages"]) > 5:
        return "end"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.set_entry_point("think")
    graph.add_edge("think", "act")
    graph.add_conditional_edges("act", should_continue, {
        "continue": "think",
        "end": END
    })
    return graph.compile()


def main():
    app = build_graph()          # 书稿停在这一步：编译成功，但什么都没发生
    print("[main] compile() 完成（这一步不驱动执行）\n")

    # ↓↓↓ 书稿缺的 3 行：真正跑起来
    final_state = app.invoke({"messages": [], "next_action": ""})
    # ↑↑↑

    print(f"\n[main] 最终消息条数: {len(final_state['messages'])}")
    print(f"[main] 最终 next_action: {final_state['next_action']}")


if __name__ == "__main__":
    main()
