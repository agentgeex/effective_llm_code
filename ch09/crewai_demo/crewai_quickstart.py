# -*- coding: utf-8 -*-
"""CrewAI 多 Agent 协作最小示例（双 Agent：数据分析师 + 报告撰写人）

① 这是什么
   书稿 §9.3.1 的 CrewAI 示例：两个 Agent(role/goal/backstory) + 两个 Task + Crew 顺序协作，
   最后 kickoff() 执行。演示「组团队」而非「单智能体」。

② 对应书稿：第 9 章 §9.3.1，书稿对应小节（代码块 `from crewai import Agent, Task, Crew`）

③ 怎么跑（仓库根目录执行）
   # 无 key：只打印 crew 结构（验证代码组织正确，不构造对象、不调用模型）
   .venv/bin/python ch09/crewai_demo/crewai_quickstart.py --dry-run
   # 有 key：真正跑（默认 OpenAI，需 OPENAI_API_KEY）
   export OPENAI_API_KEY=<你的key>
   .venv/bin/python ch09/crewai_demo/crewai_quickstart.py

④ 依赖
   - 包：crewai（本机实测 1.6.1）
   - 服务：OPENAI_API_KEY（或通过 llm= 指定其他模型）；无 key 走 --dry-run

⑤ 实测解释器
   python（Python 3.10.12 / crewai 1.6.1）
   实测：--dry-run rc=0；无 key 直跑 rc=2（按设计给出提示）；真 kickoff 未跑（缺 OPENAI_API_KEY）

⑥ 与书稿的差异
   (1) 书稿对应小节 `Task(description="...", agent=...)` 缺 expected_output。
       crewai 1.x 起 expected_output 是必填 → pydantic ValidationError: expected_output — Field required。
       本文件为每个 Task 补 expected_output="..."。
   (2) key 处理：书稿只在环境说明里提「需设置 OPENAI_API_KEY」，代码里没有检查。
       实测 crewai 1.6.1 在 Agent() 构造阶段就 create_llm，无 key 报
       ImportError: Error importing native provider: OPENAI_API_KEY is required。
       本文件改为 --dry-run 打印结构（不构造对象）；无 key 直跑给出提示并 rc=2。
   (3) 其余保持书稿写法：Agent(role/goal/backstory) ×2、Crew(agents, tasks)、kickoff()。

定级：B（代码完整；真跑需要付费 API key）
"""

import argparse
import os
import sys

from crewai import Agent, Task, Crew

MODEL_HINT = "默认使用 OpenAI（需 OPENAI_API_KEY）；也可用 llm= 指定其他模型"


def build_crew():
    researcher = Agent(
        role="数据分析师",
        goal="从用户数据中提取关键洞察",
        backstory="你是一位经验丰富的数据分析师，擅长从原始数据中发现模式。",
    )

    writer = Agent(
        role="报告撰写人",
        goal="将分析结果转化为清晰的商业报告",
        backstory="你擅长用简洁的语言解释复杂的数据结论。",
    )

    # 书稿少了 expected_output（crewai 1.15.x 必填；本仓库解析到 1.6.1，补上也不影响）
    task1 = Task(
        description="分析上周的用户增长数据",
        expected_output="一份要点式的数据洞察清单（3-5 条）",
        agent=researcher,
    )
    task2 = Task(
        description="根据分析结果撰写周报",
        expected_output="一段 300 字以内的中文周报",
        agent=writer,
    )

    return Crew(agents=[researcher, writer], tasks=[task1, task2])


# 书稿里的两个 Agent / 两个 Task，抽成常量，便于 --dry-run 展示结构
AGENT_SPECS = [
    ("数据分析师", "从用户数据中提取关键洞察", "你是一位经验丰富的数据分析师，擅长从原始数据中发现模式。"),
    ("报告撰写人", "将分析结果转化为清晰的商业报告", "你擅长用简洁的语言解释复杂的数据结论。"),
]
TASK_SPECS = [
    ("分析上周的用户增长数据", "一份要点式的数据洞察清单（3-5 条）"),
    ("根据分析结果撰写周报", "一段 300 字以内的中文周报"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印结构，不构造对象、不调用模型")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] 将要构造的 crew 结构：")
        for role, goal, backstory in AGENT_SPECS:
            print(f"  Agent(role={role!r}, goal={goal!r}, backstory={backstory[:12]}...)")
        for desc, expected in TASK_SPECS:
            print(f"  Task(description={desc!r}, expected_output={expected!r})")
        print(f"[模型] {MODEL_HINT}")
        print("\n[dry-run] 未构造 crewai 对象：crewai 1.6.1 在 Agent() 构造阶段就要 create_llm，"
              "\n          无 key 会在 __init__ 里直接 ImportError（OPENAI_API_KEY is required）。")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[跳过] 未设置 OPENAI_API_KEY（crewai 在 Agent() 构造阶段即需要它）。"
              "\n      先 export OPENAI_API_KEY=... 再跑，或加 --dry-run 只看结构。", file=sys.stderr)
        sys.exit(2)

    crew = build_crew()
    print(f"[crew] agents={[a.role for a in crew.agents]}")
    print(f"[crew] tasks={[t.description for t in crew.tasks]}")
    print(f"[模型] {MODEL_HINT}")
    print("\n[kickoff] 开始执行 ...")
    result = crew.kickoff()
    print("\n[结果]\n", result)


if __name__ == "__main__":
    main()
