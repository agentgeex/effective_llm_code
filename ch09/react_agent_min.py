# -*- coding: utf-8 -*-
"""手搓一个最小可运行智能体 —— 完整 ReAct 循环（第 9 章最长的一段可运行代码）

① 这是什么
   书稿 §9.4.1 手搓的 ReAct 智能体：大模型 + 两个工具（查天气 / 计算器）+ 循环决策。
   模型每轮自主决定「调工具」还是「给最终答案」，工具结果回喂对话历史再问一轮，
   max_turns 熔断防死循环。零框架，只依赖 openai。

② 对应书稿：第 9 章 §9.4.1，书稿对应小节；
   另含 §9.5.3 书稿对应小节 的 MAX_ITERATIONS=5 熔断片段（等价于本文件 max_turns=5）

③ 怎么跑（仓库根目录执行）
   # A) 真调模型（需要 key）
   export DEEPSEEK_API_KEY=<你的key>
   .venv/bin/python ch09/react_agent_min.py
   # B) 不联网、不花钱，只验证「工具循环」逻辑本身（默认即此模式，无需 key）
   .venv/bin/python ch09/react_agent_min.py --mock

④ 依赖
   - 包：openai>=1.x（本机实测 3.3.0）；--mock 模式零依赖（除 openai 的异常类型）
   - 服务：DEEPSEEK_API_KEY（base_url 指向 api.deepseek.com）；无 key 时只能走 --mock

⑤ 实测解释器
   python（Python 3.10.12 / openai 3.3.0）
   实测：--mock rc=0（2 次工具调用 → 最终答案，3 轮，max_turns 未触发）；
         真调模型未跑：缺 DEEPSEEK_API_KEY

⑥ 与书稿的差异（三处，均为「能让它跑起来」的最小改动）
   1. client 从模块级改为可选：书稿对应小节 直接 `client = OpenAI(api_key=os.getenv(...))`，
      openai 3.3.0 在 api_key 为 None 时构造即抛
      `OpenAIError: Missing credentials. Please pass an \`api_key\`, \`workload_identity\`,
       \`admin_api_key\`, or set the \`OPENAI_API_KEY\` ...`
      本文件改为 build_client()：有 key 返回真 client，--mock 返回脚本化的假 client。
      循环体、工具定义、TOOLS、TOOL_MAP、SYSTEM_PROMPT 全部一字未改。
   2. key 名按 base_url 对齐：书稿顶部写 `export OPENAI_API_KEY="your-key"`，
      但下面 base_url="https://api.deepseek.com"、model="deepseek-chat"。
      本文件优先读 DEEPSEEK_API_KEY，其次 OPENAI_API_KEY（两个都认，不悄悄失败）。
   3. max_turns 默认值与书稿一致（5）；末尾「超过最大轮数」提示文案原样保留。

★ 这个文件验证的是什么（说清楚，免得被当成「智能体跑通了」）
   --mock 模式证明的是 **循环与工具协议正确**：模型说要调工具 → 我们执行 → 回喂结果 →
   再问一轮 → 模型说不用工具 → 收尾。它不证明模型选得对，也不证明真 API 的返回结构与
   mock 一致（后者是书稿自己的假设，没有 key 无法核对）。真调模型那一步，等有 key 再补实测。
"""

import json
import os
import sys

from openai import OpenAI

# ── 初始化模型 ───────────────────────────────────────────
BASE_URL = "https://api.deepseek.com"   # 如果用 OpenAI，改为 "https://api.openai.com/v1"
MODEL = "deepseek-chat"


def build_client():
    """书稿在此处直接构造 client；改为可返回 mock，见文件头「与书稿差异 1」。"""
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        return OpenAI(api_key=key, base_url=BASE_URL), "live"
    if "--mock" in sys.argv:
        return _MockClient(), "mock"
    print("⚠️ 未设置 key，若不想真调模型请用 --mock")
    print("   export DEEPSEEK_API_KEY=<your-key>")
    sys.exit(1)


# ── 工具定义 ─────────────────────────────────────────────
def get_weather(city: str) -> str:
    """查询城市天气（模拟）"""
    # 实际项目这里会调用天气 API
    weather_db = {
        "北京": "晴，28°C，湿度 45%",
        "上海": "多云，32°C，湿度 70%",
        "深圳": "雷阵雨，30°C，湿度 85%",
    }
    return weather_db.get(city, f"未找到 {city} 的天气数据")


def calculator(expression: str) -> str:
    """安全的数学计算器"""
    try:
        # 安全评估：仅允许数字和基本运算符
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)  # noqa: S307 —— 已用 allowed 白名单过滤字符
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# 工具描述（告诉模型什么时候调用）—— 与书稿对应小节 一致
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。当用户问包含数字运算的问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 3*15+8"}
                },
                "required": ["expression"],
            },
        },
    },
]

# 工具名称到函数的映射
TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
}

# ── 核心循环 ─────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个有用的助手。你可以使用工具来帮助回答问题。
- 当需要天气信息时，调用 get_weather 工具
- 当需要计算时，调用 calculator 工具
- 如果不需要工具，直接回答用户问题"""


def run_agent(user_input: str, max_turns: int = 5, client=None, tag: str = "live") -> str:
    """
    智能体核心循环
    max_turns: 最大工具调用轮数，防止无限循环消耗成本
    """
    if client is None:
        client, tag = build_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    tool_calls_made = 0

    for turn in range(max_turns):
        print(f"\n── 第 {turn + 1} 轮 ──")

        # 调用模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,  # 工具调用追求确定性
        )

        msg = response.choices[0].message

        # 模型决定不再调用工具 → 直接返回最终答案
        if not msg.tool_calls:
            print(f"[{tag}] 第 {turn + 1} 轮 → 模型给最终答案")
            print(f"[模型最终回复]: {msg.content}")
            print(f"统计：工具调用 {tool_calls_made} 次，轮次 {turn + 1}，"
                  f"{'max_turns 未触发' if turn + 1 < max_turns else 'max_turns 已用尽'}")
            return msg.content

        # 模型决定调用工具 → 执行工具并反馈结果
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"[{tag}] 第 {turn + 1} 轮 → 模型请求工具: {tool_name}({tool_args})")

        # 实际执行工具
        tool_func = TOOL_MAP[tool_name]
        result = tool_func(**tool_args)
        tool_calls_made += 1

        print(f"[工具返回]: {result}")

        # 将工具调用和结果追加到对话历史
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # 超过最大轮数
    return "（智能体超过最大工具调用轮数，已停止。建议简化问题或增加 max_turns。）"


# ── mock 客户端（仅为让循环可离线验证，不参与教学） ────────────────
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _MockClient:
    """按脚本依次返回：调 get_weather → 调 calculator → 给最终答案。"""

    def __init__(self):
        self._steps = [
            ("get_weather", {"city": "北京"}),
            ("calculator", {"expression": "(28*9/5)+32"}),
            ("FINAL", "北京今天晴，28°C（约 82.4°F），湿度 45%，适合户外运动。"),
        ]
        self._i = 0
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kwargs):
        kind, payload = self._steps[self._i]
        self._i += 1
        if kind == "FINAL":
            msg = _Obj(content=payload, tool_calls=None)
        else:
            msg = _Obj(
                content=None,
                tool_calls=[_Obj(
                    id=f"call_{self._i}",
                    function=_Obj(name=kind, arguments=json.dumps(payload, ensure_ascii=False)),
                )],
            )
        return _Obj(choices=[_Obj(message=msg)])


# ── 测试 ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("测试: 北京今天天气怎么样，体感温度相当于多少华氏度？")
    print("=" * 60)
    cli, mode = build_client()
    answer = run_agent("北京今天天气怎么样，体感温度相当于多少华氏度？",
                       client=cli, tag=mode)
    print(f"\n✅ 最终答案: {answer}")

    if mode == "mock":
        print("\n（--mock：只验证工具循环逻辑，未真调模型。去掉 --mock 并设 key 才是真跑。）")
