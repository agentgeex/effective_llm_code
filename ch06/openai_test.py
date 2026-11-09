# -*- coding: utf-8 -*-
"""ch06 §6.2.6 —— OpenAI SDK 两种最小用法（书稿对应小节）

① 这是什么：OpenAI Python SDK 的两个最小可用场景——
   `question_test()`：chat.completions.create 普通对话；
   `query_function_call_test()`：tools + tool_calls 两轮往返（本地执行函数 → role="tool" 回填 → 第二轮）。
   类封装版：`Openai_api_demo` 持有一个客户端。

② 对应书稿：第 6 章 §6.2.6
   - 书稿：messages 数组 + chat.completions.create（→ `question_test`）
   - 书稿：tools 定义与 tool_calls 取用（→ `query_function_call_test`）
   同节 的流式（stream=True）片段本文件**未实现**，见 README §4.2。

③ 怎么跑：
       export OPENAI_API_KEY=<你的 key>
       # 可选：换成 OpenAI 官方或本地 vLLM/Ollama 的兼容端点
       export OPENAI_BASE_URL=https://api.openai.com/v1
       .venv/bin/python ch06/openai_test.py
   没有 key 时脚本直接退出（rc=2），不会发请求。

④ 依赖：openai（本机 `.venv` 实测 3.3.0）；需要有效 API key + 可访问的 OpenAI 兼容端点。
   默认打进 OpenAI 官方端点 `https://api.openai.com/v1`，默认模型 `gpt-4o-mini`。

⑤ 实测解释器：python（openai 3.3.0 / Python 3.10.12），见文件末尾「实测记录」。

⑥ 与书稿的差异：
   1. **密钥出码**：原文件硬编码 key（已清除），改为 `OPENAI_API_KEY` 环境变量；
      书稿本身写 `OpenAI()`（隐式读环境变量），这一点反而与书稿一致了。
   2. `base_url` 默认就是官方 `https://api.openai.com/v1`，与书稿一致；改 `OPENAI_BASE_URL`
      可指向任何兼容 OpenAI 协议的服务（书稿后文也提到能换成 vLLM / Ollama）。
   3. 默认模型 `gpt-4o-mini`（书稿示例用 `gpt-4o`，换成 `OPENAI_MODEL=gpt-4o` 就是书稿那一版）；
      接别的服务时模型名必须跟着换。
   4. 工具函数名保留原仓库的 `get_current_weather`；书稿是 `get_weather`，语义相同。
   5. 请求体、两轮往返结构、`json.loads(tool_call.function.arguments)` 与书稿一致。
"""
import os
import sys
from openai import OpenAI
from openai import AuthenticationError
from datetime import datetime, timedelta
import json


class Openai_api_demo():

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("缺少环境变量 OPENAI_API_KEY：请先 export OPENAI_API_KEY=<你的 key>", file=sys.stderr)
            print("（本脚本不硬编码密钥；没有 key 时不发请求，直接退出）", file=sys.stderr)
            raise SystemExit(2)
        self.openai_api_client = OpenAI(
            api_key=api_key,
            # 任何兼容 OpenAI 协议的服务都能接（换服务时 OPENAI_MODEL 要一起换）：
            #   https://api.openai.com/v1        （默认，官方）
            #   https://api.deepseek.com/v1      （DeepSeek 官方，模型名 deepseek-chat）
            #   http://127.0.0.1:8000/v1         （本地 vLLM / Ollama 兼容层）
            #   https://www.chataiapi.com/v1     （第三方中转，模型名如 gemini-2.5-flash）
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def question_test(self):
        """简单的测试
        """
        try:

            client = self.openai_api_client

            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),  # 换服务时这里也要跟着换
                messages=[
                    {"role": "system", "content": "你是资深的大模型技术顾问，熟悉 Transformer 架构、训练与推理的各种细节，回答要具体、可操作。请用中文回答"},
                    {
                        "role": "user",
                        "content": 'Transformer 里的"自注意力机制"到底在算什么？Q、K、V 三个矩阵各自的作用是什么？为什么要除以 √d_k？返回内容请采用 markdown 的格式'
                    }
                ]
            )

            anawer = response.choices[0].message.content

            anawer = anawer.replace("\n\n---\n\n", "\n\n")

            print(anawer)
        except AuthenticationError as e:
            error_str = str(e)
            print(f"错误信息：{error_str}")
            # print(f"{e.__traceback__}")

        except Exception as e:
            # 记录错误日志
            print(f"错误：{e}")
            print(f"{e.__traceback__}")
            # logger.error(e, exc_info=True)

    def query_function_call_test(self):

        client = self.openai_api_client
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "获取指定位置的当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称，例如：北京, 上海",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "请问上海今天的天气怎么样？"}]

        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),  # 换服务时这里也要跟着换
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 允许模型自动选择是否使用工具
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print(f"第一次提交模型的反馈  {response_message}")

        if tool_calls:
            available_functions = {
                "get_current_weather": self.get_current_weather,
            }
            messages.append(response_message)  # 将模型响应添加到对话记录中

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)

                # 步骤 3: 实际调用函数
                function_response = function_to_call(
                    location=function_args.get("location"),
                )

                # 步骤 4: 将函数结果作为新的消息发送给模型
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        # 不放 name：官方 spec 的 tool 消息只要 tool_call_id + content。
                        # name 是旧的 function 消息字段，兼容网关一般会忽略，但没必要带。
                        # 该写法按官方 spec 判断为非必需字段，未用官方 key 实测过。
                        "content": function_response,
                    }
                )

            second_response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),  # 换服务时这里也要跟着换
                messages=messages,
            )
            secnod_result = second_response.choices[0].message.content
            print(f"第二次模型反馈: {secnod_result}")

    def get_current_weather(self, location):
        """
        获取指定位置的当前天气。

        参数：
        location (str): 城市名称
        """
        if "北京" in location:
            return json.dumps({"location": "北京", "temperature": "22", "unit": "摄氏度", "forecast": ["晴朗", "风力微弱"]})
        elif "上海" in location:
            return json.dumps({"location": "上海", "temperature": "28", "unit": "摄氏度", "forecast": ["多云", "有阵雨"]})
        else:
            return json.dumps({"location": location, "temperature": "未知", "unit": "未知", "forecast": ["未知"]})


if __name__ == '__main__':

    clientapi = Openai_api_demo()
    # clientapi.question_test()
    clientapi.query_function_call_test()


# ─────────────────────────────────────────────────────────
# 实测记录
#   解释器：python（openai 3.3.0 / Python 3.10.12）
#
#   1) 无 key：
#        $ python ch06/openai_test.py
#        缺少环境变量 OPENAI_API_KEY：请先 export OPENAI_API_KEY=<你的 key>
#        RC=2
#   2) 假 key + 第三方中转端点（https://www.chataiapi.com/v1）：
#        openai.APITimeoutError: Request timed out.   # httpx2.ConnectTimeout，约 90s，RC=1
#   3) 假 key + 官方端点（OPENAI_BASE_URL=https://api.openai.com/v1）：
#        openai.APIConnectionError: Connection error.  # RC=1
#
#   结论：本机到这两个端点都不通，**无法验证 401/200**；不写成「跑通」。
#   代码层面的 messages / tools / tool_calls 结构与书稿一致，未改动。
# ─────────────────────────────────────────────────────────
