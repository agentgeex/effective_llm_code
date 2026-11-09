# -*- coding: utf-8 -*-
"""ch02 §2.2.4 在线推理服务：零部署直接调用（书稿对应小节）

书稿给的是两段直连调用（OpenRouter / DeepSeek 官方）。本文件是同一件事的另一种写法：
把客户端封成类，再演示 function calling（tools + tool_calls 两轮往返）。

    export OPENAI_API_KEY=sk-xxxx
    .venv/bin/python ch02/openai_test.py

默认打 OpenAI 官方端点 `https://api.openai.com/v1`，默认模型 `gpt-4o-mini`。

★ 只要能说 OpenAI 协议的服务都能用，改两个环境变量即可：
      OPENAI_BASE_URL   端点地址
      OPENAI_MODEL      模型名（每个服务的模型名都不一样，必须跟着换）

    官方（默认）        https://api.openai.com/v1        gpt-4o-mini
    DeepSeek 官方       https://api.deepseek.com/v1      deepseek-chat
    本地 vLLM           http://127.0.0.1:8000/v1          <起服务时 --served-model-name 的值>
    第三方中转          https://www.chataiapi.com/v1      gemini-2.5-flash

    两个变量必须成对设：官方端点不认 `gemini-2.5-flash` 这类网关模型名，
    反过来网关也不认 `gpt-4o-mini`。配错会直接 404 `model_not_found`。

依赖：openai（实测 3.3.0）。缺包 → ImportError。
    实测解释器：.venv/bin/python

实测状态：缺 `OPENAI_API_KEY` 时按设计退出（rc=2），不发任何请求。
    真正调通需要你自己的 key —— 本仓库不带任何密钥，也没有替读者实测过付费端点。
"""
import os
from openai import OpenAI
from openai import AuthenticationError
from datetime import datetime, timedelta
import json


class Openai_api_demo():

    def __init__(self):
        self.openai_api_client = OpenAI(
            # （原注释里的备用 key 已清除）
            api_key=os.environ.get("OPENAI_API_KEY", ""),  # 密钥改为环境变量，见 .env.example
            # base_url 可用 OPENAI_BASE_URL 覆盖，任何兼容 OpenAI 协议的服务都能接：
            #   https://api.deepseek.com/v1         （DeepSeek 官方）
            #   http://127.0.0.1:8000/v1            （本地 vLLM / Ollama 的兼容层）
            #   https://www.chataiapi.com/v1        （第三方中转，模型名要跟着换）
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

    if not os.environ.get("OPENAI_API_KEY"):
        print("缺少环境变量 OPENAI_API_KEY（书稿 §2.2.4 的 api_key 占位符要换成真实 key）")
        raise SystemExit(2)

    clientapi = Openai_api_demo()
    # clientapi.question_test()
    clientapi.query_function_call_test()
