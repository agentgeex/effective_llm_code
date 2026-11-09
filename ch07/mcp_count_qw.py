"""
【本书配套示例】LLM + MCP 自动工具调用循环：模型返回 {tool, arguments}，客户端执行后把结果回喂模型
① 这是什么：让模型自己决定调哪个 MCP 工具 —— 不是 OpenAI 原生 function calling，而是靠系统提示
   要求模型只回纯 JSON {tool, arguments}，客户端解析后 call_tool，再把结果塞回对话。
② 对应书稿：ch07 §7.3.5 MCP部署与客户端调用（书稿对应小节 起）；客户端骨架另可对照 ch11 §11.5.3（书稿对应小节 起）。
③ 怎么跑：
   # 1) 先起 MCP Server（本文件连 127.0.0.1:8011，对应 ch07/mcp_server.py）
   .venv/bin/python ch07/mcp_server.py --sse
   # 2) 再跑本脚本（需要 DeepSeek Key）
   export DEEPSEEK_ARK_API_KEY=<你的 DeepSeek API Key>
   .venv/bin/python ch07/mcp_count_qw.py
   # 交互式：输入问题回车；输入 quit / exit / 退出 结束
④ 依赖：fastmcp（实测 4.1.4）、openai（实测 3.40.0）、本地 8011 上的 MCP Server、DEEPSEEK_ARK_API_KEY。
⑤ 实测解释器与结果：python， 复测
   - 无 DEEPSEEK_ARK_API_KEY → rc=1，openai.OpenAIError: Missing credentials. ...
   - 假 key（sk-fake）+ 服务端已起：能连上 SSE 并 list_tools，走到交互提示后
       有 stdin：printf '退出\n' | ... → rc=0（打印「用户: AI助手退出」）
       无 stdin：EOFError: EOF when reading a line（交互式程序的正常反应，不是 bug）
     → MCP 一半通；LLM 调用未验证（我没有可用的 DeepSeek Key，也不该拿别人的）。
   - 交互式程序不适合放进批处理/CI 里跑。
⑥ 与书稿的差异：
   1. 环境变量名沿用作者的 DEEPSEEK_ARK_API_KEY（不是 DEEPSEEK_API_KEY），与 .env.example 一致。
   2. 其余逻辑与书稿一致：靠「让模型返回 JSON {tool, arguments}」触发工具调用，非原生 function calling
      —— 这是本示例的教学点，未改动。
"""

import json
from fastmcp import Client
import asyncio
from openai import OpenAI
import os


deepseek_appkey = os.environ.get("DEEPSEEK_ARK_API_KEY")
base_url = "https://api.deepseek.com/v1"  # DeepSeek API 地址
api_key = deepseek_appkey         # 替换为你的 DeepSeek API Key
model = "deepseek-chat"                    # DeepSeek 模型名称



class LLMClient:
    """LLM客户端，负责与大语言模型API通信"""

    def __init__(self, model_name: str, url: str, api_key: str) -> None:
        self.model_name: str = model_name
        self.url: str = url
        self.client = OpenAI(api_key=api_key, base_url=url)

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """发送消息给LLM并获取响应"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content


class ChatSession:
    """聊天会话，处理用户输入和LLM响应，并与MCP工具交互"""

    def __init__(self, llm_client: LLMClient, mcp_client: Client, ) -> None:
        self.mcp_client: Client = mcp_client
        self.llm_client: LLMClient = llm_client

    async def process_llm_response(self, llm_response: str) -> str:
        """处理LLM响应，解析工具调用并执行"""
        try:
            # 尝试移除可能的markdown格式
            if llm_response.startswith('```json'):
                llm_response = llm_response.strip('```json').strip('```').strip()
            tool_call = json.loads(llm_response)
            if"tool"in tool_call and"arguments"in tool_call:
                # 检查工具是否可用
                tools = await self.mcp_client.list_tools()
                if any(tool.name == tool_call["tool"] for tool in tools):
                    # 执行工具调用
                    result = await self.mcp_client.call_tool(
                        tool_call["tool"], tool_call["arguments"]
                    )

                    return f"Tool execution result: {result}"

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            # 如果不是JSON格式，直接返回原始响应
            return llm_response
        except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        print(error_msg)
                        return error_msg

    async def start(self, system_message) -> None:
        """启动聊天会话的主循环"""
        messages = [{"role": "system", "content": system_message}]
        while True:
            try:
                # 获取用户输入
                user_input = input("用户: ").strip().lower()
                if user_input in ["quit", "exit", "退出"]:
                    print('AI助手退出')
                    break
                messages.append({"role": "user", "content": user_input})

                # 获取LLM的初始响应
                print("###################")
                print(f"助手提交GPT{json.dumps(messages,indent=6,ensure_ascii=False)}")
                print("###################")
                llm_response = self.llm_client.get_response(messages)
                print("助手: ", llm_response)

                # 处理可能的工具调用
                result = await self.process_llm_response(llm_response)

                # 如果处理结果与原始响应不同，说明执行了工具调用，需要进一步处理
                while result != llm_response:
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content": f"MCP回复: {result}"})

                    # 将工具执行结果发送回LLM获取新响应

                    print(f"助手提交GPT{json.dumps(messages,indent=6,ensure_ascii=False)}")
                    llm_response = self.llm_client.get_response(messages)
                    result = await self.process_llm_response(llm_response)
                    print("助手: ", llm_response)

                messages.append({"role": "assistant", "content": llm_response})

            except KeyboardInterrupt:
                print('AI助手退出')
                break


async def main():

    async with Client("http://127.0.0.1:8011/sse") as mcp_client:
        # 初始化LLM客户端，使用通义千问模型
        llm_client = LLMClient(model_name=model, 
                               api_key=api_key,
                               url=base_url)

        # 获取可用工具列表并格式化为系统提示的一部分
        tools = await mcp_client.list_tools()
        dict_list = [tool.__dict__ for tool in tools]
        tools_description = json.dumps(dict_list, ensure_ascii=False)

        # 系统提示，指导LLM如何使用工具和返回响应
        system_message = f'''
                你是一个智能助手，严格遵循以下协议返回响应：

                可用工具：{tools_description}

                响应规则：
                1、当需要计算时，返回严格符合以下格式的纯净JSON：
                {{
                    "tool": "tool-name",
                    "arguments": {{
                        "argument-name": "value"
                    }}
                }}
                2、禁止包含以下内容：
                - Markdown标记（如```json）
                - 自然语言解释（如"结果："）
                - 格式化数值（必须保持原始精度）
                - 单位符号（如元、kg）

                校验流程：
                ✓ 参数数量与工具定义一致
                ✓ 数值类型为number
                ✓ JSON格式有效性检查

                正确示例：
                用户：单价88.5买235个多少钱？
                响应：{{"tool":"multiply","arguments":{{"a":88.5,"b":235}}}}

                错误示例：
                用户：总金额是多少？
                错误响应：总价500元 → 含自然语言
                错误响应：```json{{...}}``` → 含Markdown

                3、在收到工具的响应后：
                 - 将原始数据转化为自然、对话式的回应
                 - 保持回复简洁但信息丰富
                 - 聚焦于最相关的信息
                 - 使用用户问题中的适当上下文
                 - 避免简单重复使用原始数据
                '''
        # 启动聊天会话
        chat_session = ChatSession(llm_client=llm_client, mcp_client=mcp_client)
        await chat_session.start(system_message=system_message)

if __name__ == "__main__":
    asyncio.run(main())