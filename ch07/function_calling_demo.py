"""
【本书配套示例】Function Calling（原生 tool_calls）完整闭环：定义工具 → 模型请求调用 → 本地执行 → 回喂结果
对应章节：ch07 §7.1.4 一个 Function Calling 的完整示例（书稿对应小节，88 行完整脚本）
运行方式：
    export DEEPSEEK_API_KEY=sk-xxx        # 书稿明确要求 key 走环境变量
    python ch07/function_calling_demo.py
    # 换模型/换兼容端点（书稿写 deepseek/deepseek-v4-pro，该模型名见下方说明）：
    export FC_MODEL=deepseek/deepseek-chat
    export FC_BASE_URL=https://api.deepseek.com
依赖：litellm>=1.83（实测 **1.106.6**）。缺 litellm → ImportError: No module named 'litellm'
      （书稿对应小节 提醒过：别装 v1.82.7 / v1.82.8，那两个版本被确认含恶意代码。）
    实测解释器：python（需 onnxruntime + torch 的环境）（litellm 1.106.6）
实测状态：❌ 未跑通 —— **唯一原因是缺 DEEPSEEK_API_KEY**
    实测命令与输出：
        venv-onnx/bin/python ch07/function_calling_demo.py  → 真实退出码 rc=1
        （stderr 还有 2 条 litellm 自己的告警，与本例无关，见下方「已知噪音」）
        ⚠️ 请设置环境变量 DEEPSEEK_API_KEY
           export DEEPSEEK_API_KEY='sk-xxx'
    litellm 本身没问题：litellm 1.106.6 可 import、litellm.completion 可调用；
    外网也可达（curl https://api.deepseek.com/v1/models → 401，域名通、只缺凭证）。
    给出 key 后即可跑通，届时请把这行改成实测结果。
    实测时间：

**已知噪音**：litellm 1.106.6 启动时会去拉
    https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
本机访问 GitHub raw 超时，于是打两条 `LiteLLM:WARNING ... Falling back to local backup`。
它不影响调用（litellm 自带本地价目表兜底），也不是配置问题；不想看就把 litellm 日志级别调高。

与书稿差异（两处，都是为「不写死」）：
  1. 模型名与 base_url 改为环境变量 FC_MODEL / FC_BASE_URL，默认值 `deepseek/deepseek-chat`。
     书稿写的是 `deepseek/deepseek-v4-pro`，并在注释里注明「具体模型名和 API 地址以官方文档为准」
     —— 本书成稿时该模型名未在官方文档核实过（全书 9 处 deepseek-v4 系列），
     故不把未核实的名字设成默认；想照书跑就 `export FC_MODEL=deepseek/deepseek-v4-pro`。
  2. try/except 内层结构保持书稿原样（含 mock 的 get_stock_price）；只把外层函数化，
     方便 import 而不执行。
"""

import json
import os
import sys

import litellm

# DeepSeek-V4-Pro 是 DeepSeek 截止 2026 年中推荐的主力模型
# 具体模型名和 API 地址以官方文档为准：https://api-docs.deepseek.com
model_name = os.environ.get("FC_MODEL", "deepseek/deepseek-chat")
base_url = os.environ.get("FC_BASE_URL", "https://api.deepseek.com")

# 从环境变量中读取 API Key（实践中不要在代码中硬编码）
api_key = os.getenv("DEEPSEEK_API_KEY")


def main() -> int:
    if not api_key:
        print("⚠️ 请设置环境变量 DEEPSEEK_API_KEY")
        print("   export DEEPSEEK_API_KEY='sk-xxx'")
        return 1

    # 1. 定义工具：告诉模型有哪些外部能力可用
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "获取指定股票的实时价格（模拟数据）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "股票代码，如 AAPL、600519",
                        }
                    },
                    "required": ["symbol"],
                },
            },
        }
    ]

    try:
        # 2. 第一轮：把「问题 + 工具清单」交给模型，看它要不要调工具
        messages = [
            {"role": "user", "content": "帮我查一下 AAPL 现在多少钱？"}
        ]
        msg = litellm.completion(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        ).choices[0].message

        print("第 1 轮模型回复:", getattr(msg, "content", None))

        # 3. 模型没有要求调工具，说明它直接答了
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            print("模型未请求工具调用，流程结束。")
            return 0

        # 4. 解析工具调用请求
        call = msg.tool_calls[0].function
        print(f"\n🎯 模型请求调用: {call.name}")
        print(f"   参数: {call.arguments}")

        # 5. 模拟执行"真实"的函数调用
        args = json.loads(call.arguments)

        def get_stock_price(symbol: str) -> str:
            """模拟股价查询（实际应接入行情 API）"""
            # 此处应替换为真正的数据源查询
            mock_data = {"AAPL": 198.75, "GOOGL": 152.30}
            price = mock_data.get(symbol)
            if price is None:
                return json.dumps({"error": f"未找到股票代码 {symbol}"})
            return json.dumps({"symbol": symbol, "price": price})

        tool_result = get_stock_price(args["symbol"])
        print(f"   查询结果: {tool_result}")

        # 6. 第二轮：把工具结果回喂给模型，让它用自然语言收尾
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": msg.tool_calls[0].id,
            "content": tool_result,
        })
        final = litellm.completion(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            tools=tools,
        ).choices[0].message
        print("\n第 2 轮模型回复（带工具结果）:", final.content)
        return 0

    except Exception as e:
        print(f"❌ 调用失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
