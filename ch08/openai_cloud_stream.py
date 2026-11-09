# -*- coding: utf-8 -*-
"""① 这是什么
    ch08 §8.3.1 路径A 的可运行版：用 OpenAI SDK 流式调用云端推理服务，
    逐 delta 打印，收完打印总字符数。

② 对应书稿
    第 8 章 §8.3.1「路径 A：直接调用第三方模型服务」。

③ 怎么跑
    export OPENAI_API_KEY=<你的key>
    .venv/bin/python ch08/openai_cloud_stream.py
    # 换 OpenAI 兼容端点（可选）：
    #   export CLOUD_BASE_URL=https://your-endpoint/v1
    #   export CLOUD_MODEL=<模型名>

④ 依赖
    包：openai>=1.x（本机实测 3.3.0）
    服务：可访问的 OpenAI / 兼容端点 + 有效 API Key
    权重：无（云端推理）
    网络：本机实测不可达，见 ⑤

⑤ 实测解释器 + 结果
    解释器：python（Python 3.10.12 / openai 3.3.0）
    - 不设 key：rc=2
        ⚠️ 请设置环境变量 OPENAI_API_KEY
           export OPENAI_API_KEY=<你的key>
    - 设一个假 key：rc=1
        openai.APIConnectionError: Connection error.
        底层：httpx2.ConnectError: [Errno 101] Network is unreachable
    - 结论：客户端构造与请求组装都走到了发请求这一步；「真实云端返回流式内容」未验证
      （有效 key 与本机出网两样都缺，不是代码问题）。

⑥ 与书稿的差异（一处）
    书稿对应小节 写死 `OpenAI(api_key="your-api-key")`，而它上面两行注释自己写着
    「生产环境中请通过环境变量管理 API Key，不要硬编码」。本文件按它自己的提醒实现：
    读 OPENAI_API_KEY，缺就退出（rc=2），不静默降级、不假装跑通。
    另：书稿 `model="gpt-4o"`，这里可用 CLOUD_MODEL 覆盖（便于换兼容端点）。
    请求内容（system/user 文案、stream=True、逐 chunk 打印）与书稿一致。
"""


import os
import sys

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.environ.get("CLOUD_BASE_URL")  # 不设则用 SDK 默认 https://api.openai.com/v1
model = os.environ.get("CLOUD_MODEL", "gpt-4o")


def main() -> int:
    if not api_key:
        print("⚠️ 请设置环境变量 OPENAI_API_KEY")
        print("   export OPENAI_API_KEY=<你的key>")
        return 2

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的架构顾问。"},
            {"role": "user", "content": "帮我分析一下微服务和单体架构的选型标准。"},
        ],
        stream=True,  # 启用流式输出
    )

    # 流式消费推理结果
    chars = 0
    for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
            chars += len(delta)
    print(f"\n\n[流式收完，共 {chars} 字符]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
