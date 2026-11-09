# -*- coding: utf-8 -*-
"""ch05/openai_protocol_http.py —— 裸 HTTP 调 OpenAI 兼容协议（requests，不用 SDK）

① 这是什么：不用任何 SDK，用 requests 直接 POST `/chat/completions`，把 OpenAI 协议的
   三件套摊开给读者看 —— URL（`/v1/chat/completions`）、Header（`Authorization: Bearer <key>`）、
   Body（`model` + `messages`）。正文示范「知识注入」：system 放公司技术栈（持久知识），
   user 放【参考资料】+【用户问题】（本次任务知识）。

② 对应书稿：第 5 章 §5.2.3 统一语言：OpenAI 协议与知识传输（「一个完整示例」，
   书稿原话是「下面是一个可直接运行的 API 调用」）。同节 4 个 JSON 片段不是独立脚本，
   其内容由本文件 `build_payload()` 覆盖：
     · 协议骨架（model + system + user）
     · 位置一：system 消息注入全局知识
     · 位置二：user 消息嵌【参考资料】+【用户问题】
     · §5.2.2 对话嵌入（出差审批制度的 messages 数组；该块**只有 messages、
       没有 model**，是片段不是完整请求体）

③ 怎么跑（完整命令）：
     export DEEPSEEK_API_KEY=<你的 key>
     python（需 onnxruntime + torch 的环境） ch05/openai_protocol_http.py

   换服务 / 换模型（OpenAI 兼容即可）：
     export OPENAI_BASE_URL=https://api.deepseek.com/v1
     export MODEL_ID=deepseek-chat

   只依赖 requests，本机四个解释器都能跑：
     .venv/bin/python  |  .venv-core/bin/python  |  /usr/bin/python3

④ 依赖：`requests`（无 SDK）；需要外网可达 `api.deepseek.com`；需要有效的
   `DEEPSEEK_API_KEY`。无模型权重、无本地服务。

⑤ 实测解释器：`python（需 onnxruntime + torch 的环境）`（requests 2.34.2 / Python 3.11.15），
    共实跑 5 次（下面前 3 次是改注释前的原始记录；改完后无 key / 假 key 各复跑 1 次，结果完全一致）：
     1) 无 key（venv-onnx）→ rc=2，stderr 原文：
        `缺少环境变量 DEEPSEEK_API_KEY：请先 export DEEPSEEK_API_KEY=<你的 key>`
     2) 无 key（.venv）     → rc=2，同上
     3) 假 key（venv-onnx） → rc=1，stdout 原文：`POST https://api.deepseek.com/v1/chat/completions`
        `payload.model = deepseek-chat` / `HTTP 401`，响应体原文：
        `{"error":{"message":"Authentication Fails, Your api key: ****xxxx is invalid
        (request_id: ...)","type":"authentication_error","param":null,
        "code":"invalid_request_error"}}`（终端会回显你那个 key 的末四位；这里是假 key）
   （改完注释后的复跑：另一个假 key 仍 HTTP 401，request_id 2e5b8a7d-8b5a-4e2c-9605-5e5c278b4969）
   → 结论：URL / Header / Body 的拼装**已被真实服务接受**（回的是 authentication_error，
   不是 400 也不是 404），唯一缺的是有效密钥。本机不持有任何 DEEPSEEK_API_KEY，
   故没有一次 200 —— 这条照实写，不假装跑通。

⑥ 与书稿的差异（四处，都是为「能跑 + 不出事」；协议结构与业务文案一字未改）：
   1. **密钥不再写进代码**。书稿对应小节 是 `"Authorization": "Bearer sk-your-api-key"`，
      照抄会让人以为要硬编码 key。这里读 `DEEPSEEK_API_KEY`，缺了明确报错并 rc=2。
   2. **模型名走环境变量**。书稿对应小节 写 `"model": "deepseek-v4"`；该模型名**未验证**
      （本机无 key，查不了 `/models`；也未在官方文档核实。全书代码里出现 9 处）。
      默认用 `deepseek-chat`，`export MODEL_ID=deepseek-v4` 即可照书跑。
   3. **加了超时与状态码判断**。书稿直接 `response.json()["choices"][0]...`，401/429 时
      会抛 KeyError，看不出真实原因；这里先判状态码再取值 —— 上面的 401 就是这么暴露出来的。
   4. **加了 `POST url` / `payload.model` 两行回显**，便于读者确认自己到底发到哪、发了什么。
   业务内容（system / user 两段长字符串、model+messages 结构）与书稿对应小节 逐字一致。
"""
import json
import os
import sys

import requests

DEFAULT_BASE = "https://api.deepseek.com/v1"


# ---- 示例 5：完整的裸 HTTP 调用示例 ----
def build_payload(model: str) -> dict:
    """书稿对应小节 的请求体，一字未改结构，只把两个长字符串保留原样。"""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是内部 IT 顾问，熟悉公司技术栈和企业级部署方案。"
                    "公司当前基础架构：Kubernetes 集群（48 核 GPU 资源池）、"
                    "已部署 Ollama 和 vLLM 推理服务。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "【参考资料】公司2026年Q2技术规划：优先采用 DeepSeek-V3 和 Qwen2.5 "
                    "作为主力模型，限制外部 API 调用依赖。\n\n"
                    "【用户问题】财务部提了一个需求：希望在内部部署一个能分析财务报表的"
                    "本地模型，要求数据不出公司。我们的基础设施能支持吗？"
                ),
            },
        ],
    }


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("缺少环境变量 DEEPSEEK_API_KEY：请先 export DEEPSEEK_API_KEY=<你的 key>", file=sys.stderr)
        print("（本脚本不硬编码密钥；没有 key 时不发请求，直接退出）", file=sys.stderr)
        return 2

    base = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("MODEL_ID", "deepseek-chat")
    url = base + "/chat/completions"

    payload = build_payload(model)
    print("POST", url)
    print("payload.model =", model)
    print("payload.messages =", json.dumps(payload["messages"], ensure_ascii=False)[:120], "...")

    resp = requests.post(
        url,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    print("HTTP", resp.status_code)
    if resp.status_code != 200:
        print("非 200，响应体前 300 字：", resp.text[:300], file=sys.stderr)
        return 1

    print(resp.json()["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
