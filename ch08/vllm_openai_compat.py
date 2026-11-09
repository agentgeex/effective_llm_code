# -*- coding: utf-8 -*-
"""① 这是什么
    ch08 §8.3.1 路径B 的可运行版：连 vLLM 暴露的 OpenAI 兼容端点做一次 chat 调用。
    另带 --stub：本机起一个只应答 /v1/chat/completions 的桩服务，
    用同一段客户端代码打过去，把「代码对不对」和「服务在不在」分开验证。

② 对应书稿
    第 8 章 §8.3.1「路径 B：自建推理服务」。

③ 怎么跑
    # A. 有 vLLM 服务（书稿做法）
    vllm serve <模型> --port 8000
    .venv/bin/python ch08/vllm_openai_compat.py

    # B. 没有 vLLM：只验证客户端代码
    .venv/bin/python ch08/vllm_openai_compat.py --stub

    # 端点/模型可覆盖：VLLM_BASE_URL / VLLM_MODEL

④ 依赖
    包：openai>=1.x（本机实测 3.3.0）
    服务：vLLM（或任何 OpenAI 兼容端点）；--stub 模式不需要
    权重：无（权重在 vLLM 侧）
    GPU：真跑书稿那个 `deepseek-ai/DeepSeek-V3` 需多卡；本机 RTX 3080 10GB 放不下

⑤ 实测解释器 + 结果
    解释器：.venv/bin/python（Python 3.10.12 / openai 3.3.0）
    - --stub：rc=0
        [stub] 已监听 http://127.0.0.1:8000/v1/chat/completions
        [stub] 模型: deepseek-ai/DeepSeek-V3
        [stub] 回复: 自注意力机制让序列中每个位置都能直接关注到其他所有位置：它把输入映射成
                     Query / Key / Value 三组向量，用 Q·K 的点积算相关性权重，再对 V 加权求和。
        [stub] 收到 1 次请求，已关闭
    - 不带参数（直连 localhost:8000，本机无服务）：rc=1
        直连 http://localhost:8000/v1（与书稿一致）。没有 vLLM 服务时会连接失败，可用 --stub 先验证客户端代码。
        ❌ 调用失败: APIConnectionError: Connection error.
    - 真 vLLM 服务：未验证（本机没起过服务；10GB 显存放不下 V3）。

⑥ 与书稿的差异（两处）
    1. 加了 `--stub` 开关：书稿假设 8000 端口已有服务，多数读者第一次跑并没有。
       不加参数时行为与书稿一致（直连 localhost:8000）。
    2. base_url / model 可用 VLLM_BASE_URL / VLLM_MODEL 覆盖，默认值与书稿相同
       （http://localhost:8000/v1、deepseek-ai/DeepSeek-V3）。
    调用的 4 个参数（model / messages / temperature / max_tokens）与书稿一字不差，
    见正文 `# ↓↓↓ 以下与书稿对应小节 一字不差` 标注段。
"""


import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from openai import OpenAI

base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
model = os.environ.get("VLLM_MODEL", "deepseek-ai/DeepSeek-V3")

REPLY = (
    "自注意力机制让序列中每个位置都能直接关注到其他所有位置："
    "它把输入映射成 Query / Key / Value 三组向量，"
    "用 Q·K 的点积算相关性权重，再对 V 加权求和。"
)


class _StubHandler(BaseHTTPRequestHandler):
    """只实现 /v1/chat/completions 的最小桩，用来验证客户端代码路径。"""

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self.server.seen.append(req)
        body = json.dumps({
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": req.get("model", model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 静音
        pass


def call_once(tag: str) -> None:
    # ↓↓↓ 以下与书稿对应小节 一字不差（只把 print 加了 tag） ↓↓↓
    client = OpenAI(
        base_url=base_url,      # 指向本地 vLLM 服务
        api_key="not-needed"    # 私有部署通常不需要真实 API Key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "解释一下 Transformer 中的自注意力机制。"}
        ],
        temperature=0.7,
        max_tokens=1024
    )

    print(f"[{tag}] 模型: {response.model}")
    print(f"[{tag}] 回复: {response.choices[0].message.content}")
    # ↑↑↑ 与书稿一致的部分结束 ↑↑↑


def main() -> int:
    if "--stub" not in sys.argv:
        print(f"直连 {base_url}（与书稿一致）。没有 vLLM 服务时会连接失败，"
              f"可用 --stub 先验证客户端代码。")
        try:
            call_once("live")
        except Exception as e:
            print(f"❌ 调用失败: {type(e).__name__}: {e}")
            return 1
        return 0

    srv = HTTPServer(("127.0.0.1", 8000), _StubHandler)
    srv.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[stub] 已监听 http://127.0.0.1:8000/v1/chat/completions")
    try:
        call_once("stub")
    finally:
        srv.shutdown()
        print(f"[stub] 收到 {len(srv.seen)} 次请求，已关闭")
    return 0


if __name__ == "__main__":
    sys.exit(main())
