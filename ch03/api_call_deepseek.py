# -*- coding: utf-8 -*-
"""ch03 §3.1 方法1：直接调用现有大模型服务 / 「API 调用：开发者的正确打开方式」

① 这是什么
   用 OpenAI SDK 调 DeepSeek 官方 API 的最小可运行示例：改 base_url + api_key 就能复用
   OpenAI 调用代码。没有 key 时本脚本不假装成功 —— 打印缺什么并退出码 2。

② 对应书稿
   第 3 章 §3.1「API 调用：开发者的正确打开方式」，书稿对应小节（唯一代码块，正文逐行照抄）。

③ 怎么跑
   export DEEPSEEK_API_KEY=sk-xxxx          # 书稿明确要求：key 只走环境变量，绝不硬编码
   <仓库根>/.venv-core/bin/python ch03/api_call_deepseek.py
   # 也可用 .venv/bin/python（openai 3.3.0）或 venv-onnx/bin/python（openai 2.54.0）

④ 依赖
   包：openai>=1.0（三个环境都有：.venv 3.3.0 / .venv-core 3.40.0 / venv-onnx 2.54.0）
   服务：api.deepseek.com（需外网 + 有效 DEEPSEEK_API_KEY）
   权重：无
   ⚠️ 系统 /usr/bin/python3 不行 —— 它的 openai 真 import 失败（缺 sniffio），见 ENV_MATRIX §1

⑤ 实测解释器
   实测解释器：<仓库根>/.venv-core/bin/python（openai 3.40.0）
   实测结果：❌ 未跑通 —— 本机没有 DEEPSEEK_API_KEY（`env | grep DEEPSEEK` 为空）
      · 无 key：rc=2，输出「缺少环境变量 DEEPSEEK_API_KEY（书稿 §3.1：绝不硬编码 key）」
      · 用无效 key（sk-invalid-probe-20261030）实跑到网络层：`openai.AuthenticationError: Error code: 401`
        —— 证明 openai 导入、base_url、请求构造、外网都是通的，**只缺有效凭证**
      · 语法/导入：.venv-core 与 venv-onnx 两个解释器 `python -m py_compile` 均 rc=0

⑥ 与书稿的差异
   1. 书稿把调用写在模块顶层；这里包进 main() + `sys.exit(main())`，只为把「缺 key」变成退出码。
   2. 缺 key 时提前返回，不再执行到 OpenAI() —— 书稿没有这段防御（书稿假设你已经 export 了）。
   3. 模型名与 base_url 与书稿一致（deepseek-v4-flash / https://api.deepseek.com）；
      模型名未验证：无有效 key，未确认该模型名当前是否仍可用，书稿自己已注明「以官方最新文档为准」。
"""
import os
import sys


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少环境变量 DEEPSEEK_API_KEY（书稿 §3.1：绝不硬编码 key）")
        return 2

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",          # 模型名以官方最新文档为准（本机无 key，未验证）
        messages=[
            {"role": "system", "content": "你是一个简洁的技术助手。"},
            {"role": "user", "content": "用一句话解释什么是 API。"},
        ],
    )
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
