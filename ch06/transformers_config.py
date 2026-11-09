# -*- coding: utf-8 -*-
"""ch06 §6.2.4 —— 只读 config、按 config 搭骨架

书稿这段用两行代码说明白了一件事：`AutoConfig` 只读「结构」，不下载权重；
`AutoModelForCausalLM.from_config` 能按结构搭出一个**权重随机**的同构网络。
这解释了「为什么下载 3G 权重前就能知道它有几层、多大」——也是排查
`model_type` 不认识的报错（书稿对应小节 那个 ValueError）的第一手段。

运行：

    # 本地目录（推荐，零网络）：先在 modelscope 下一个模型，再指过来
    export MODEL_DIR=models/Qwen2.5-0.5B-Instruct
    python3 ch06/transformers_config.py

    # 也可以直接给仓库名（需要能访问 HF / 镜像）
    export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

依赖：transformers（实测 4.57.6）、torch（from_config 会实例化网络，必须有 torch）
      numpy —— 缺 torch 时 from_config 那步会报 ImportError，前半段仍能打印 config。
    实测解释器：python（需 onnxruntime + torch 的环境）（transformers 4.57.6）

实测状态：✅ 跑通 —— 环境 venv-onnx，
      模型 Qwen2.5-0.5B-Instruct（本地目录）：
      model_type=qwen2、num_hidden_layers=24、按 config 搭出的参数量 0.494B。
      实测时间：

与书稿差异：
  1. 模型来源改为「优先读环境变量 MODEL_DIR（本地目录），否则读 MODEL_ID（仓库名），
     都没有就报错退出」。书稿写死 "Qwen/Qwen2.5-0.5B-Instruct" 并注明「模型名示意」，
     照抄会强依赖外网。
  2. 加了一行「结构 vs 权重」的说明输出，让 from_config 与 from_pretrained 的区别一眼可见。
"""

import os
import sys

from transformers import AutoConfig, AutoModelForCausalLM


def main() -> int:
    src = os.environ.get("MODEL_DIR") or os.environ.get("MODEL_ID")
    if not src:
        print("缺少模型来源：请设置 MODEL_DIR（本地目录）或 MODEL_ID（仓库名）", file=sys.stderr)
        return 2

    # 只读配置，不下载权重文件
    config = AutoConfig.from_pretrained(src)
    print("来源:", src)
    print("model_type:", config.model_type)          # qwen2：决定用哪个实现类
    print("num_hidden_layers:", config.num_hidden_layers)
    print("hidden_size:", getattr(config, "hidden_size", "?"))
    print("vocab_size:", getattr(config, "vocab_size", "?"))

    # 按配置搭一个同结构的网络，权重是随机初始化的
    model = AutoModelForCausalLM.from_config(config)
    n = sum(p.numel() for p in model.parameters())
    print("按 config 搭出的参数量: %.3f B（权重随机，仅用于验证结构）" % (n / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
