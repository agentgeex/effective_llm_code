# -*- coding: utf-8 -*-
"""ch06 §6.2.1 —— PyTorch 环境自检

书稿这一段是「装完 PyTorch 后先确认装对了」的最小检查：版本、CUDA 是否可用、
显卡型号、CUDA 版本、显存总量。五句话，抄下来就能跑，也是最容易踩坑的地方
（装成 CPU 版、CUDA 版本不匹配、显存被别的进程占着）。

运行：

    python3 ch06/torch_env_check.py

依赖：torch。没装会 ImportError；装了 CPU 版不会报错，但「CUDA 可用」会是 False，
      这本身就是最有用的信息。
    实测解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu）

实测状态：✅ 跑通 —— 本机 /usr/bin/python3（torch 2.7.0+cu126）输出
      版本 2.7.0+cu126、CUDA 可用 True、显卡 NVIDIA GeForce RTX 3080、显存 10.0 GB。
      注意：显存那一行书稿写的是 `total_mem`，torch 2.x 已改名为 `total_memory`，
      直接照抄会 AttributeError。本文件做了兜底（见第 3 段注释）。

与书稿差异：
  1. 显存字段 `total_mem` → `total_memory`（torch 2.x 改名）。为避免读者在不同版本间来回跳，
     这里用 getattr 双探测，两代都打印。
  2. 加了 `if __name__` 保护与「CPU 版 torch」的提示语 —— 书稿是纯片段，
     作为示例文件需要一个明确出口。
"""

import torch


def main() -> int:
    # 1. 检查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # 2. 检查 CUDA 是否可用
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_ok}")

    # 3. 如果 CUDA 可用，查显卡型号和 CUDA 版本
    if cuda_ok:
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", None) or getattr(props, "total_mem", None)
        print(f"显存总量: {total / 1024**3:.1f} GB")
    else:
        print("提示：当前是 CPU 版 torch（或没有可用 GPU）。要跑训练/GPTQ/4bit 量化，需换 CUDA 版。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
