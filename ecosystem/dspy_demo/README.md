# dspy_demo/ —— DSPy 示例去哪了

这个目录**没有代码**。DSPy 的示例统一放在 [`../../ch06/dspy_min.py`](../../ch06/dspy_min.py)，跑法见 [`../../ch06/README.md`](../../ch06/README.md)。

保留这个目录只是为了不让旧的链接 404。

## 为什么 DSPy 放在第 6 章而不是这里

因为它对应书里 §6.7.1「DSPy：当编译器开始替你写 Prompt」，属于第 6 章的主线内容 —— 按章归档比按技术栈归档更好找。

## 关于版本

DSPy 变化很快，本仓库的示例是在 **2.6.x** 上写的（`Signature` / `Predict` / `ChainOfThought` / `BootstrapFewShot` 这四个东西的用法在 2.x 里是稳定的）。

如果你装了 3.x 发现某个 import 路径不对，先看 `pip show dspy` 的版本，再对照官方迁移说明 —— 不要一上来就以为是示例写错了。

## 快速验证

```bash
pip install dspy
python3 ch06/dspy_min.py            # 离线桩 LM，不需要 key
python3 ch06/dspy_min.py --live     # 接真模型，需要 DEEPSEEK_API_KEY
```
