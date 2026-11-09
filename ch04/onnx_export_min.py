# -*- coding: utf-8 -*-
"""ch04/onnx_export_min.py —— PyTorch → ONNX → ONNX Runtime 全链路最小示例

① 这是什么：定一个 LSTM 文本分类器，用 torch.onnx.export 导出成 .onnx，再用 onnxruntime
   加载推理；两步分开跑，用于证明「推理不再需要 torch」。
② 对应书稿：第4章 §4.4.3 ONNX：跨平台的「模型世界语」
   TextClassifier + torch.onnx.export / ONNX Runtime 推理。
   书稿用 `python -m onnx.checker` 校验；本文件换成了 ③ 的 `check` 子命令（原因见 ⑥-4）。
③ 怎么跑：
     V=python（需 onnxruntime + torch 的环境）
     $V ch04/onnx_export_min.py export     # 需要 torch + onnx
     $V ch04/onnx_export_min.py check      # 真校验导出结果（onnx.checker.check_model）
     $V ch04/onnx_export_min.py infer      # 只需要 onnxruntime + numpy
④ 依赖：torch、onnx、onnxruntime、numpy —— **四个要在同一个环境里**。
   本仓 `uv sync --all-extras` 一次装齐（torch 2.11.0 / onnx 1.23.2 / onnxruntime 1.23.2 / numpy 2.3.5）。
   无模型权重、无网络、无外部服务。
⑤ 实测解释器：.venv/bin/python（torch 2.11.0 / onnx 1.23.2 / onnxruntime 1.23.2 / numpy 2.3.5），
    重跑 export / check / infer 三步 rc 全 0（见文件末实测记录）。
⑥ 与书稿的差异：
   1) 书稿两段分成两个文件/两块，这里合成一个文件用 export|check|infer 子命令分开跑；
   2) `text_classifier.onnx` 是跑完即弃的中间产物，不入库（已在仓库根目录跑完删除）。
   3) 导出调用显式加了 `dynamo=False`：torch 2.9 起默认走 dynamo 导出器，而 dynamo 不接受
      书稿教的 `dynamic_axes` 写法。这是 **legacy（TorchScript）导出器**，torch 官方已标记为
      旧路径、将来会移除 —— 届时书稿和本文件要一起改。这里刻意与书稿保持一致，不换成
      `dynamic_shapes`。
   4) 书稿用 `python -m onnx.checker <file>` 校验。实测 onnx 1.23 上这条命令**不执行任何检查**
      （连不存在的文件都返回 rc=0，只吐一行 runpy 警告），所以本文件改用
      `onnx.checker.check_model(onnx.load(path))` 放在 `check` 子命令里，坏文件会真的报错。
   导出代码与推理代码本身与书稿一致。
"""
import os
import sys

ONNX_PATH = "text_classifier.onnx"


def build_and_export():
    """示例 1：定义 TextClassifier 并导出 ONNX"""
    import torch
    import torch.nn as nn

    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
            self.fc = nn.Linear(128, num_classes)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            _, (hidden, _) = self.lstm(x)
            return self.fc(hidden[-1])

    model = TextClassifier(vocab_size=10000, embed_dim=256, num_classes=5)
    model.eval()

    dummy_input = torch.randint(0, 10000, (1, 50))  # batch=1, seq_len=50
    # torch 2.9 起默认走 dynamo 导出器，而 dynamo 不接受老式 dynamic_axes；
    # 这里显式关掉以保持与书稿一致（代价：走 legacy TorchScript 导出器，torch 已标记为旧路径）
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    print("ONNX 模型已导出:", os.path.abspath(ONNX_PATH))


def check():
    """校验导出的 .onnx 是否合法。

    书稿用的是 `python -m onnx.checker <file>`，但实测 onnx 1.23 上那条命令
    连不存在的文件都返回 rc=0（不执行任何检查），因此这里直接调 API。

    失败分两种，都会非零退出：
      · 根本不是 protobuf：onnx.load 抛 google.protobuf.message.DecodeError
      · 能解析但结构不合法：onnx.checker.check_model 抛 onnx.checker.ValidationError
    """
    import onnx

    assert os.path.exists(ONNX_PATH), "先跑 export 子命令生成 " + ONNX_PATH
    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)
    print("ONNX 校验通过:", os.path.abspath(ONNX_PATH))


def infer():
    """示例 2：用 ONNX Runtime 推理（不需要 PyTorch）"""
    import numpy as np
    import onnxruntime as ort

    assert os.path.exists(ONNX_PATH), "先跑 export 子命令生成 " + ONNX_PATH
    session = ort.InferenceSession(ONNX_PATH)
    input_ids = np.random.randint(0, 10000, (1, 50), dtype=np.int64)
    outputs = session.run(None, {"input_ids": input_ids})
    print("输出 shape:", outputs[0].shape)  # (1, 5)


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "export":
        build_and_export()
    elif action == "check":
        check()
    elif action == "infer":
        infer()
    else:
        print(__doc__)
        sys.exit(2)

# ---- 实测记录 ----
# 【旧】venv-onnx / torch 2.6.0+cpu / onnx 1.23.2 / onnxruntime 1.31.1
#   export → rc=0（ONNX 模型 11,035,007 字节）
#   `python -m onnx.checker /tmp/text_classifier.onnx` → rc=0
#     ⚠️ 这条 rc=0 是假的：同一条命令喂不存在的文件也返回 0，它根本没校验。
#   infer  → rc=0，输出 shape (1, 5)
#
# 【新】.venv（uv sync --all-extras）/ Python 3.11.15
#        torch 2.11.0+cu130 / onnx 1.23.2 / onnxruntime 1.23.2 / numpy 2.3.5
#   export                     → rc=0  ONNX 模型已导出（11,035,007 字节）
#   check  text_classifier.onnx → rc=0  ONNX 校验通过
#   check  <随便写的 txt>       → rc=1  DecodeError: Wire format was corrupt（证明校验真的在工作）
#   infer                      → rc=0  输出 shape (1, 5)
# 导出时 torch 会发 LSTM + 非 1 batch 的 UserWarning，属正常提示，未屏蔽。
