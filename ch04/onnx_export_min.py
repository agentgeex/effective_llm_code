# -*- coding: utf-8 -*-
"""ch04/onnx_export_min.py —— PyTorch → ONNX → ONNX Runtime 全链路最小示例

① 这是什么：定一个 LSTM 文本分类器，用 torch.onnx.export 导出成 .onnx，再用 onnxruntime
   加载推理；两步分开跑，用于证明「推理不再需要 torch」。
② 对应书稿：第4章 §4.4.3 ONNX：跨平台的「模型世界语」
   TextClassifier + torch.onnx.export / ONNX Runtime 推理。
   书稿的 `python -m onnx.checker` 校验命令见 ③。
③ 怎么跑：
     V=python（需 onnxruntime + torch 的环境）
     $V ch04/onnx_export_min.py export     # 需要 torch + onnx
     $V -m onnx.checker text_classifier.onnx
     $V ch04/onnx_export_min.py infer      # 只需要 onnxruntime + numpy
④ 依赖：torch、onnx、onnxruntime、numpy —— **四个要在同一个环境里**。
   实测 venv-onnx：torch 2.6.0+cpu / onnx 1.23.2 / onnxruntime 1.31.1 / numpy 2.4.6。
   无模型权重、无网络、无外部服务。
⑤ 实测解释器：python（需 onnxruntime + torch 的环境）（唯一满足的解释器：需 onnxruntime+torch），
    重跑 export / checker / infer 三步（见文件末实测记录）。
⑥ 与书稿的差异：
   1) 书稿两段分成两个文件/两块，这里合成一个文件用 export|infer 子命令分开跑；
   2) `text_classifier.onnx` 是跑完即弃的中间产物，不入库（已在仓库根目录跑完删除）。
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
    )
    print("ONNX 模型已导出:", os.path.abspath(ONNX_PATH))


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
    elif action == "infer":
        infer()
    else:
        print(__doc__)
        sys.exit(2)

# ---- 实测记录（本机 venv-onnx）----
# 解释器：python（需 onnxruntime + torch 的环境）（torch 2.6.0+cpu / onnx 1.23.2 / onnxruntime 1.31.1）
#   ch04/onnx_export_min.py export → rc=0
#       ONNX 模型已导出: /tmp/text_classifier.onnx（11,035,007 字节）
#   python -m onnx.checker /tmp/text_classifier.onnx → rc=0
#   ch04/onnx_export_min.py infer  → rc=0  输出「输出 shape: (1, 5)」
# 导出时 torch 会发 LSTM + 非 1 batch 的 UserWarning，属正常提示，未屏蔽。
