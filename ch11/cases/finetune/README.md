# 对一个大模型进行微调（§11.3）

## 这个案例干什么

微调的成败在数据，不在算法。这个案例把「造数据」这件事拆成三步，每步都能单独跑、单独看结果：

| 文件 | 这一步做什么 |
|---|---|
| `prepare_data.py` | 采集 → 清洗 → 语义聚类 + 三合一划分（train/val/test），产出 `data/*.jsonl` 与统计 |
| `train_lora.py` | 用 Qwen2.5-0.5B-Instruct 做 LoRA 真训练 |
| `infer_compare.py` | 同一批 prompt 的微调前后并排对比（原文并排，不评分不排名） |

领域是 **CT 医疗设备维保**（球管、探测器、环形伪影、空气校准…），和书里 §11.3 的场景一致。

## 对应书哪一节

第 11 章 §11.3「对一个大模型进行微调」：§11.3.1 数据采集、§11.3.2 数据清洗、§11.3.3 数据分布与防污染、§11.3.4 实战验证。

## 怎么跑

```bash
# 第 1 步：造数据（不需要 GPU、不需要 API key）
pip install scikit-learn
python ch11/cases/finetune/prepare_data.py
# → data/train.jsonl  data/val.jsonl  data/test.jsonl  outputs/dataset_stats.json

# 第 2 步：训练（需要 GPU；CPU 也能跑但很慢）
pip install torch transformers peft datasets
python ch11/cases/finetune/train_lora.py

# 第 3 步：对比微调前后（需要第 2 步的 adapter）
python ch11/cases/finetune/infer_compare.py
```

## 依赖什么

| 步骤 | 依赖 | 没有它会怎样 |
|---|---|---|
| `prepare_data.py` | `scikit-learn` | `ModuleNotFoundError: sklearn` |
| `train_lora.py` | `torch` / `transformers` / `peft` / `datasets`，以及基座权重 | 报缺包；没有权重时无法离线加载 |
| `infer_compare.py` | 同上 + 第 2 步产出的 `adapter/` | 明确报「找不到 adapter」 |

基座模型默认 `Qwen/Qwen2.5-0.5B-Instruct`。0.5B 是为了让一张消费卡也能跑完；换更大的模型改脚本顶部的常量即可。

## 数据从哪来

**没有外部数据集，全部由脚本合成。** 仓库里带 48 条人工撰写的 CT 维保事实 × 6 种问法 = 约 288 条问答对，`prepare_data.py` 现场生成，不落库到 git（`data/` 在 `.gitignore` 里）。

这意味着：

- 你 clone 下来就能跑第 1 步，**不需要下载任何数据**。
- 但这份数据的**风格极其单一**（模板合成），训出来的模型也只会那一种语气。这是本案例最大的局限，README 不掩饰。

## 提醒

- 48 条种子事实是**教学示意**，不是任何厂家维修手册的原文，参数与阈值不能当操作规程用。
- 本案例证明的是**数据管道的形状**（怎么清洗、怎么防污染、怎么划分），不是「微调能提升多少准确率」。
- `outputs/`、`data/`、`adapter/` 都不入库，跑一遍就有了。
