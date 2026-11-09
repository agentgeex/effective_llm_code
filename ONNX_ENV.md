# ONNX_ENV —— ONNX 专用环境与四个示例的实测记录

> 本文件记录**这台机器上真实跑过的**结果。模型与虚拟环境都在仓库外（仓库外的模型与环境目录），
> 不随 git 走，所以读者按下面的命令自己建一遍即可复现。

## 一、环境

| 项 | 值 |
|---|---|
| 位置 | `venv-onnx`（Python 3.11.15） |
| 创建命令 | `uv venv venv-onnx --python 3.11` |
| 硬件 | RTX 3080 10GB（本环境用 **CPU 版 torch**，不占显存）；内存 31G |
| 磁盘 | `/` 剩 67G（模型 + 导出 + 量化共占约 6G） |

### 装了什么（逐个 import 验过）

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 1) torch 用 CPU 轮子，避免拖 2GB+ 的 CUDA 依赖
#    注意：+cpu 轮子只在 download.pytorch.org 上有，清华源没有同名版本
curl -O https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl
uv pip install --python venv-onnx/bin/python --find-links . \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --index-strategy unsafe-best-match "torch==2.6.0+cpu"

# 2) 其余依赖
uv pip install --python venv-onnx/bin/python \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy onnx onnxruntime safetensors transformers accelerate \
    "optimum[onnxruntime]" modelscope
```

实测版本（`import` 打印，不是猜的）：

```
torch          2.6.0+cpu
numpy          2.4.6
onnx           1.23.2
onnxruntime    1.31.1
safetensors    0.8.0
transformers   4.57.6      # 装 optimum 时从 5.22.0 降下来的，optimum 2.1.0 尚不支持 5.x
accelerate     1.15.0      # ch03/transformers_infer.py 的 device_map="auto" 必须它，缺了报
                           #   ValueError: ... requires accelerate
optimum        2.1.0
optimum-onnx   0.1.0
modelscope     1.40.2
```

> ⚠️ 踩坑记一笔：先装的是 `transformers 5.22.0`，再装 `optimum[onnxruntime]` 时被自动降到 `4.57.6`。
> 如果反过来先装 optimum，再手工升级 transformers 到 5.x，`optimum` 会 import 失败。

## 二、模型

| 项 | 值 |
|---|---|
| 源模型 | `Qwen/Qwen2.5-0.5B-Instruct`（modelscope 下载） |
| 下载体积 | **954M**（`models/Qwen2.5-0.5B-Instruct`） |
| 下载命令 | `modelscope.snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir=...)` |
| 速度 | 11.8 MB/s，用时 1 分 26 秒 |

派生目录（都由脚本生成，删了可重建）：

| 目录 | 体积 | 说明 |
|---|---|---|
| `models/Qwen2.5-0.5B-onnx` | 2.4G | `--task text-generation`（**无** KV cache），量化脚本的输入 |
| `models/Qwen2.5-0.5B-onnx-past` | 2.4G | `--task text-generation-with-past`（**有** KV cache），`onnx_m.py` 必须用这个 |
| `models/Qwen2.5-0.5B-int8` | 602M | INT8 动态量化产物（`model.onnx` + `model.onnx.data`） |

导出命令：

```bash
venv-onnx/bin/optimum-cli export onnx \
    --model models/Qwen2.5-0.5B-Instruct --task text-generation-with-past \
    models/Qwen2.5-0.5B-onnx-past
```

> 导出的 fp32 比原 bf16 大一倍多（988M → 2.4G），这是正常的：ONNX 默认 fp32。
> 导出结束会有 `max diff = 0.00023` 的精度警告（atol 1e-5 未达标），属 fp32 转换舍入误差，不影响使用。

## 三、四个示例的实测结果

统一前置：`export` 好模型目录后，各脚本的路径**全部走环境变量**，源码里没有绝对路径。

### 1. `ch04/onnx_export_min.py` —— ✅ 跑通

```bash
V=python（需 onnxruntime + torch 的环境）
$V ch04/onnx_export_min.py export          # 导出 text_classifier.onnx（11M）
$V -m onnx.checker text_classifier.onnx    # 校验通过（无输出即通过）
$V ch04/onnx_export_min.py infer           # → 输出 shape: (1, 5)
```

关键输出：

```
ONNX 模型已导出: <仓库根>/text_classifier.onnx
输出 shape: (1, 5)
```

导出时的警告（真实存在，不影响运行）：

```
UserWarning: Exporting a model to ONNX with a batch_size other than 1, with a variable length
with LSTM can cause an error when running the ONNX model with a different batch size.
```

> 这个 `text_classifier.onnx` 是**跑完即弃**的中间产物，已从仓库删掉，不入库。

### 2. `_archive_not_in_book/legacy_stack_20261102/model/onnx/onnx_m.py` —— ✅ 跑通（1.80 秒）

```bash
export ONNX_MODEL_DIR=models/Qwen2.5-0.5B-onnx-past
$V _archive_not_in_book/legacy_stack_20261102/model/onnx/onnx_m.py
```

真实输出：

```
Device set for ORT: cpu
Loading ORTModel...
Model Loaded successfully.
--- ONNX 模型生成结果 ---
量子计算利用量子比特（qubits）进行运算，比传统计算机更高效。

生成耗时: 1.80 秒
```

> ❌ 若指向 `Qwen2.5-0.5B-onnx`（无 past）：`ValueError: use_cache=True was passed to the model
> but the loaded model does not support pkv cache reuse`。**必须用 with-past 导出的目录。**

### 3. `_archive_not_in_book/legacy_stack_20261102/model/onnx/quantize_qwen.py` —— ✅ 跑通（约 3 分钟）

```bash
export ONNX_MODEL_DIR=models/Qwen2.5-0.5B-onnx
export ONNX_QUANT_DIR=models/Qwen2.5-0.5B-int8
$V _archive_not_in_book/legacy_stack_20261102/model/onnx/quantize_qwen.py
```

真实输出：

```
ORTQuantizer 初始化成功。
配置创建成功 (INT8 动态量化, AVX2, Per-Channel)。
--- ✅ 量化成功！---
量化后的模型已保存到: models/Qwen2.5-0.5B-int8
```

体积：**2.4G（fp32） → 602M（INT8）**，约 1/4。

⚠️ **改了一处代码**（不改逻辑，只兼容 API）：optimum 2.x 的 `ORTQuantizer.quantize()`
已删除 `file_names` / `export_file_names`，原写法直接

```
TypeError: ORTQuantizer.quantize() got an unexpected keyword argument 'file_names'
```

现在按 `inspect.signature` 探测：有 `file_names` 走旧路（optimum 1.x），没有则传 `file_suffix=""`
+ `use_external_data_format=True`（optimum 2.x）。两条路都保留。

### 4. `_archive_not_in_book/legacy_stack_20261102/model/onnx/onnx_quantized.py` —— ✅ 跑通（0.06 秒）

```bash
export ONNX_QUANT_DIR=models/Qwen2.5-0.5B-int8
$V _archive_not_in_book/legacy_stack_20261102/model/onnx/onnx_quantized.py
```

真实输出：

```
--- 量化模型生成结果 ---
量子

生成耗时: 0.06 秒
```

> 只出一个词是对的：这个示例只做「一次前向 + 取末位 argmax」，是书稿有意简化
> （原文注释「简化的生成过程（实际需要处理past_key_values等）」）。
> 要连续生成请用 `onnx_m.py`。

⚠️ **改了两处**：
1. 权重文件名从硬编码 `model_quantized.onnx` 改成「先找 `model_quantized.onnx`，
   再找 `model.onnx`」——optimum 2.x 的量化产物叫后者。
2. 输入从「只喂 `input_ids`」补成三件套。导出的图声明了三个输入，缺一个直接报错：

```
inputs: [('input_ids', int64), ('attention_mask', int64), ('position_ids', int64)]
outputs: [('logits', ['batch_size','sequence_length', 151936])]
```

## 四、四个示例的结论

| 示例 | 状态 | 耗时 | 备注 |
|---|---|---|---|
| `ch04/onnx_export_min.py` | ✅ 跑通 | 秒级 | export + checker + infer 三步全过 |
| `_archive_not_in_book/legacy_stack_20261102/model/onnx/onnx_m.py` | ✅ 跑通 | 1.80s | 必须 with-past 目录 |
| `_archive_not_in_book/legacy_stack_20261102/model/onnx/quantize_qwen.py` | ✅ 跑通 | ~3min | 2.4G → 602M；API 已变，加了兼容分支 |
| `_archive_not_in_book/legacy_stack_20261102/model/onnx/onnx_quantized.py` | ✅ 跑通 | 0.06s | 单步 argmax，只出一个词 |

**4/4 跑通。** 但要说清：跑通用的是 0.5B 小模型 + CPU；书稿原文用的是 Qwen1.5-4B，
本机没下（4B fp32 导出约 16G，超出「≤2GB 小模型」的口径，也没有必要）。
换大模型只需改 `ONNX_MODEL_DIR` 指向对应导出目录，代码不用动。

## 五、清理与不入库的东西

- 模型权重、导出目录、量化产物：全在 仓库外的模型与环境目录（仓库外），**git 里没有任何大文件**。
- `ch04/onnx_export_min.py` 生成的 `text_classifier.onnx`（11M）跑完已删。
- 不要把这些目录软链进仓库 —— `.gitignore` 没挡它们。

## 六、这个环境顺带验证的另外三个示例（同口径，避免与文件头自相矛盾）

`venv-onnx` 装齐 torch + transformers + accelerate + safetensors 之后，ch03/ch04 有几个
原本「没环境跑」的示例也通了。文件头的「实测状态」与本节**同一口径**，写的就是下面的结果。

| 文件 | 命令 | 关键输出 | rc |
|---|---|---|---|
| `ch03/transformers_infer.py` | `MODEL_NAME=.../Qwen2.5-0.5B-Instruct venv-onnx/bin/python ch03/transformers_infer.py` | 「人工智能（Artificial Intelligence，简称AI）是指由计算机系统所表现出的智能行为。…」 | 0 |
| `ch04/safetensors_read.py` | `venv-onnx/bin/python ch04/safetensors_read.py` | `tensor: (8, 8) torch.float32` / `storage offset: 0` | 0 |
| `ch04/onnx_export_min.py` | `export` → `-m onnx.checker` → `infer` | `输出 shape: (1, 5)`，checker rc=0 | 0 |

### 两个卫生/依赖细节（本轮修）

1. **`ch04/safetensors_read.py` 不再弄脏仓库**：原先分片写在当前工作目录
   （`model-00001-of-00002.safetensors`），读者一跑仓库就多一个未跟踪文件。
   现在写到 `tempfile.mkdtemp()`，`finally` 里 `shutil.rmtree` 清掉，实测跑完 `git status --porcelain` 为 0 行。
2. **`ch03/transformers_infer.py` 的依赖补上 `accelerate`**：`device_map="auto"` 缺它会报
   `ValueError: ... requires accelerate`，已写进该文件「依赖」段和上面的版本清单。

### 仍然跑不通的（唯一一条真的）

`ch03/api_call_deepseek.py` —— 缺 `DEEPSEEK_API_KEY`。外网是通的
（`curl https://api.deepseek.com/v1/models` → **401**，域名可达、只是没凭证），
所以「无外网」这个说法不成立，已从文件头删掉。
