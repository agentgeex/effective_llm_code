# 第 13 章 边界与克制：大模型不该做的「深水区」

## 这章讲什么

上一章讲怎么把事情做成，这一章讲**哪些事情不要交给模型**。

两个最典型的陷阱：**算术**（它能算出看起来很像的错答案，而且语气一样自信）和**工具调度**（让它自己决定调哪个工具、参数怎么填，错一步后面全错）。这一章用可运行的代码把这些坑复现出来，再给出护栏写法。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`arithmetic_trap_demo.py`](arithmetic_trap_demo.py) | 算术陷阱：同一个式子，让模型算 vs 用代码算，把差异摆出来；含 `Decimal` 的正确做法 | **A 可真跑**（零第三方依赖，离线） |
| [`tool_dispatch_demo.py`](tool_dispatch_demo.py) | 工具调度的护栏：参数校验、失败隔离、错误回填，而不是「相信模型填对了」 | **A 可真跑**（零第三方依赖，离线） |

两个脚本都带 `--live` 开关，用来接真模型对照。

## 怎么跑

```bash
# 离线模式（默认）：不需要 key，不需要网络
python3 ch13/arithmetic_trap_demo.py
python3 ch13/tool_dispatch_demo.py

# 接真模型对照
export DEEPSEEK_API_KEY=你的key     # 或 OPENAI_API_KEY
python3 ch13/arithmetic_trap_demo.py --live
python3 ch13/tool_dispatch_demo.py --live
```

## 依赖与前置

| 模式 | 需要什么 |
|---|---|
| 离线（默认） | 只要 Python 标准库。**任何 Python 3.8+ 都能跑** |
| `--live` | `openai` SDK + `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` |

不需要 GPU、不需要模型权重。

## 常见报错

**`✗ --live 需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY`（退出码 1）**
设计如此。离线模式照样能跑，而且离线模式才是这一章的重点 —— 它证明的是「模型算错」这个现象本身，不需要真调模型就能看清。

**`ModuleNotFoundError: No module named 'sniffio'`（导入 openai 时）**
系统解释器里 openai 的依赖链不完整。换虚拟环境，或者 `pip install sniffio`。只影响 `--live`，离线模式不受影响。

**算术结果差一位小数 / 尾数不对**
不是你的代码错了，这正是这一章要演示的东西：浮点和十进制的表示差异，模型「心算」时也踩同一个坑。要精确就用 `Decimal`，别用 `float`。
