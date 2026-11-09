# 第 10 章 大模型应用架构设计

## 这章讲什么

一个能上线的系统，跟一段能跑的 demo 之间隔着架构：**怎么让模型稳定地输出结构化结果**、**怎么把参数映射到工具**、**串行还是并行**、**上下文怎么组装**、**七类模型的职责怎么分**。

这一章的代码都很小，但每一个都对应一个真实会翻车的点。三个脚本全是**零第三方依赖**，装个 Python 就能跑 —— 目的是让你把注意力放在设计上，而不是环境上。

## 本目录的文件

| 文件 | 作用 | 能跑吗 |
|---|---|---|
| [`concurrency_vs_serial.py`](concurrency_vs_serial.py) | 串行 vs 并发的真实差距：同一批任务两种跑法，打印耗时对比 | **A 可真跑**（零依赖） |
| [`structured_output.py`](structured_output.py) | 让模型稳定吐 JSON：从「解析自由文本」到「约束输出格式」的演进 | **A 可真跑**（零依赖） |
| [`tool_param_adapter.py`](tool_param_adapter.py) | 工具参数适配器：模型给的参数和工具要的参数对不上时怎么兜，以及失败隔离 | **A 可真跑**（零依赖） |

## 怎么跑

```bash
python3 ch10/concurrency_vs_serial.py
python3 ch10/structured_output.py
python3 ch10/tool_param_adapter.py
```

三个脚本互不依赖，顺序随便。**建议看输出再读代码** —— 尤其是第一个：串行 0.90 秒、并发 0.31 秒，这个 3 倍差距是这一章最直观的一课。

## 依赖与前置

- **零第三方依赖**，只用标准库（`json` / `re` / `concurrent.futures` / `time` 等）。
- 不需要 key、不需要权重、不需要 GPU、不需要网络。
- 任何 Python 3.8+ 都能跑。

## 常见报错

**`TypeError: unhashable type: 'dict'`**
这是书里那版并发示例会踩的坑：把 dict 当 key 塞进集合/字典。要用 dict 做键，转成 `tuple(sorted(d.items()))` 或 `json.dumps(..., sort_keys=True)`。

**`NameError: name 'io' is not defined` / `name 'Image' is not defined`**
书里 §10.1.4 那段图片预处理示例本身缺 `import io` 和 `from PIL import Image`，照抄会直接 NameError。补上这两行即可 —— 这属于示例的疏漏，不是你的环境问题。

**`ModuleNotFoundError: No module named 'sniffio'`（导入 openai 时）**
系统解释器里装了个残缺的 openai 依赖链。换虚拟环境，或者补装 `sniffio`。本目录的三个脚本都不需要 openai，影响不到它们。

**`import vllm` 在仓库根能成功，换个目录就失败**
不是幻觉：如果仓库里有个叫 `vllm` 的空目录，Python 3 会把它当命名空间包导进去。判断方法：`python3 -c "import vllm; print(vllm.__file__)"`，看路径指向哪。

**并发跑完比串行还慢**
如果你那台机器只有一个核，或者任务本身在等同一把锁，并发不会更快。并发的收益来自「等待」被重叠，不来自「CPU 更多」。
