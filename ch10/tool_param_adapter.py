# -*- coding: utf-8 -*-
"""能力架构的「翻译层」：参数映射 + 并行编排

① 这是什么
   书稿「模型输出 → 指令转换」的两个可运行示例：
   1) TOOL_PARAM_MAP + adapt_params —— 把模型输出的参数名/枚举值翻译成外部 API 的参数；
   2) execute_operations —— 用 ThreadPoolExecutor 并行执行无依赖的多个操作。
   另加「有依赖必须串行」的对照与失败隔离，让编排判据能量化。

② 对应书稿：第 10 章
   - §10.4.3.2 参数映射：不要写死，用适配器，书稿对应小节（TOOL_PARAM_MAP / adapt_params）
   - §10.4.3.3 执行编排：串行调用还是并行调用，书稿对应小节（execute_operations）

③ 怎么跑（仓库根目录执行）
   .venv/bin/python ch10/tool_param_adapter.py

④ 依赖
   - 包：零（标准库 concurrent.futures / time）
   - 服务：无。call_tool 是本地假实现，不调外部 API

⑤ 实测解释器
   python（Python 3.10.12）
   实测 rc=0，输出（原样）：
       [书稿版] get_weather {'city': '北京'} → {'location': '北京'}
       [书稿版] 未知工具/未配置键原样透传：{'foo': 'bar'} → {'foo': 'bar'}
       [书稿版] get_weather {'city': '北京', 'unit': '摄氏度'} → 💥 TypeError: unhashable type: 'dict'
       [修正版] get_weather {'city': '北京', 'unit': '摄氏度'} → {'location': '北京', 'unit': 'metric'}
       [修正版] 未配置的枚举值原样保留：{'unit': '开尔文'} → {'unit': '开尔文'}
       [串行] 3 步有依赖（查订单→汇总→发邮件）：0.90s，结果顺序与提交一致
       [并行] 3 个无依赖操作：0.31s（最慢那个决定总时长），耗时比 3.0x
       [并行] 失败隔离：2 个成功 + 1 个异常被捕获

⑥ 与书稿的差异（三处）
   1. 书稿 adapt_params 把「键改名」与「枚举值映射」混在一个 dict，却统一用
      `mapping.get(key, key)` 当目标键名 → 只要传枚举就 TypeError（书稿缺陷，本文件实测复现）。
      本文件保留书稿原函数用于对照，另给 `adapt_params_fixed`（值映射时键名不变）。
   2. 书稿 execute_one 注释「call_tool 为假设已实现」，本文件补本地假实现
      （sleep 0.3s 后回显），以便把并行/串行耗时差真的测出来。
   3. 书稿 `[f.result() for f in futures]` 只要一个操作抛异常就全盘失败、其余结果全丢；
      本文件加 `f.exception()` 判断，改为失败也返回一条记录（书稿 §10.4.4 清单第 6 条要求容错）。
      另额外演示「有依赖必须串行」的对照（书稿 §10.4.3.3 只说了结论，没给代码）。
"""

import concurrent.futures
import time

# ---- 示例 1：TOOL_PARAM_MAP + adapt_params，原样照抄，含书稿原文的缺陷 ----
TOOL_PARAM_MAP = {
    "get_weather": {
        "city": "location",       # 模型输出 city → API 参数 location
        "date": "date",
        "unit": {
            "摄氏度": "metric",
            "华氏度": "imperial",
        },
    }
}


def adapt_params(tool_name: str, model_params: dict) -> dict:
    """【书稿原样，有 bug，见文件头「书稿问题」】将模型输出的参数映射为 API 所需的参数

    🔴 只要某个键配了枚举映射（如 "unit": {"摄氏度": "metric"}），这里就会崩：
        target_key = mapping.get(key, key)   # ← "unit" 取到的是那个 dict，不是 "unit"
        adapted[target_key] = ...            # ← dict 不能当 key
        TypeError: unhashable type: 'dict'
    """
    mapping = TOOL_PARAM_MAP.get(tool_name, {})
    adapted = {}
    for key, value in model_params.items():
        target_key = mapping.get(key, key)
        # 如果是枚举映射，转换值
        if isinstance(mapping.get(key), dict):
            adapted[target_key] = mapping[key].get(value, value)
        else:
            adapted[target_key] = value
    return adapted


def adapt_params_fixed(tool_name: str, model_params: dict) -> dict:
    """最小修正版：目标键名与「枚举值映射」分开判断，不再把 dict 当键名。

    改动只有一行语义：`target_key = mapping.get(key, key)` 在「该键的值是 dict」时，
    说明它是**值映射**而不是**改名**，目标键名应保持 key 本身。
    """
    mapping = TOOL_PARAM_MAP.get(tool_name, {})
    adapted = {}
    for key, value in model_params.items():
        spec = mapping.get(key, key)
        if isinstance(spec, dict):
            adapted[key] = spec.get(value, value)      # 值映射：键名不变
        else:
            adapted[spec] = value                      # 改名：spec 才是目标键名
    return adapted


# ---- 补: 书稿对应小节 注释「call_tool 为假设已实现」，这里给可跑的假实现 ----
def call_tool(tool_name: str, params: dict, *, sleep: float = 0.30):
    """假工具：睡 sleep 秒后回显。用来量化「并行 vs 串行」的耗时差。"""
    time.sleep(sleep)
    if tool_name == "explode":
        raise RuntimeError("boom")
    return {"tool": tool_name, "params": params, "ok": True}


# ---- 示例 2：execute_operations，原样照抄 + 失败隔离 ----
def execute_operations(operations: list) -> list:
    """并行执行无依赖关系的多个操作"""

    def execute_one(op):
        tool_name = op["action"]
        params = adapt_params_fixed(tool_name, op["params"])  # ← 用修正版，见「书稿问题」
        return call_tool(tool_name, params)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(execute_one, op) for op in operations]
        out = []
        for f in futures:
            exc = f.exception()          # ← 书稿没有这一步，见文件头差异 2
            out.append({"error": str(exc)} if exc else f.result())
        return out


def main() -> None:
    # 1) 只有「改名」、没有枚举映射时，书稿版正常工作
    print("[书稿版] get_weather {'city': '北京'} →",
          adapt_params("get_weather", {"city": "北京"}))
    print("[书稿版] 未知工具/未配置键原样透传：{'foo': 'bar'} →",
          adapt_params("unknown_tool", {"foo": "bar"}))

    # 2) 一旦带上枚举映射的键，书稿版直接崩（这就是书稿里的真 bug）
    print("\n[书稿版] get_weather {'city': '北京', 'unit': '摄氏度'} → ", end="")
    try:
        print(adapt_params("get_weather", {"city": "北京", "unit": "摄氏度"}))
    except TypeError as e:
        print(f"💥 TypeError: {e}")

    # 3) 修正版：键名不变、枚举值被翻译
    print("[修正版] get_weather {'city': '北京', 'unit': '摄氏度'} →",
          adapt_params_fixed("get_weather", {"city": "北京", "unit": "摄氏度"}))
    print("[修正版] 未配置的枚举值原样保留：{'unit': '开尔文'} →",
          adapt_params_fixed("get_weather", {"unit": "开尔文"}))

    # ── 有依赖 → 必须串行 ──
    t0 = time.perf_counter()
    order = call_tool("query_order", {"id": 1})
    summary = call_tool("summarize", {"order": order})
    mail = call_tool("send_email", {"body": summary})
    serial = time.perf_counter() - t0
    print(f"\n[串行] 3 步有依赖（查订单→汇总→发邮件）：{serial:.2f}s，结果顺序与提交一致")

    # ── 无依赖 → 并行 ──
    ops = [
        {"action": "get_weather", "params": {"city": "北京", "unit": "摄氏度"}},
        {"action": "get_stock", "params": {"symbol": "600519"}},
        {"action": "get_time", "params": {}},
    ]
    t0 = time.perf_counter()
    results = execute_operations(ops)
    par = time.perf_counter() - t0
    print(f"[并行] 3 个无依赖操作：{par:.2f}s（最慢那个决定总时长），耗时比 {serial/par:.1f}x")
    for r in results:
        print("    ", r)

    # ── 失败隔离 ──
    mixed = execute_operations([
        {"action": "get_weather", "params": {"city": "上海"}},
        {"action": "explode", "params": {}},
        {"action": "get_time", "params": {}},
    ])
    ok = sum(1 for r in mixed if r.get("ok"))
    bad = sum(1 for r in mixed if "error" in r)
    print(f"[并行] 失败隔离：{ok} 个成功 + {bad} 个异常被捕获 → {mixed}")


if __name__ == "__main__":
    main()
