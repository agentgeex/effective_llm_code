# -*- coding: utf-8 -*-
"""并发 vs 串行 —— 大模型批处理为什么必须并发

① 这是什么
   书稿「并发 vs 串行」示例的可运行版：100 张工单，分别用串行、并发(10)、全并发跑一遍，
   把「单任务延迟」与「整体吞吐量」的差别量化出来。

② 对应书稿：第 10 章 §10.1.5（核心目标的评估与确认 · 时间维度），书稿对应小节；
   紧随其后的设计要点正文（本文件末尾的 print 就是这段的浓缩）。

③ 怎么跑（仓库根目录执行）
   .venv/bin/python ch10/concurrency_vs_serial.py

④ 依赖
   - 包：零（只用标准库 asyncio / time）
   - 服务：无。不需要 API key、不需要 GPU

⑤ 实测解释器 + 日期
   python（Python 3.10.12）· 
   实测 rc=0，输出（原样）：
       100 张工单，单张耗时 0.30s（书稿假设 30s，这里压缩 100 倍便于本地跑）
         串行            :  30.06s  理论  30.00s
         并发(10)        :   3.02s  理论   3.00s
         并发(全部100)     :   0.30s  理论   0.30s
       加速比：串行/并发(10) = 10.0x，串行/并发(全部) = 98.8x
   （与  首跑 30.06 / 3.01 / 0.30 的差异仅为调度毛刺，结论不变）

⑥ 与书稿的差异（一处，只为实现；结论与书稿一致）
   书稿用 `llm.chat_async(ticket)`（每张 30 秒），该函数在书里未定义。
   本文件用 `asyncio.sleep(0.30)` 代替单张耗时（压缩 100 倍便于本地跑），
   并把并发度做成参数（书稿注释写「10 并发」，代码里其实没体现限流）。
   书稿「100 张 × 30s = 3000s；10 并发 → 300s」的比例，实测完全吻合
   （30.06s vs 3.02s，差值来自事件循环调度尾数）。
"""

import asyncio
import time

N = 100             # 工单数（同书稿）
ONE = 0.30          # 单张耗时，书稿是 30s，压缩 100 倍便于本地跑
CONCURRENCY = 10    # 书稿举例的并发度


async def fake_llm(ticket: int) -> str:
    """代替书稿的 llm.chat_async(ticket)：模拟一次 0.30s 的模型调用"""
    await asyncio.sleep(ONE)
    return f"工单{ticket}已处理"


async def process_serial(tickets: list) -> list:
    """串行：一次一张（书稿 process_tickets_serial）"""
    results = []
    for t in tickets:
        results.append(await fake_llm(t))
    return results


async def process_parallel(tickets: list, concurrency: int) -> list:
    """并行：书稿是 asyncio.gather 一次性全发。
    这里多一个 concurrency 参数（信号量限流）—— 真实系统里并发要受 RPM / 连接数约束，
    书稿注释里「并行（10并发）」其实也是限流的，但代码没体现。"""
    sem = asyncio.Semaphore(concurrency)

    async def one(t):
        async with sem:
            return await fake_llm(t)

    return await asyncio.gather(*[one(t) for t in tickets])


async def timed(name: str, coro, expected: float) -> float:
    t0 = time.perf_counter()
    await coro
    dt = time.perf_counter() - t0
    print(f"  {name:14s}: {dt:6.2f}s  理论 {expected:6.2f}s")
    return dt


async def main() -> None:
    tickets = list(range(N))
    print(f"{N} 张工单，单张耗时 {ONE:.2f}s（书稿假设 30s，这里压缩 100 倍便于本地跑）")

    s = await timed("串行", process_serial(tickets), N * ONE)
    p10 = await timed(f"并发({CONCURRENCY})",
                      process_parallel(tickets, CONCURRENCY),
                      (N + CONCURRENCY - 1) // CONCURRENCY * ONE)
    pall = await timed(f"并发(全部{N})", process_parallel(tickets, N), ONE)

    print(f"\n加速比：串行/并发({CONCURRENCY}) = {s/p10:.1f}x，"
          f"串行/并发(全部) = {s/pall:.1f}x")
    print("设计要点（书稿 §10.1.5）：单任务延迟决定体验，整体吞吐量决定处理能力；"
          "对延迟敏感的场景单任务延迟是硬约束，批量场景吞吐量才是核心指标。")


if __name__ == "__main__":
    asyncio.run(main())
