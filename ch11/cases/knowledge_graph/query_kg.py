#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：在 kg.json 上做 search / neighbors / path / intent / context / filter 六类查询
# ② 对应书稿：ch11 §11.4.5 第四、五步，（意图映射与 5 个 Tool；对应骨架）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/knowledge_graph/query_kg.py stats
# ④ 依赖：仅标准库（json / collections / argparse）；无需 Neo4j、无 key
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 rc=0；系统 python3 亦可
# ⑥ 与书稿的差异：书稿把图谱写成模块常量 KG；本文件用邻接表 + JSON，并多一个 filter 子命令
# ══════════════════════════════════════════════════════════════════
"""
ch11/cases/knowledge_graph/query_kg.py — 在落盘的图谱上查询

对应书稿：ch11 §11.4.5 第四步「构建意图识别与检索路径映射」
与第五步列出的一组 Tool。

================================================================
书稿列了 5 个 Tool，本文件实现了几个
================================================================
    书稿里的 Tool                    本文件的子命令        说明
    -------------------------------  --------------------  ------------------------------
    search_entity(keyword)           search <关键词>       按名字/类型/文档模糊找节点
    get_relations(entity_id)         neighbors <实体>      某实体的邻居（可按关系/方向过滤）
    traverse_path(start, end)        path <A> <B>          两实体间最短路径（BFS）
    query_by_intent(question)        intent <问句>         第四步的意图映射表，真跑
    get_entity_context(entity_id)    context <实体>        属性 + 上下游 + 证据句
    （书稿没列，但过滤要用）          filter               按关系类型 / 节点类型过滤

================================================================
为什么没有 Neo4j
================================================================
书稿 §11.4.3 反复提到"不管底层用的是 Neo4j 还是你自己写的图引擎"——
它没有指定图数据库。本案例用**邻接表 + JSON**，理由：
  ① 读者不用装服务、不用起 docker、不用申请端口；
  ② 这个规模（几十个节点、几十条边）用内存图是毫秒级，图数据库没优势；
  ③ 想换成 Neo4j 时，`kg.json` 就是现成的导入源（nodes/edges 两个数组）。
差异写进 README，不藏着。

用法
    uv run python ch11/cases/knowledge_graph/query_kg.py stats
    uv run python ch11/cases/knowledge_graph/query_kg.py search 母线
    uv run python ch11/cases/knowledge_graph/query_kg.py neighbors 备自投装置
    uv run python ch11/cases/knowledge_graph/query_kg.py neighbors 备自投装置 --rel 提供 --dir out
    uv run python ch11/cases/knowledge_graph/query_kg.py path 主变跳闸 有限转供能力
    uv run python ch11/cases/knowledge_graph/query_kg.py filter --rel 触发
    uv run python ch11/cases/knowledge_graph/query_kg.py filter --node-type 部件
    uv run python ch11/cases/knowledge_graph/query_kg.py intent "主变跳闸了怎么办"
    uv run python ch11/cases/knowledge_graph/query_kg.py context 备自投装置
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_KG = HERE / "outputs" / "kg.json"


# ---------------------------------------------------------------------------
# 图加载
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self, data: dict):
        self.meta = data["meta"]
        self.nodes: dict[str, dict] = {n["id"]: n for n in data["nodes"]}
        self.edges: list[dict] = data["edges"]
        self.out: dict[str, list[dict]] = defaultdict(list)
        self.inc: dict[str, list[dict]] = defaultdict(list)
        for e in self.edges:
            self.out[e["from"]].append(e)
            self.inc[e["to"]].append(e)

    # --- 查询 ---
    def search(self, kw: str) -> list[dict]:
        kw = kw.lower()
        hits = []
        for n in self.nodes.values():
            hay = f"{n['name']} {n['type']} {' '.join(n.get('docs', []))}".lower()
            if kw in hay:
                hits.append(n)
        return sorted(hits, key=lambda n: -n["freq"])

    def neighbors(self, name: str, rel: str = "", direction: str = "both") -> list[dict]:
        rows = []
        if direction in ("both", "out"):
            rows += [{**e, "_dir": "out"} for e in self.out.get(name, []) if not rel or e["type"] == rel]
        if direction in ("both", "in"):
            rows += [{**e, "_dir": "in"} for e in self.inc.get(name, []) if not rel or e["type"] == rel]
        return rows

    def shortest_path(self, a: str, b: str, max_depth: int = 6) -> list[dict] | None:
        """BFS，把边当无向（找的是"两者有没有关系"，不纠结方向）。"""
        if a not in self.nodes or b not in self.nodes:
            return None
        prev: dict[str, tuple[str, dict] | None] = {a: None}
        q = deque([(a, 0)])
        while q:
            cur, d = q.popleft()
            if cur == b:
                break
            if d >= max_depth:
                continue
            for e in self.out.get(cur, []) + self.inc.get(cur, []):
                nxt = e["to"] if e["from"] == cur else e["from"]
                if nxt not in prev:
                    prev[nxt] = (cur, e)
                    q.append((nxt, d + 1))
        if b not in prev:
            return None
        steps = []
        cur = b
        while prev[cur] is not None:
            parent, edge = prev[cur]
            steps.append({**edge, "_from_side": parent})
            cur = parent
        return list(reversed(steps))

    def filter_edges(self, rel: str = "", confidence: str = "", node_type: str = "") -> list[dict]:
        rows = self.edges
        if rel:
            rows = [e for e in rows if e["type"] == rel]
        if confidence:
            rows = [e for e in rows if e["confidence"] == confidence]
        if node_type:
            keep = {n["id"] for n in self.nodes.values() if n["type"] == node_type}
            rows = [e for e in rows if e["from"] in keep or e["to"] in keep]
        return rows


# ---------------------------------------------------------------------------
# 第四步：意图映射表（书里那张 JSON 的可执行版）
# ---------------------------------------------------------------------------
# 书稿的 graph_query 写的是伪路径（"entity_by_keyword → relations(type='触发') → ..."），
# 本文件把它落成真能执行的函数。
INTENT_MAPPINGS = [
    {
        "intent": "故障排查",
        "patterns": ["怎么办", "出问题", "失效", "故障", "异常", "处置"],
        "plan": "关键词→实体 → 出边(触发/导致) → 目标实体 → 入边(涉及/提供)",
        "handler": "troubleshoot",
    },
    {
        "intent": "原理讲解",
        "patterns": ["原理", "为什么", "工作机制", "如何工作", "是什么"],
        "plan": "关键词→实体 → 双向关系 → 上下游实体",
        "handler": "explain",
    },
    {
        "intent": "数据查询",
        "patterns": ["参数", "数值", "多少", "限制", "温度", "压力", "速度"],
        "plan": "关键词→实体 → 属性/参数类节点",
        "handler": "lookup",
    },
]


def find_entities_in_question(g: Graph, question: str) -> list[dict]:
    """从问句里找实体：按名字长度降序逐个试，长名优先（避免'母线'抢'母线电压'）。"""
    hits = []
    for name in sorted(g.nodes, key=len, reverse=True):
        if name in question:
            hits.append(g.nodes[name])
    return hits


def handler_troubleshoot(g: Graph, ents: list[dict]) -> dict:
    """故障排查：从故障实体出发，先走 触发/导致 出边，再看目标实体的 涉及/提供 关系。"""
    lines, chain = [], []
    for e in ents:
        lines.append(f"起点：{e['name']}（{e['type']}）")
        cause_edges = [x for x in g.neighbors(e["name"], direction="out") if x["type"] in ("触发", "导致", "影响")]
        if not cause_edges:
            lines.append("  （这个实体没有触发/导致类的出边，试试换个实体名）")
        for x in cause_edges:
            chain.append(f"{e['name']} --[{x['type']}]--> {x['to']}")
            lines.append(f"  → {x['type']}：{x['to']}（{g.nodes.get(x['to'], {}).get('type', '?')}，{x['confidence']}）")
            nxt = [y for y in g.neighbors(x["to"], direction="out") if y["type"] in ("涉及", "提供", "包含", "依赖")]
            for y in nxt:
                chain.append(f"{x['to']} --[{y['type']}]--> {y['to']}")
                lines.append(f"      · {y['type']}：{y['to']}")
                lines.append(f"        依据：{y['evidence'][:70]}")
        # 入边当"被谁触发"
        for x in g.neighbors(e["name"], direction="in"):
            lines.append(f"  ← 被 {x['from']} {x['type']}（{x['confidence']}）")
    return {"intent": "故障排查", "entities": [e["name"] for e in ents],
            "path": chain, "answer": "\n".join(lines) or "未找到相关实体"}


def handler_explain(g: Graph, ents: list[dict]) -> dict:
    lines, chain = [], []
    for e in ents:
        lines.append(f"{e['name']}（{e['type']}，出现 {e['freq']} 次）")
        for x in g.neighbors(e["name"]):
            arrow = "-->" if x["_dir"] == "out" else "<--"
            other = x["to"] if x["_dir"] == "out" else x["from"]
            lines.append(f"  {arrow} [{x['type']}] {other}")
            chain.append(f"{e['name']} {arrow} [{x['type']}] {other}")
    return {"intent": "原理讲解", "entities": [e["name"] for e in ents],
            "path": chain, "answer": "\n".join(lines) or "未找到相关实体"}


def handler_lookup(g: Graph, ents: list[dict]) -> dict:
    lines, chain = [], []
    for e in ents:
        lines.append(f"{e['name']}（{e['type']}）")
        params = [x for x in g.neighbors(e["name"]) 
                  if g.nodes.get(x["to"] if x["_dir"] == "out" else x["from"], {}).get("type") == "参数"]
        if not params:
            lines.append("  未关联参数类节点")
        for x in params:
            other = x["to"] if x["_dir"] == "out" else x["from"]
            lines.append(f"  参数：{other}（依据：{x['evidence'][:60]}）")
            chain.append(f"{e['name']} -> {other}")
    return {"intent": "数据查询", "entities": [e["name"] for e in ents],
            "path": chain, "answer": "\n".join(lines) or "未找到相关实体"}


HANDLERS = {"troubleshoot": handler_troubleshoot, "explain": handler_explain, "lookup": handler_lookup}


def dispatch_intent(g: Graph, question: str) -> dict:
    """先判意图，再按映射表执行 —— 书稿第四步那张表的核心。"""
    matched = None
    for m in INTENT_MAPPINGS:
        if any(p in question for p in m["patterns"]):
            matched = m
            break
    if matched is None:
        matched = INTENT_MAPPINGS[1]  # 兜底走"原理讲解"
    ents = find_entities_in_question(g, question)
    if not ents:
        return {"intent": matched["intent"], "entities": [], "path": [],
                "plan": matched["plan"], "answer": "问句里没有识别到图谱中的实体"}
    out = HANDLERS[matched["handler"]](g, ents)
    out["plan"] = matched["plan"]
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def show_node(n: dict) -> str:
    return f"{n['name']}（{n['type']}，{n['source']}，×{n['freq']}）"


def main() -> int:
    ap = argparse.ArgumentParser(description="知识图谱查询")
    ap.add_argument("--kg", default=str(DEFAULT_KG), help="kg.json 路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="图谱规模")
    p = sub.add_parser("search", help="按关键词找节点")
    p.add_argument("keyword")
    p = sub.add_parser("neighbors", help="某实体的邻居")
    p.add_argument("entity")
    p.add_argument("--rel", default="", help="只看某种关系类型")
    p.add_argument("--dir", default="both", choices=["both", "in", "out"])
    p = sub.add_parser("path", help="两实体间最短路径")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--max-depth", type=int, default=6)
    p = sub.add_parser("filter", help="按关系类型/节点类型过滤")
    p.add_argument("--rel", default="")
    p.add_argument("--confidence", default="", choices=["", "high", "medium", "low"])
    p.add_argument("--node-type", default="")
    sub.add_parser("intent", help="端到端：意图识别 → 图谱查询").add_argument("question")
    sub.add_parser("context", help="实体完整上下文").add_argument("entity")

    args = ap.parse_args()
    kg_path = Path(args.kg)
    if not kg_path.exists():
        print(f"❌ 找不到 {kg_path}\n   先跑: uv run python ch11/cases/knowledge_graph/build_kg.py")
        return 1
    g = Graph(json.loads(kg_path.read_text(encoding="utf-8")))

    if args.cmd == "stats":
        print(f"节点 {len(g.nodes)} 个 / 边 {len(g.edges)} 条")
        print(f"语料：{', '.join(g.meta['corpus'])}（{g.meta['chars']} 字符）")
        print("节点类型：", dict(Counter(n["type"] for n in g.nodes.values())))
        print("节点来源：", dict(Counter(n["source"] for n in g.nodes.values())))
        print("关系类型：", dict(Counter(e["type"] for e in g.edges)))
        print("置信度：  ", dict(Counter(e["confidence"] for e in g.edges)))
        top = sorted(g.nodes.values(), key=lambda n: (-n["freq"], n["name"]))[:10]
        print("高频节点：", ", ".join(f"{n['name']}×{n['freq']}" for n in top))

    elif args.cmd == "search":
        hits = g.search(args.keyword)
        print(f"关键词「{args.keyword}」命中 {len(hits)} 个节点：")
        for n in hits:
            print("  " + show_node(n))

    elif args.cmd == "neighbors":
        if args.entity not in g.nodes:
            print(f"❌ 图里没有「{args.entity}」。相近的：{[n['name'] for n in g.search(args.entity)[:5]]}")
            return 1
        rows = g.neighbors(args.entity, args.rel, args.dir)
        print(f"「{args.entity}」的邻居 {len(rows)} 条"
              f"{'（关系=' + args.rel + '）' if args.rel else ''}{'（方向=' + args.dir + '）' if args.dir != 'both' else ''}：")
        for x in rows:
            if x["_dir"] == "out":
                print(f"  --> [{x['type']}/{x['confidence']}] {x['to']}   依据: {x['evidence'][:50]}")
            else:
                print(f"  <-- [{x['type']}/{x['confidence']}] {x['from']}   依据: {x['evidence'][:50]}")

    elif args.cmd == "path":
        steps = g.shortest_path(args.a, args.b, args.max_depth)
        if steps is None:
            print(f"❌ 「{args.a}」到「{args.b}」在 {args.max_depth} 步内没有通路")
            return 1
        print(f"「{args.a}」→「{args.b}」最短路径 {len(steps)} 步：")
        cur = args.a
        for i, e in enumerate(steps, 1):
            if e["from"] == cur:
                print(f"  {i}. {cur} --[{e['type']}/{e['confidence']}]--> {e['to']}")
                cur = e["to"]
            else:
                print(f"  {i}. {cur} <--[{e['type']}/{e['confidence']}]-- {e['from']}")
                cur = e["from"]
        print(f"  终点: {cur}（{g.nodes.get(cur, {}).get('type', '?')}）")

    elif args.cmd == "filter":
        rows = g.filter_edges(args.rel, args.confidence, args.node_type)
        cond = ", ".join(x for x in [
            f"关系={args.rel}" if args.rel else "",
            f"置信度={args.confidence}" if args.confidence else "",
            f"节点类型={args.node_type}" if args.node_type else "",
        ] if x) or "无过滤（全部边）"
        print(f"过滤条件：{cond} → {len(rows)} 条边")
        for e in rows[:40]:
            print(f"  {e['from']} --[{e['type']}/{e['confidence']}]--> {e['to']}")

    elif args.cmd == "intent":
        out = dispatch_intent(g, args.question)
        print(f"问句：{args.question}")
        print(f"识别意图：{out['intent']}")
        print(f"命中实体：{out['entities']}")
        print(f"查询路径（书稿第四步的 graph_query 落成可执行版）：{out['plan']}")
        print("--- 结果 ---")
        print(out["answer"])

    elif args.cmd == "context":
        name = args.entity
        if name not in g.nodes:
            print(f"❌ 图里没有「{name}」。相近的：{[n['name'] for n in g.search(name)[:5]]}")
            return 1
        n = g.nodes[name]
        print(f"=== {n['name']} ===")
        print(f"类型: {n['type']}    来源: {n['source']}（pos={n['pos_tag']}）   出现: {n['freq']} 次")
        print(f"文档: {', '.join(n['docs'])}")
        out_edges, in_edges = g.out.get(name, []), g.inc.get(name, [])
        print(f"\n出边 {len(out_edges)} 条：")
        for e in out_edges:
            print(f"  --> [{e['type']}/{e['confidence']}] {e['to']}\n      依据: {e['evidence'][:80]}")
        print(f"\n入边 {len(in_edges)} 条：")
        for e in in_edges:
            print(f"  <-- [{e['type']}/{e['confidence']}] {e['from']}\n      依据: {e['evidence'][:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
