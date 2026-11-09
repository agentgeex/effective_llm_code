#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：从中文语料抽取实体/关系，建成知识图谱并落盘 kg.json
# ② 对应章节：第 11 章 §11.4「如何构建大模型知识图谱」
# ③ 怎么跑：python ch11/cases/knowledge_graph/build_kg.py
# ④ 依赖：jieba。无需数据库、无需模型权重、不连外部服务
# ⑤ 实测：Python 3.10.12 + jieba，三篇自带语料跑通（节点/边数见运行输出）
# ══════════════════════════════════════════════════════════════════
"""
从中文文本抽实体与关系，建图落盘。

对应章节：第 11 章 §11.4「如何构建大模型知识图谱」。

================================================================
这套代码在做什么
================================================================
书里讲知识建模时给的是**手填的实体/关系 JSON** —— 因为是讲"怎么建模"，
不是讲"怎么抽取"。手填 4 个实体没问题，但读者面对自己那本 500 页手册时，
手填 4000 个是不现实的。所以这里补上"抽取"这一步：

  ① 领域词表命中（LEXICON）
     只定义"词 → 类型"，不定义"图里有哪些节点"。词表里有、正文没出现的词不进图。

  ② jieba 词性开放抽取
     取名词(n) / 专名(nz) / 英文(eng)，长度 ≥ 2，去停用词。
     这一路负责捞出词表没覆盖的词，才算真的"抽"而不是"查表"。

两路结果都写进节点的 `source` 字段（lexicon / pos / lexicon+pos），
每个节点是怎么来的，一眼可见。

  ③ 句内共现 + 触发词定向
     同一句出现 ≥2 个实体才连边；关系类型由触发词决定
     （导致/引发 → 导致；触发/启动 → 触发；提供/供给 → 提供 …）。
     两个实体间没有任何触发词 → 记一条 `共现` 边，confidence=low。
     每条边带 `evidence`（原文那一句）与 `confidence`。

`GOLDEN` 里放着一份"人工标注答案"，build 完自动比对几条关键关系能否被抽出来，
结论打到日志里。它不参与构建，只用来自检。

================================================================

用法
    # 默认语料（自带 3 篇短文）
    python ch11/cases/knowledge_graph/build_kg.py

    # 换自己的语料（目录或单个文件都行）
    python ch11/cases/knowledge_graph/build_kg.py --corpus /path/to/docs

    # 调抽取门槛
    python ch11/cases/knowledge_graph/build_kg.py --min-freq 2 --min-len 3

依赖：jieba。不连数据库、不起服务。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba.posseg as pseg

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "sample_corpus"


# ---------------------------------------------------------------------------
# ① 领域词表：只定义"词 → 类型"，不定义"图里有哪些节点"
# ---------------------------------------------------------------------------
# 注意：这张表是"词典"，不是"节点清单"。词表里有但正文没出现的词，不会进图。
LEXICON: dict[str, str] = {
    # 部件
    "备自投装置": "部件", "BZT": "部件", "主变": "部件", "备用变": "部件",
    "断路器": "部件", "母联开关": "部件", "隔离开关": "部件", "进线开关": "部件",
    "母线": "部件", "保护装置": "部件", "电流互感器": "部件", "电压互感器": "部件",
    "站用变": "部件", "蓄电池": "部件", "直流屏": "部件", "合闸线圈": "部件",
    "辅助触点": "部件", "通信电源": "部件",
    # 系统
    "站用电系统": "系统", "继电保护": "系统", "直流系统": "系统",
    "监控系统": "系统", "备自投系统": "系统", "事故照明": "系统",
    # 程序 / 文档
    "主变跳闸事故处理规程": "程序", "SOP-3.2": "程序", "事故处理规程": "程序",
    "限电序位表": "程序", "事故报告": "程序", "巡视记录": "程序",
    # 故障 / 状态
    "主变差动保护动作": "故障", "87T": "故障", "主变跳闸": "故障",
    "母线失压": "故障", "备自投拒动": "故障", "开关拒分": "故障",
    "合闸线圈烧损": "故障", "定值整定错误": "故障",
    "有限转供能力": "状态", "转供容量受限": "状态", "单电源运行": "状态",
    "全站失压": "状态", "手动转供": "状态",
    # 参数 / 量
    "定值": "参数", "整定值": "参数", "电流": "参数", "电压": "参数",
    "母线电压": "参数", "零序电流": "参数", "负荷": "参数", "容量": "参数",
    "时限": "参数", "载流能力": "参数", "电池电量": "参数", "遥信": "参数",
    # 人 / 角色
    "值班员": "角色", "正值": "角色", "副值": "角色", "调度": "角色", "调度员": "角色",
    # 其他
    "故障录波": "概念", "重合闸": "概念", "遥测": "概念", "五防": "概念",
    "动作状态": "概念", "保护动作信号": "概念", "运行方式": "概念",
}

# 长词优先匹配，避免"备自投"先命中"备自投装置"里的三个字
_LEX_SORTED = sorted(LEXICON, key=len, reverse=True)

# 只取名词类。曾经把 vn（动名词）也算进来，结果"驱动/供电/决策/监控"这些动词
# 全被当成实体入了图 —— 动词不是"名词表"里的东西，去掉。
ENTITY_POS = {"n", "nz", "eng", "nr", "ns", "nt"}
STOPWORDS = {
    "变电站", "状态", "情况", "水平", "当时", "部分", "全部", "同时", "第一步",
    "最后", "三个", "两套", "任意", "对应", "内部", "逻辑", "编号", "位置",
    "指令", "反馈", "输出", "功率", "时间", "期间", "范围", "温度", "速度",
    "基本", "前部", "核心", "中心", "方式", "过程", "问题", "答案", "下一步",
    "两类", "三类", "任何一个", "剩余", "下一步操作", "动作",
}

# ---------------------------------------------------------------------------
# ③ 触发词 → 关系类型
# ---------------------------------------------------------------------------
RELATION_TRIGGERS: list[tuple[str, str]] = [
    ("导致", "导致"), ("引发", "导致"), ("造成", "导致"),
    ("触发", "触发"), ("启动", "触发"),
    ("提供", "提供"), ("供给", "提供"), ("产生", "提供"),
    ("属于", "属于"), ("分为", "包含"), ("包括", "包含"), ("组成", "包含"),
    ("依赖", "依赖"), ("依靠", "依赖"), ("需要", "依赖"),
    ("涉及", "涉及"),
    ("监控", "监控"), ("检查", "监控"),
    ("影响", "影响"), ("抵消", "影响"), ("缓解", "影响"),
    ("安装在", "位于"), ("位于", "位于"),
    ("控制", "控制"), ("驱动", "控制"),
]

# 分句：除了句末标点，逗号也断 —— 中文长句里"站用电由一号站用变供电，直流负荷由蓄电池组供电"
# 这种并列结构，不断开就会把两个从句的实体连成一条假边。
SENT_END = "。！？；，、!?;,\n"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def guess_type(name: str) -> str:
    """给词表没覆盖的词按后缀猜一个类型（猜不出就'概念'）。"""
    rules = [
        (("系统",), "系统"),
        (("规程", "程序", "步骤", "手册", "序位表"), "程序"),
        (("动作", "跳闸", "失压", "拒动", "拒分", "烧损", "错误"), "故障"),
        (("定值", "电流", "电压", "负荷", "容量", "时限", "能力"), "参数"),
        (("值班员", "正值", "副值", "调度"), "角色"),
        (("装置", "开关", "变", "器", "线", "触点", "母"), "部件"),
    ]
    for suffixes, label in rules:
        if any(name.endswith(s) for s in suffixes):
            return label
    return "概念"


def split_sentences(text: str) -> list[str]:
    buf, out = [], []
    for ch in text:
        if ch in SENT_END:
            if buf:
                out.append("".join(buf).strip())
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [s for s in out if len(s) >= 4]


def extract_entities(sentence: str, min_len: int) -> list[dict]:
    """抽一句里的实体，返回 [{name, type, source, pos}]，按出现位置排序。"""
    found: dict[str, dict] = {}
    covered = [False] * len(sentence)

    # ① 词表命中（长词优先）
    for term in _LEX_SORTED:
        if len(term) < min_len:
            continue
        start = 0
        while True:
            idx = sentence.find(term, start)
            if idx < 0:
                break
            # 已被更长的词占用的位置跳过
            if any(covered[idx:idx + len(term)]):
                start = idx + 1
                continue
            for k in range(idx, idx + len(term)):
                covered[k] = True
            found.setdefault(term, {
                "name": term, "type": LEXICON[term], "source": "lexicon",
                "pos_tag": "-", "pos": idx,
            })
            start = idx + len(term)

    # ② jieba 词性开放抽取（跳过词表已占用的字符）
    pos = 0
    for word, flag in pseg.cut(sentence):
        width = len(word)
        span = range(pos, pos + width)
        pos += width
        if not word.strip() or width < min_len:
            continue
        if sentence[span.start:span.stop] != word:
            # jieba 的输出与原文对不上（极少见），放弃位置判定，仍接受该词
            span_covered = False
        else:
            span_covered = all(covered[i] for i in span if i < len(covered))
        if span_covered:
            continue
        if word in STOPWORDS or flag not in ENTITY_POS:
            continue
        if word.isdigit():
            continue
        entry = found.get(word)
        if entry:
            entry["source"] = "lexicon+pos"
            continue
        found[word] = {
            "name": word,
            "type": LEXICON.get(word, guess_type(word)),
            "source": "pos",
            "pos_tag": flag,
            "pos": sentence.find(word),
        }

    # 按在句中的出现位置排序 —— 不排就只能按词表长度顺序，配对会乱
    return sorted(found.values(), key=lambda e: e["pos"])


def find_trigger(sentence: str) -> tuple[str, str] | None:
    """返回 (触发词, 关系类型)；没有返回 None。多个时取最长触发词。"""
    hits = [(w, t) for w, t in RELATION_TRIGGERS if w in sentence]
    if not hits:
        return None
    hits.sort(key=lambda x: len(x[0]), reverse=True)
    return hits[0]


def build_graph(corpus_files: list[Path], min_len: int, min_freq: int) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    docs_meta = []
    cooc = Counter()

    for path in corpus_files:
        text = path.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        docs_meta.append({"file": path.name, "chars": len(text), "sentences": len(sentences)})

        for sent in sentences:
            ents = extract_entities(sent, min_len)
            if not ents:
                continue
            for e in ents:
                cooc[e["name"]] += 1
                node = nodes.get(e["name"])
                if node is None:
                    nodes[e["name"]] = {
                        "id": e["name"],
                        "name": e["name"],
                        "type": e["type"],
                        "source": e["source"],
                        "pos_tag": e["pos_tag"],
                        "freq": 1,
                        "docs": [path.name],
                    }
                else:
                    node["freq"] += 1
                    if path.name not in node["docs"]:
                        node["docs"].append(path.name)

            if len(ents) < 2:
                continue

            trig = find_trigger(sent)
            if trig:
                trigger_word, rel_type = trig
                # ents 已按句中位置排序：触发词前最近一个当 from，后最近一个当 to
                t_idx = sent.find(trigger_word)
                before = [e for e in ents if e["pos"] < t_idx]
                after = [e for e in ents if e["pos"] >= t_idx]
                if before and after:
                    pairs = [(before[-1]["name"], after[0]["name"])]
                    gap = sent[t_idx + len(trigger_word): after[0]["pos"]]
                    # "偏航控制**的**单发失效" 这类定语结构不是主谓宾，降置信度
                    confidence = "medium" if "的" in gap else "high"
                else:
                    pairs = [(ents[i]["name"], ents[i + 1]["name"]) for i in range(len(ents) - 1)]
                    confidence = "medium"
            else:
                rel_type = "共现"
                pairs = [(ents[i]["name"], ents[i + 1]["name"]) for i in range(len(ents) - 1)]
                confidence = "low"

            for f, t in pairs:
                if f == t:
                    continue
                key = (f, t, rel_type)
                edge = edges.get(key)
                if edge is None:
                    edges[key] = {
                        "from": f,
                        "to": t,
                        "type": rel_type,
                        "confidence": confidence,
                        "trigger": trigger_word if trig else "",
                        "evidence": sent[:120],
                        "count": 1,
                        "doc": path.name,
                    }
                else:
                    edge["count"] += 1

    # 频次过滤（只影响低信噪比的 POS 抽词，词表命中一律保留）
    if min_freq > 1:
        keep = {n for n, c in cooc.items() if c >= min_freq or nodes[n]["source"].startswith("lexicon")}
        nodes = {n: v for n, v in nodes.items() if n in keep}
        edges = {k: v for k, v in edges.items() if k[0] in nodes and k[1] in nodes}

    return {
        "meta": {
            "corpus": [d["file"] for d in docs_meta],
            "docs": docs_meta,
            "chars": sum(d["chars"] for d in docs_meta),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "min_len": min_len,
            "min_freq": min_freq,
        },
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


# 人工标注的对照答案：构建完自动比对这几条关系能否被抽出来。
# 它不参与建图，只用来自检抽取质量。
GOLDEN = {
    "entities": ["主变差动保护动作", "备自投装置", "主变跳闸事故处理规程", "有限转供能力"],
    "relations": [
        ("主变差动保护动作", "主变跳闸事故处理规程", "触发"),
        ("主变跳闸事故处理规程", "备自投装置", "涉及"),
        ("备自投装置", "有限转供能力", "提供"),
        ("主变跳闸", "备自投装置", "触发"),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="从中文文本抽实体与关系，建知识图谱")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="语料目录或单个 .md/.txt 文件")
    ap.add_argument("--out", default=str(HERE / "outputs" / "kg.json"), help="图谱落盘路径")
    ap.add_argument("--min-len", type=int, default=2, help="实体最小字数（默认 2）")
    ap.add_argument("--min-freq", type=int, default=2, help="POS 抽词的词频门槛（默认 2；词表命中不受限）")
    ap.add_argument("--top", type=int, default=10, help="打印前 N 条边（默认 10）")
    args = ap.parse_args()

    src = Path(args.corpus)
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix in {".md", ".txt"})
    elif src.is_file():
        files = [src]
    else:
        print(f"❌ 语料不存在: {src}")
        return 1
    if not files:
        print(f"❌ {src} 下没有 .md / .txt 文件")
        return 1

    graph = build_graph(files, args.min_len, args.min_freq)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 打印可核对的数字 ----------------
    meta = graph["meta"]
    print("=" * 62)
    print("知识图谱构建结果")
    print("=" * 62)
    print(f"语料: {', '.join(meta['corpus'])}")
    print(f"字符数: {meta['chars']}   句子数: {sum(d['sentences'] for d in meta['docs'])}")
    print(f"**节点 {meta['node_count']} 个 / 边 {meta['edge_count']} 条**")
    print(f"落盘: {out}  ({out.stat().st_size} 字节)")

    print("\n-- 节点按类型 --")
    for t, c in Counter(n["type"] for n in graph["nodes"]).most_common():
        print(f"   {t}: {c}")
    print("-- 节点按来源 --")
    for s, c in Counter(n["source"] for n in graph["nodes"]).most_common():
        print(f"   {s}: {c}")
    print("-- 边按类型 --")
    for t, c in Counter(e["type"] for e in graph["edges"]).most_common():
        print(f"   {t}: {c}")

    print(f"\n-- 前 {args.top} 个高频节点 --")
    for n in sorted(graph["nodes"], key=lambda x: (-x["freq"], x["name"]))[: args.top]:
        print(f"   {n['name']}({n['type']}/{n['source']}) ×{n['freq']}")

    print(f"\n-- 前 {args.top} 条边（按出现次数）--")
    top_edges = sorted(graph["edges"], key=lambda e: (-e["count"], e["from"]))[: args.top]
    for e in top_edges:
        print(f"   {e['from']} --[{e['type']}/{e['confidence']}]--> {e['to']}   ×{e['count']}")

    # ---------------- 与人工标注答案对照 ----------------
    names = {n["name"] for n in graph["nodes"]}
    print("\n-- 与人工标注答案对照（标注是手写的，本文件是抽取的）--")
    hit = [n for n in GOLDEN["entities"] if n in names]
    miss = [n for n in GOLDEN["entities"] if n not in names]
    print(f"   标注实体 {len(GOLDEN['entities'])} 个，抽到 {len(hit)} 个: {hit}")
    if miss:
        print(f"   没抽到: {miss}")
    triples = {(e["from"], e["to"], e["type"]) for e in graph["edges"]}
    for f, to, r in GOLDEN["relations"]:
        mark = "✅" if (f, to, r) in triples else ("△" if any(x[0] == f and x[1] == to for x in triples) else "❌")
        print(f"   {mark} {f} --[{r}]--> {to}")

    print("\n提示：`--min-freq 2` 可以砍掉只出现一次的长尾词；`--corpus` 换成你自己的文档试试。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
