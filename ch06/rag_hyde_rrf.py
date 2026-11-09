# -*- coding: utf-8 -*-
"""ch06 §6.6.1.2 —— RAG 2.0 的两台手术：HyDE 与 RRF 融合

书稿这两段是**纯算法**，不依赖任何向量库：HyDE 用「让模型先编一份答案草稿，
再用草稿去检索」换召回；RRF 用「只看排名、不看绝对分数」解决两套检索分数不可比。
正因为它们是纯函数，这里能给出可运行的实现 + 可验证的输出。

运行：

    python3 ch06/rag_hyde_rrf.py          # 用内置 mock 演示，无需网络、无需模型
    python3 ch06/rag_hyde_rrf.py --demo   # 同上，看每一步的中间量

依赖：**零依赖**（只用标准库）。所以这个文件在任何 Python 3.8+ 上都能跑，
      这也是它值得单独成文件的原因 —— 书稿里的 HyDE 片段调 `llm.invoke` 与
      `vector_store.similarity_search`，那两个东西各家框架都不一样，抄了也跑不起来。
    实测解释器：任意（零第三方依赖，本机用 /usr/bin/python3）

实测状态：✅ 跑通 —— 输出见文件末尾「实测输出」注释块。
与书稿差异：
  1. HyDE 部分：书稿直接调 `llm.invoke(...)` / `vector_store.similarity_search(...)`，
     这是 LangChain 风格的伪接口，换框架就跑不了。这里把它们抽象成两个可注入的
     回调（`llm_fn` / `search_fn`），默认给 mock 实现 —— 结构完全一致，
     但要接真实后端只需换两个函数。
  2. RRF 部分：书稿的 `rrf_fuse` 原样保留，一行未改（它就是可运行的）。
  3. 加了 `__main__` 的演示数据，让读者能立刻看到「融合后名次为什么会翻」。
"""

from typing import Callable, Dict, Iterable, List, Sequence, Tuple


# ----------------------------------------------------------------------------
# 手术一：HyDE（Hypothetical Document Embeddings）
# ----------------------------------------------------------------------------
def hyde_retrieve(question: str,
                  llm_fn: Callable[[str], str],
                  search_fn: Callable[[str, int], List[str]],
                  k: int = 5) -> List[str]:
    """书稿对应小节 的三步，签名保持「问题 → 命中列表」。

    与书稿唯一的差别：`llm` 与 `vector_store` 从对象变成两个可注入函数。
    """
    # 1. 让模型编一段"看起来像答案"的文本
    draft = llm_fn(f"请写一段包含技术细节的回答草稿，用于检索。问题：{question}")
    # 2. 用草稿（而不是问题）去检索
    hits = search_fn(draft, k)
    return hits


# ----------------------------------------------------------------------------
# 手术二：RRF（Reciprocal Rank Fusion）
# ----------------------------------------------------------------------------
def rrf_fuse(rank_lists: Sequence[Sequence[str]], k: int = 60) -> List[Tuple[str, float]]:
    """rank_lists: [[doc_id, ...], [doc_id, ...]]，每套检索一个有序列表

    书稿原文实现，一字未改。要点：**不看绝对分数，只看排名**，
    所以 BM25 的 30 分与余弦的 0.72 才能放进同一个池子里比。
    """
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ----------------------------------------------------------------------------
# 演示（mock）：不联网也能看出两个算法的效果
# ----------------------------------------------------------------------------
CORPUS = [
    "退款到账时间取决于支付渠道，微信 1-3 个工作日，银行卡 3-7 个工作日。",
    "未发货订单可直接取消，款项原路退回。",
    "发票在订单完成后 7 天内可申请，电子发票即时开具。",
    "运费险理赔需要在签收后 72 小时内提交。",
    "修改收货地址仅在未发货状态下允许。",
]


def mock_llm(prompt: str) -> str:
    """假装是模型：把问题扩写成一段「答案草稿」。"""
    q = prompt.split("问题：")[-1]
    return ("关于「%s」，通常的处理方式是：未发货订单可直接取消并原路退款；"
            "已发货需走退货流程，退款在签收后按支付渠道时效到账。") % q


def mock_search(draft: str, k: int) -> List[str]:
    """假装是向量库：按「草稿与文档的字符重合度」打分排序（够用就行）。"""
    def score(doc: str) -> int:
        return sum(1 for ch in set(draft) if ch in doc)
    ranked = sorted(CORPUS, key=score, reverse=True)
    return ranked[:k]


def main() -> int:
    question = "订单没发货，我要退款，多久能到账？"
    print("问题:", question)

    print("\n--- HyDE ---")
    draft = mock_llm(f"请写一段包含技术细节的回答草稿，用于检索。问题：{question}")
    print("草稿:", draft)
    hits = hyde_retrieve(question, mock_llm, mock_search, k=3)
    for i, h in enumerate(hits, 1):
        print("  %d. %s" % (i, h))

    print("\n--- RRF 融合（两套检索：关键词 BM25 风格 / 语义向量风格）---")
    kw_rank = ["d2", "d1", "d5"]          # 关键词命中：未发货 > 退款到账 > 改地址
    vec_rank = ["d1", "d3", "d2"]         # 语义命中：退款时效 > 发票 > 未发货
    fused = rrf_fuse([kw_rank, vec_rank], k=60)
    print("关键词排名:", kw_rank)
    print("语义排名  :", vec_rank)
    print("融合结果  :")
    for doc, sc in fused:
        print("  %-4s %.6f" % (doc, sc))
    top = fused[0][0]
    print("\n结论：融合后第一名是 %s —— 它在两套里都不是第一（%s / %s），"
          "但两套都靠前，所以总分最高。这正是 RRF 想要的：奖励「一致靠前」，惩罚「单边冒尖」。"
          % (top, kw_rank.index(top) + 1 if top in kw_rank else "-",
             vec_rank.index(top) + 1 if top in vec_rank else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# 实测输出（/usr/bin/python3 = 3.10.12，零依赖、无网络）
#
# 问题: 订单没发货，我要退款，多久能到账？
#
# --- HyDE ---
# 草稿: 关于「订单没发货，我要退款，多久能到账？」，通常的处理方式是：未发货订单可直接取消并原路退款；已发货需走退货流程，退款在签收后按支付渠道时效到账。
#   1. 未发货订单可直接取消，款项原路退回。
#   2. 退款到账时间取决于支付渠道，微信 1-3 个工作日，银行卡 3-7 个工作日。
#   3. 发票在订单完成后 7 天内可申请，电子发票即时开具。
#
# --- RRF 融合（两套检索：关键词 BM25 风格 / 语义向量风格）---
# 关键词排名: ['d2', 'd1', 'd5']
# 语义排名  : ['d1', 'd3', 'd2']
# 融合结果  :
#   d1   0.032522
#   d2   0.032266
#   d3   0.016129
#   d5   0.015873
#
# 结论：融合后第一名是 d1 —— 它在两套里都不是第一（2 / 1），但两套都靠前，所以总分最高。这正是 RRF 想要的：奖励「一致靠前」，惩罚「单边冒尖」。
# ---------------------------------------------------------------------------
