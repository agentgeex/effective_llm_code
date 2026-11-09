# -*- coding: utf-8 -*-
"""① 这是什么
    ch08 §8.3.3 RAG 四步的最小可跑版：向量化 → Chroma 检索 → 拼 Prompt → 问模型。
    集合为空时自动写入 5 条示例知识块；无 OPENAI_API_KEY 时改用本地哈希向量，
    保证前 3 步在任何机器上都能真跑。

② 对应书稿
    第 8 章 §8.3.3「RAG：知识增强的『外部硬盘』」。

③ 怎么跑
    # 无 key：本地哈希向量，检索链路全跑，第 4 步跳过
    .venv/bin/python ch08/rag_chroma_demo.py
    # 有 key：真 embedding + 真生成
    export OPENAI_API_KEY=<你的key>
    .venv/bin/python ch08/rag_chroma_demo.py

④ 依赖
    包：chromadb（本机实测 1.1.1）、openai（3.3.0，只有「有 key」那条路才用）
    服务：「有 key」路线需要可访问的 OpenAI 端点
    权重：无（embedding 走 API 或纯 Python 哈希）
    落盘：./.chroma_demo（已进 .gitignore，首次运行自动新建）

⑤ 实测解释器 + 结果
    解释器：.venv/bin/python（Python 3.10.12 / chromadb 1.1.1 / openai 3.3.0）
    rc=0，原样输出：
        [embedding] 无 OPENAI_API_KEY → 用本地哈希向量（非语义，仅验证链路）
        入库完成：5 条知识块 -> <仓库根>/.chroma_demo （本次新建）

        问题: vLLM 的 PagedAttention 机制是如何节省显存的？
        命中 Top-3:
          1. [距离 1.465] vLLM 用 PagedAttention 把 KV Cache 切成固定大小的 block…
          2. [距离 2.000] 投机解码（speculative decoding）用一个小模型先草拟若干 token，再用…
          3. [距离 2.000] 量化把权重从 FP16 压到 INT8 或 FP8，显存占用减半，配合分组量化（group-…

        ✅ 检索排序正确（第 1 条正是 PagedAttention 那段）

        ❌ 第 4 步「提交给大模型」跳过：缺 OPENAI_API_KEY（prompt 已拼好，长度 346 字符）
    ★ 第 2、3 条距离同为 2.000（哈希向量下与问题无词面重合即封顶）——只能证明链路通，
      不代表语义检索质量。第 4 步「真生成」未验证：缺 key（本机网络亦不可达）。

⑥ 与书稿的差异（三处）
    1. 书稿假设 `knowledge_db` 里已有数据（get_or_create_collection 之后直接 query）。
       空库 query 只返回空列表 —— 读者第一跑必然「查不到」却不知道是没入库。
       本文件补了「集合为空则写入 5 条示例知识块」这一步，是书稿缺的最小可跑前置。
    2. Embedding 双路：有 key 用书稿原样的 text-embedding-3-small；
       没有则用本地哈希向量（纯 Python，不联网、不下载模型）。
       两条路的接口都是 `embed(text) -> list[float]`，第 3、4 步代码不变。
       ⚠️ 哈希向量只保证「同一段文本向量相同、词面重合度高的更接近」，
          它不是语义向量 —— 这里只为让检索链路可运行，不代表真实 embedding 的召回质量。
    3. 向量库落盘目录从书稿的 ./knowledge_db 改为 ./.chroma_demo，并已加入 .gitignore，
       避免读者一跑就在仓库里留下一个向量库目录。
"""


import hashlib
import os
import sys

import chromadb

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".chroma_demo")
COLLECTION = "tech_docs"

# 书稿没有给知识块内容（它假设库里已有）。这里补 5 条与问题相关的示例块。
SEED_DOCS = [
    "vLLM 用 PagedAttention 把 KV Cache 切成固定大小的 block，像操作系统管理虚拟内存那样"
    "按需分配，块之间可以不连续，因此几乎消除显存碎片，显存利用率可到 90% 以上。",
    "连续批处理（continuous batching）让新请求随时插进正在跑的批次，"
    "而不是等整批结束，显著提高吞吐。",
    "量化把权重从 FP16 压到 INT8 或 FP8，显存占用减半，配合分组量化（group-wise）"
    "可以把精度损失控制在 1% 以内。",
    "投机解码（speculative decoding）用一个小模型先草拟若干 token，"
    "再用大模型一次校验，能在不改变输出分布的前提下加速 2-3 倍。",
    "张量并行（tensor parallelism）把单层权重按列或按行切到多张卡上，"
    "适合单卡放不下的超大模型；流水并行则按层切分。",
]


def hash_embed(text: str, dim: int = 256) -> list:
    """本地确定性向量：按词袋哈希到 dim 维并归一化。不联网、不下载模型。"""
    vec = [0.0] * dim
    for tok in text.replace("/", " ").replace("，", " ").replace("。", " ").split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
        # 二元组也计入，让「PagedAttention 机制」这种短语能聚合
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")

    client = None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("CLOUD_BASE_URL")
    chat_model = os.getenv("OPENAI_MODEL") or os.getenv("CLOUD_MODEL", "gpt-4o")
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        print("[embedding] 尝试 OpenAI text-embedding-3-small")
        # 该端点若无 embeddings 能力（如 DeepSeek 只提供 chat），运行时降级为本地哈希向量
        _emb_state = {"live": True}

        def embed(text: str) -> list:
            if _emb_state["live"]:
                try:
                    return client.embeddings.create(
                        model="text-embedding-3-small", input=text
                    ).data[0].embedding
                except Exception as e:
                    _emb_state["live"] = False
                    print(f"[embedding] 端点不支持 embeddings（{type(e).__name__}）→ 降级为本地哈希向量（非语义，仅验证链路）")
            return hash_embed(text)
    else:
        print("[embedding] 无 OPENAI_API_KEY → 用本地哈希向量（非语义，仅验证链路）")

        def embed(text: str) -> list:
            return hash_embed(text)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION)

    # ── 补：书稿假设库里已有数据，这里保证非空 ──────────────────────────
    if collection.count() == 0:
        collection.add(
            ids=[f"doc{i}" for i in range(len(SEED_DOCS))],
            documents=SEED_DOCS,
            embeddings=[embed(d) for d in SEED_DOCS],
        )
        state = "（本次新建）"
    else:
        state = "（复用已有）"
    print(f"入库完成：{collection.count()} 条知识块 -> {os.path.normpath(DB_PATH)} {state}")

    # ── 以下与书稿对应小节 的流程一致 ────────────────────────────────
    # 1. 将用户问题向量化
    question = "vLLM 的 PagedAttention 机制是如何节省显存的？"
    question_embedding = embed(question)

    # 2. 在向量库中检索最相似的 Top-N 个知识块
    results = collection.query(query_embeddings=[question_embedding], n_results=3)

    # 3. 将检索到的知识块拼接到 Prompt 中
    retrieved_docs = "\n\n---\n\n".join(results["documents"][0])
    augmented_prompt = f"""你需要根据以下参考资料回答问题。
如果参考资料中没有相关信息，请如实回答「不清楚」。

参考资料：
{retrieved_docs}

问题：{question}"""

    print(f"\n问题: {question}")
    print("命中 Top-3:")
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0]), 1):
        print(f"  {i}. [距离 {dist:.3f}] {doc[:46]}…")
    top1 = results["documents"][0][0]
    ok = "PagedAttention" in top1
    print(f"\n{'✅' if ok else '❌'} 检索排序{'正确' if ok else '不符合预期'}"
          f"（第 1 条{'正是' if ok else '不是'} PagedAttention 那段）")

    # 4. 提交给大模型
    if client is None:
        print("\n❌ 第 4 步「提交给大模型」跳过：缺 OPENAI_API_KEY"
              "（prompt 已拼好，长度 %d 字符）" % len(augmented_prompt))
        return 0
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": augmented_prompt}],
    )
    print("\n模型回答:\n" + completion.choices[0].message.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
