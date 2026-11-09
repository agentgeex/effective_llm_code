# 构建大模型知识图谱（§11.4）

## 这个案例干什么

从中文语料里**抽取**实体与关系，建图并落盘，然后在图上做查询。

| 文件 | 干什么 |
|---|---|
| `build_kg.py` | 抽实体/关系建图 → `outputs/kg.json` |
| `query_kg.py` | 在落盘的图上查询：`stats` / `search` / `neighbors` / `path` / `filter` / `intent` / `context` |
| `sample_corpus/` | 三篇示例语料（变电站继电保护领域） |

抽取两路并行，来源可追溯：

1. **领域词表命中** —— 只定「词 → 类型」，不定「图里有哪些节点」。词表里有、正文没出现的词不进图。
2. **jieba 词性开放抽取** —— 取名词/专名/英文，捞出词表没覆盖的词。这才叫「抽」，不是「查表」。

关系靠**句内共现 + 触发词定向**：「导致/引发」→ 导致，「触发/启动」→ 触发，「提供/供给」→ 提供；两个实体间没有触发词就记一条 `共现` 边，置信度 low。每条边都带原文依据（`evidence`）。

## 对应书哪一节

第 11 章 §11.4「如何构建大模型知识图谱」，尤其是 §11.4.5「手搓知识图谱的实践路线」的五步（知识审计 → 实体建模 → 关系建模 → 意图映射 → 工具封装）。

书里第二步、第三步给的是**手填的实体/关系 JSON**（讲的是建模，不是抽取）。手填 4 个实体没问题，面对 4000 个就不行了 —— 所以这里补上「抽取」这一步，并用 `GOLDEN` 里的人工标注做自检：那几条关键关系能不能被抽出来。

## 怎么跑

```bash
pip install jieba

# 建图（默认用自带的 3 篇语料）
python ch11/cases/knowledge_graph/build_kg.py
# → outputs/kg.json，并在终端打印节点/边数与前若干条

# 换自己的语料（目录或单个文件都行）
python ch11/cases/knowledge_graph/build_kg.py --corpus /path/to/docs

# 调抽取门槛
python ch11/cases/knowledge_graph/build_kg.py --min-freq 2 --min-len 3

# 查询（--kg 是全局参数，放在子命令前面）
python ch11/cases/knowledge_graph/query_kg.py neighbors 备自投装置
python ch11/cases/knowledge_graph/query_kg.py path 主变跳闸 有限转供能力
python ch11/cases/knowledge_graph/query_kg.py intent "主变跳闸了怎么办"
python ch11/cases/knowledge_graph/query_kg.py context 备自投装置
```

## 依赖什么

- `jieba`（中文分词）。**没有别的第三方依赖**：不连数据库、不需要 Neo4j、不需要向量库、不需要模型权重。
- 不需要 API key。

## 数据从哪来

仓库自带 `sample_corpus/` 三篇短文（约 1000 字符，变电站站用电与继电保护场景），**不需要下载任何东西**。

想用自己的资料：`--corpus` 指到一个目录即可，`.md` / `.txt` 都吃。领域词表 `LEXICON` 在 `build_kg.py` 里，换成你自己行业的词就行 —— 不换也能跑，只是类型判断会退化成按后缀猜。

## 提醒

- 三篇短文是**教学示意**，不是真实规程；里面的设备名、参数、处置步骤不能当操作依据。
- 抽取质量取决于词表和语料规模。三篇短文能出几十个节点，不代表这套方法能直接处理一本 500 页手册而不调参。
