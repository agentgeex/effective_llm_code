# Effective LLMs — Companion Code

**English** | [简体中文](README.md)

Every technique the book walks through has code here that either runs, or says plainly why it won't.
The code is organised by book chapter: `ch01/` – `ch14/`.
**The chapter ↔ file mapping is in [INDEX.md](INDEX.md)** — that single table is where you look things up, by chapter or by file. *(Heads-up: INDEX.md, ENV_MATRIX.md and the per-chapter READMEs are currently Chinese-only. The tables are structural enough to navigate, and the code plus its header comments are self-explanatory.)*

> No demo video, no benchmark leaderboard, no "one-click run". What you get instead: **every claim has reproducible code behind it, and when something doesn't run, we say so.**

---

## 1. What this is

This is the example-code repository for the book *有效的大模型* ("Effective Large Language Models"). One judgement shaped the whole thing:

> Where readers actually get stuck is almost never "I don't know what the API looks like". It's "I copied it straight out of the book and it still doesn't work."

So the trade-off we settled on is **better to cover less than to mislead**:

- Every example file opens with a six-line header comment: what it does, which section of the book it belongs to, how to run it, what it depends on, which interpreter it was actually tested with, and how it differs from the book.
- Examples that can't run are not deleted. They're labelled with **what's missing** (VRAM, API key, external service, model weights) and with a fallback path.
- Every parameter, threshold and dataset inside the examples is **illustrative, for teaching** — not production configuration, and certainly not an operating procedure.

## 2. Getting started (three steps to your first example)

First example: **no API key, no GPU required**:

```bash
# 1) install dependencies (uv recommended; pip works too)
uv sync                      # or: pip install -e .

# 2) run a purely local example: self-attention from scratch
python ch04/transformer_min.py

# 3) want to know what a given file does and how to run it
python -c "print(open('INDEX.md').read())"   # or just open INDEX.md
```

To run an example that needs a key (e.g. `ch03/api_call_deepseek.py`):

```bash
export DEEPSEEK_API_KEY=...     # see .env.example — variable names only, no values
python ch03/api_call_deepseek.py
```

**With no key, these examples fail loudly and exit with code 2 — they never quietly pretend to have succeeded.** That's deliberate: a silent failure is far harder to track down than a loud one.

## 3. Repository layout

```
.
├── ch01/ … ch14/          examples grouped by book chapter (ch11 adds one level for its case studies)
│   └── ch11/cases/        five complete case studies, each with its own README
│       ├── video_probe/       §11.1 detecting a drowning swimmer in video
│       ├── novel_gen/         §11.2 long-form novel generation
│       ├── finetune/          §11.3 fine-tuning a large model
│       ├── knowledge_graph/   §11.4 building a knowledge graph with an LLM
│       └── mcp_system/        §11.5 building a complete MCP system from scratch
├── common/                shared modules reused across chapters (database, logging, GPT wrappers, etc.)
├── config/                config loading (real values are never committed — example file only)
├── ecosystem/             ecosystem demos (DSPy, Gemini Live)
├── INDEX.md               chapter ↔ file mapping (the most useful table in this repo)
├── ENV_MATRIX.md          four interpreters / per-file dependency measurements
└── pyproject.toml         dependency declarations
```

## 4. Finding the code for a chapter

Two steps:

1. Open [INDEX.md](INDEX.md), find the file under your chapter — one line per file, describing what it does.
2. Before running it, check that file's row in [ENV_MATRIX.md](ENV_MATRIX.md) — **the package set it needs may not be the same one the file next to it needs.**

Every `.py` file also carries a six-line structured header comment, so you can just read the file:

```
① What it is      ② Chapter/section      ③ How to run
④ Dependencies    ⑤ Tested environment    ⑥ Differences from the book
```

## 5. Environments are layered (this line will save you half a day)

**The examples here need several mutually incompatible Python environments. Don't expect a single venv to run all of them.**

The reasons are concrete: `vllm` pins a specific `torch` + CUDA combination, `onnxruntime` needs a `numpy` that conflicts with it, and frameworks like `dspy` want a newer Python. Force them into one environment and you get something that installs but won't run — you'll hit an `ImportError` or a segfault at runtime, and then assume the code is broken.

So: **one environment per example, or one per each of the four groups in ENV_MATRIX.** ENV_MATRIX lists, file by file, which environment each one needs.

## 6. Known limitations

The honest part:

1. **Not every example runs unconditionally.** Anything that needs a GPU, an API key, large model weights, or an external service (MySQL, Redis, Milvus) states its prerequisites in the file header. If a file doesn't say "runs", it hasn't been verified.
2. **The fine-tuning case study uses template-generated data**, not a real manual corpus. It demonstrates the shape of the data pipeline, not model quality.
3. **The knowledge-graph case study uses three short documents** so that you can see a graph built within ten minutes. It is not a scale demonstration.
4. **The video case study needs sample footage.** The repo ships synthetic test clips, not real recordings.
5. **Parameters, thresholds and domain data in the examples are illustrative.** Don't use them as production configuration or as operating procedures.
6. **`outputs/` is not committed**: run an example once and you'll have it. There are no "pre-computed results" in the repo, because the results are supposed to be produced on your machine, by you.

## 7. License

The code in this repository is under the **Apache License 2.0**; the full text is in [LICENSE](LICENSE).
The text of the book and its example code are under the same license.

## 8. Security

If you find a leaked credential or a security problem, please report it privately instead of opening a public issue — see [SECURITY.md](SECURITY.md).
