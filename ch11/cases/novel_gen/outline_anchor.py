#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：机制 1 提纲锚定：生成/读取 outline.json，并构造记忆卡 prompt 前缀
# ② 对应书稿：ch11 §11.2，（机制 1）+（实践演示框架）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/novel_gen/outline_anchor.py --mock（真 LLM 去掉 --mock，需 DEEPSEEK_API_KEY）
# ④ 依赖：openai（经 common/gpt/gptservice.py）；mock 模式零 key
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 --mock rc=0
# ⑥ 与书稿的差异：补 generate_outline()（书稿大纲写死）；LLM 层改走仓库已有 gptservice；新增 --mock
# ══════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────
# 书稿出处：ch11 §11.2「长篇小说生成」（实践演示：一个可运行的框架）
# 本文件对应书稿的这几块：
#   · 配置块（MODEL_NAME / client）                    → 换成 LLM 类（真调 common/gpt/gptservice.py）
#   · class MemoryCard + build_prompt_context()        → 原样保留（含「只留最近 5 节」）
#   · 书里没有「生成大纲」这一步（大纲是写死的字符串）→ 补 generate_outline()
# 相对书稿的改动：
#   1. 书稿 `client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))` 直接建客户端
#      → 无 key 时 import 就会带着空 key 跑，且与仓库已有封装重复
#      → 改成走 common/gpt/gptservice.py 的 simplechat()（DEEPSEEK_API_KEY 或 OPENAI_API_KEY）
#   2. 书稿的大纲是 demo_workflow 里写死的三行中文
#      → 没法「输入题目→产出提纲」，机制 1 等于没演示
#      → 补 generate_outline(topic)：真模式让模型输出 JSON，--mock 用模板产出同结构 JSON
#   3. 书稿 `chapter_summaries[-5:]` 只写死在代码里
#      → 保留默认 5，但提成 RECENT_KEEP 常量，按书稿「悬疑类 8~12、日常类 3~5」可调
#   4. --mock 模式：书稿完全没有。本文件让 mock 走同一个 LLM 接口（真读 prompt 内容再产出），
#      不是「打印一句 OK」—— 详见 MockLLM.mock_respond 的注释
# 改动原则：只改到能跑，不改教学意图（记忆卡结构、5 节滑动窗口、温度取值全部照书）
# ─────────────────────────────────────────────────────────
"""
机制 1：提纲锚定（Outline Stem Prompting）

一句话：先把「大纲 + 角色设定 + 章节摘要」做成一张记忆卡，每次生成都把它当 prompt 前缀。

跑法：
    # 无 key：mock（真跑数据流，产出 outputs/outline.json）
    uv run python ch11/cases/novel_gen/outline_anchor.py --mock
    # 有 key：真 LLM
    export DEEPSEEK_API_KEY=***
    uv run python ch11/cases/novel_gen/outline_anchor.py --topic "父亲的旧木盒"

定级：B（mock 链路已验证；真 LLM 需 key，本机无 key 未验证）
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# 仓库根加进 sys.path：脚本在 ch11/cases/novel_gen/ 下，直接跑时 sys.path[0] 是脚本目录，
# import common.xxx 会失败。书稿没有这个问题（它只 import openai）。
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("NOVEL_MODEL", "deepseek-chat")
RECENT_KEEP = 5          # 书稿默认值：只保留最近 5 节摘要（悬疑类可调到 8~12）
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

# 温度取值照书稿「技术要点 2」：写剧情 0.8 / 校对 0.3 / 润色 0.6
TEMP_WRITE = 0.8
TEMP_CHECK = 0.3
TEMP_POLISH = 0.6


# ─────────────────────────── LLM 层 ───────────────────────────
class LLM:
    """真调用：走仓库已有的 common/gpt/gptservice.py（不另造 HTTP 客户端）

    书稿是 `client.chat.completions.create(...)` 直连；这里换成 simplechat()，
    它内部也是同一个 OpenAI SDK 调用，返回 (content, response)。
    """

    def __init__(self, mock: bool = False):
        self.mock = mock
        if not mock:
            key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit(
                    "未检测到 DEEPSEEK_API_KEY / OPENAI_API_KEY。\n"
                    "  想看流水线怎么跑：加 --mock（无需 key，产出物照样落文件）\n"
                    "  想用真模型：export DEEPSEEK_API_KEY=*** 后再跑"
                )

    def chat(self, system: str, user: str, temperature: float) -> str:
        if self.mock:
            return mock_respond(system, user)
        from common.gpt.gptservice import simplechat   # 延迟 import：mock 模式不需要它
        content, _resp = simplechat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=2048,
            model=MODEL,
        )
        return content


def mock_respond(system: str, user: str) -> str:
    """假模型 —— 但假在「谁写的」，不假在「数据流」。

    它真的解析传进来的 prompt：
      · 写作任务：从 user 里的【已有章节摘要】和【当前任务】取场景与道具，拼出正文
      · 校对任务：从 user 里的【待检查内容】检查角色对话字数（陈远 ≤20 字），出报告
      · 润色任务：从 user 里的【原文】取对话，做确定性改写
      · 提纲任务：按题目返回同结构 JSON
    所以同一段 prompt 变，mock 产出就跟着变 —— 这是「跑通数据流」和「打印 OK」的区别。
    """
    if "策划编辑" in system or "JSON" in system:
        topic = _first(user, r"【题目】\s*([^\n]+)") or "父亲的旧木盒"
        return json.dumps(mock_outline(topic), ensure_ascii=False, indent=2)

    if "文字编辑" in system:
        body = _first(user, r"【待检查内容】\s*\n(.*)") or ""
        return mock_check(body)

    if "小说编辑助手" in system:
        body = _first(user, r"【本节正文】\s*\n(.*)") or ""
        return mock_summary(body)

    if "润色专家" in system:
        original = _first(user, r"【原文】\s*\n(.*)") or ""
        style = _first(user, r"【风格要求】\s*([^\n]+)") or ""
        return mock_polish(original, style)

    # 剩下的只有「小说作家」的写作任务（书稿 WRITE_SYSTEM 的抬头）
    return mock_write(user)


def _first(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.S)
    return m.group(1).strip() if m else ""


def mock_outline(topic: str) -> dict:
    """按题目产出结构化提纲（章节树）。结构与真模式要求的 JSON 完全一致。"""
    return {
        "title": f"《{topic}》",
        "topic": topic,
        "logline": f"一个程序员在父亲的遗物里发现{topic}，顺着它找到一段被抹掉的三十年前的科研往事。",
        "characters": {
            "陈远": {
                "desc": "28岁程序员，好奇心强但性格内向",
                "keywords": ["内向", "理性", "偶尔冷幽默"],
                "speech_style": "话少，每句话不超过20字，很少用感叹号",
            },
            "老吴": {
                "desc": "退休物理老师，65岁，知识渊博但脾气古怪",
                "keywords": ["博学", "古怪", "刀子嘴豆腐心"],
                "speech_style": "爱用反问句，偶尔引经据典，说话带点文言腔",
            },
        },
        "chapters": [
            {
                "no": 1,
                "title": "起：遗物里的东西",
                "core_event": "陈远整理父亲遗物，发现一只旧木盒。",
                "sections": [
                    {"no": 1, "title": "周末的整理",
                     "goal": "陈远在周末整理父亲遗物时发现旧木盒，描写樟脑味、盒面的划痕和开盒的迟疑。",
                     "props": ["旧木盒", "父亲的遗物", "樟脑味"]},
                    {"no": 2, "title": "盒底",
                     "goal": "陈远发现盒底刻着符号，旁边卡着一把铜钥匙，决定找人问。",
                     "props": ["铜钥匙", "符号", "旧木盒"]},
                ],
            },
            {
                "no": 2,
                "title": "承：找老吴",
                "core_event": "陈远带着盒子去找退休物理老师老吴，老吴认出符号。",
                "sections": [
                    {"no": 1, "title": "第一次上门",
                     "goal": "陈远带旧木盒找老吴，两人初次对话；老吴用反问句压他，铜钥匙被老吴拿起来看。",
                     "props": ["老吴", "铜钥匙", "旧木盒"]},
                    {"no": 2, "title": "三十年前的项目",
                     "goal": "老吴说符号属于三十年前一个被撤销的项目，但只肯说一半。",
                     "props": ["符号", "项目", "老吴"]},
                ],
            },
            {
                "no": 3,
                "title": "转：另一半",
                "core_event": "图纸背面还有一半符号，指向第二只木盒。",
                "sections": [
                    {"no": 1, "title": "背面的字",
                     "goal": "陈远回到住处翻看图纸背面，发现与符号配套的半句话。",
                     "props": ["图纸", "符号"]},
                    {"no": 2, "title": "第二只木盒",
                     "goal": "循着半句话，陈远在旧仓库找到第二只木盒，这次有人先到过。",
                     "props": ["第二只木盒", "旧仓库", "图纸"]},
                ],
            },
        ],
    }


KNOWN_OBJECTS = ["旧木盒", "铜钥匙", "符号", "图纸", "第二只木盒", "父亲的遗物"]


def mock_write(user: str) -> str:
    """从 prompt 里取【当前任务】和【已有章节摘要】，拼出与它们相关的中文正文。

    注意：正文不是把 goal 原样抄一遍（那只是任务描述），而是
      ① 用 goal 里出现的具体物件当道具；
      ② 开头先接住【已有章节摘要】里最后一条提到的物件 —— 这一句就是「锚定生效」的可见证据。
    """
    task = _first(user, r"【当前任务】(.*?)(?:\n\n|$)") or ""
    prior_lines = re.findall(r"^第\d+章第\d+节：.+$", user, re.M)
    prior = prior_lines[-1] if prior_lines else ""

    ch = re.search(r"第(\d+)章第(\d+)节", task)
    ch_no, sec_no = (ch.group(1), ch.group(2)) if ch else ("1", "1")

    props = [p for p in KNOWN_OBJECTS if p in task]
    obj1 = props[0] if props else "那个东西"
    obj2 = props[1] if len(props) > 1 else obj1
    prior_obj = next((p for p in KNOWN_OBJECTS if p in prior), "")

    # ① 开场：接住前文（有摘要时点名上一节出现过的物件）
    lead = (f"{prior_obj}还压在桌角，他没动。" if prior_obj
            else "周末的下午，屋里只有冰箱的嗡嗡声。")

    para1 = (f"{lead}陈远把{obj1}挪到灯下，指腹在磨得发亮的地方停了一下。"
             f"抽屉里那些年攒下的票据还散着，他没去收。")

    # ② 老吴只在他该出场的节里说话（第 1 章是陈远独自发现，别让他提前登场）
    if "老吴" in task or "老吴" in prior:
        para2 = (f"“这东西…你从哪儿翻出来的？”老吴问，眼睛没离开{obj2}。\n"
                 f"“柜子最下层。”\n"
                 f"“问了也没用吧，你自己心里没数？”\n"
                 f"“没有。”")
        para3 = (f"老吴把{obj2}翻过来，指节在边缘敲了两下，不说话。"
                 f"窗外有人推着车过去，铁皮在地上拖出一串响。"
                 f"陈远数着自己的呼吸，数到第七下，老吴才把东西放下。")
        para4 = (f"“先放我这儿。”老吴说。\n"
                 f"“明天我来拿。”\n"
                 f"陈远把{obj1}收回盒子，扣上。走出楼道时天已经暗了，"
                 f"他想起{obj2}上那道划痕，忽然觉得那不是磨损，是有人刻意刻的。")
    else:
        para2 = (f"盒盖卡了一下才开。里面铺着一层发黄的绒布，{obj1}就搁在正中，"
                 f"铜扣的颜色早没了，只剩一圈暗绿。")
        para3 = (f"他把{obj1}端到灯下。内壁有一道浅浅的划痕，"
                 f"不是刀刻的，更像是反复摩挲出来的。父亲生前从没提过它。")
        para4 = (f"陈远坐了很久，直到楼下的孩子散尽。"
                 f"他把{obj1}重新扣好，又觉得不妥，改成塞进背包最里层。"
                 f"总得找个人问问——父亲那辈人里，还认得这物件的没几个了。")

    return f"【第{ch_no}章第{sec_no}节】\n\n{para1}\n\n{para2}\n\n{para3}\n\n{para4}"


def mock_summary(body: str) -> str:
    """把一节压成「发生了什么 + 埋了什么伏笔」，≤200 字（对齐书稿示例的写法）

    注意：摘要只写正文里真发生过的事 —— 正文里没有老吴，摘要就不许出现老吴，
    否则下一节的前缀会带着一个不存在的人物，越写越歪（这正是要防的「漂移」）。
    """
    objs = [p for p in KNOWN_OBJECTS if p in body]
    obj1 = objs[0] if objs else "那个东西"
    obj2 = objs[1] if len(objs) > 1 else ""
    tail = "结尾暗示划痕是人为刻的，埋下伏笔。" if "划痕" in body else "留下未解释的线索。"
    if "老吴" in body:
        return f"陈远与老吴就{obj1}{('、' + obj2) if obj2 else ''}交谈，老吴只肯说一半；{tail}"[:200]
    return f"陈远独自整理{obj1}{('，并注意到' + obj2) if obj2 else ''}；{tail}"[:200]


def mock_check(body: str) -> str:
    """确定性一致性检查：陈远的对话行不得超过 20 字（书稿角色设定里就是这么写的）。"""
    problems = []
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        m = re.match(r"^“(.+)”$", line)
        if not m:
            continue
        said = m.group(1)
        # 上一行如果是老吴的发问，则这一行是陈远的回答（书稿示例的对话格式）
        prev = lines[i - 1] if i > 0 else ""
        if "老吴" in prev or "问" in prev:
            if len(said) > 20:
                problems.append(f"- 陈远的回答 {len(said)} 字，超过设定上限 20 字：“{said}”")
    if problems:
        return "发现以下一致性问题：\n" + "\n".join(problems)
    return "一致性检查通过：陈远对话均 ≤20 字，未发现与角色设定/已有摘要冲突之处。"


def mock_polish(original: str, style: str) -> str:
    """确定性润色：去掉填充词、把长句拆短、补一个节拍。不新增情节。"""
    out = original.replace("我觉得吧，", "").replace("其实", "").replace("然后", "")
    out = re.sub(r"[，,]{2,}", "，", out)
    tag = f"（润色：{style[:24]}）" if style else "（润色）"
    return f"{out.strip()}\n\n{tag}"


# ─────────────────────── 记忆卡（书稿原结构） ───────────────────────
class MemoryCard:
    """全局记忆卡：大纲 + 角色设定 + 章节摘要（书里那个类，字段与方法同名）"""

    def __init__(self, outline: str, characters: dict):
        self.outline = outline
        self.characters = characters      # {name: {desc, keywords, speech_style}}
        self.chapter_summaries = []       # [{chapter, section, summary}]

    def add_summary(self, chapter: int, section: int, summary: str):
        self.chapter_summaries.append(
            {"chapter": chapter, "section": section, "summary": summary}
        )

    def build_prompt_context(self) -> str:
        """构建可作为 prompt 前缀的记忆卡上下文（书稿原实现，只把 5 提成常量）"""
        parts = ["【故事大纲】", self.outline, ""]
        parts.append("【角色设定】")
        for name, info in self.characters.items():
            speech = info.get("speech_style", "")
            keywords = ", ".join(info.get("keywords", []))
            desc = info.get("desc", "")
            parts.append(f"- {name}：{desc}（{keywords}）说话风格：{speech}")

        if self.chapter_summaries:
            parts.append("\n【已有章节摘要】")
            # 只保留最近 RECENT_KEEP 节摘要 —— 书稿的「选择性遗忘」
            for s in self.chapter_summaries[-RECENT_KEEP:]:
                parts.append(f"第{s['chapter']}章第{s['section']}节：{s['summary']}")
        return "\n".join(parts)


# ─────────────────────────── 机制 1 ───────────────────────────
OUTLINE_SYSTEM = """你是一位小说策划编辑。请根据题目输出严格的 JSON（不要 markdown 代码块），字段：
title / topic / logline / characters{姓名:{desc,keywords,speech_style}} /
chapters[{no,title,core_event,sections[{no,title,goal,props}]}]
要求：3 章，每章 2 节；goal 一句话说清这一节发生什么；props 列该节出现的具体物件。"""


def outline_to_text(data: dict) -> str:
    """把提纲压成一行行文本，塞进记忆卡的【故事大纲】（书稿的 outline 就是一串中文）"""
    lines = [data.get("logline", "")]
    for ch in data.get("chapters", []):
        lines.append(f"第{ch['no']}章 {ch['title']}：{ch['core_event']}")
        for sec in ch.get("sections", []):
            lines.append(f"  第{ch['no']}章第{sec['no']}节 {sec['title']}：{sec['goal']}")
    return "\n".join(lines)


def generate_outline(llm: LLM, topic: str) -> dict:
    """输入题目 → 输出结构化提纲（章节树）"""
    raw = llm.chat(OUTLINE_SYSTEM, f"【题目】{topic}\n\n请输出 JSON。", TEMP_CHECK)
    raw = raw.strip()
    if raw.startswith("```"):                      # 真模型爱套代码块围栏
        raw = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"提纲不是合法 JSON（{exc}）。原文前 200 字：\n{raw[:200]}")
    if not data.get("chapters"):
        raise SystemExit("提纲缺少 chapters 字段")
    return data


def build_memory(data: dict) -> MemoryCard:
    return MemoryCard(outline=outline_to_text(data), characters=data.get("characters", {}))


def save_outline(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_outline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="机制 1：提纲锚定")
    ap.add_argument("--mock", action="store_true", help="用假模型跑通数据流（无需 key）")
    ap.add_argument("--topic", default="父亲的旧木盒", help="小说题目")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "outline.json"))
    args = ap.parse_args()

    llm = LLM(mock=args.mock)
    print(f"[模式] {'mock（假模型，数据流真跑）' if args.mock else '真 LLM：' + MODEL}")
    print(f"[题目] {args.topic}")

    data = generate_outline(llm, args.topic)
    mem = build_memory(data)
    out = save_outline(data, Path(args.out))

    total_sections = sum(len(c["sections"]) for c in data["chapters"])
    print(f"[提纲] {data.get('title')} — {len(data['chapters'])} 章 / {total_sections} 节")
    for ch in data["chapters"]:
        print(f"  第{ch['no']}章 {ch['title']}")
        for sec in ch["sections"]:
            print(f"    第{ch['no']}章第{sec['no']}节 {sec['title']}：{sec['goal'][:34]}…")
    print(f"\n[记忆卡前缀] {len(mem.build_prompt_context())} 字符（含角色 "
          f"{len(mem.characters)} 个）")
    print(f"[产出] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
