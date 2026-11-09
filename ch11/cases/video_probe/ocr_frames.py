#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：关键帧 OCR 并与 groundtruth 比对（字符级 diff）
# ② 对应书稿：ch11 §11.1（书稿未写 OCR 预处理）
# ③ 怎么跑：cd <仓库根> && python（需 torch 的环境） \
#            ch11/cases/video_probe/ocr_frames.py --engine pytesseract
# ④ 依赖：opencv-python（cv2）、pytesseract、Pillow、numpy；系统 tesseract 可执行文件
# ⑤ 实测解释器：python（需 torch 的环境）（cv2 5.0.0 / pytesseract 0.3.13）
#    系统 tesseract 4.1.1，tessdata = chi_sim / eng / osd
#     三种跑法**全部 rc=0**（当次真跑，outputs/ocr.json 为证）：
#      --engine pytesseract    → 真识别
#      --engine tesseract-cli  → 真识别（本轮补的分支，之前是空分支）
#      （不带 --engine = auto）→ 选中 tesseract-cli，真识别；「可用引擎」= 真正能用的那 2 个
#    三种跑法 5/5 帧相似度全 1.0；`--engine easyocr`（不存在）→ rc=2 且明说原因
# ⑥ 与书稿的差异：书稿无此脚本
# ══════════════════════════════════════════════════════════════════
"""ocr_frames.py — 关键帧文字提取

★ 更新：**tesseract 已装上，真 OCR 已跑通**。
    环境：apt 装 tesseract-ocr 4.1.1（含 chi_sim）+ venv-gpu 的 pytesseract 0.3.13
    命令：venv-gpu/bin/python ch11/cases/video_probe/ocr_frames.py --engine pytesseract
    结果：rc=0，引擎探测 2/5 可用（tesseract-cli、pytesseract），真识别出
          frame 25 → "SCENE 1 CH-01 FRAMES 000-049"、frame 147 → "SCENE 3 CH-03 FRAMES 110-184"
    仍缺：easyocr / paddleocr / cnocr（都要额外模型权重，本轮没装）

✅ **已修（PM 批准）**：此前不带 `--engine` 时自动探测选中 `tesseract-cli`，
   但代码只有 `pytesseract` 分支 → 落 `status=error: 未实现针对 tesseract-cli 的调用分支`，
   同一份输出里又写「可用引擎：2 个」—— 自相矛盾。改了三处：
     ① 补 `tesseract-cli` 真分支（subprocess + 超时 60s，失败写进 status/error，不抛栈）
     ② 探测不再只看「有没有」，而是**真能不能用**：
        `tesseract-cli` = 可执行文件在 且 `tesseract --version` rc=0；
        `pytesseract`   = 模块能 import 且 **tesseract 本体在**（没有本体，pytesseract 只是壳）；
        其余三个 = 能 import。
        探测到但不满足上述条件的，**不进「可用引擎」列表**。
     ③ 自动选引擎时按 ENGINES 顺序取第一个**可用**的，不再假设「探测到 = 能用」。

★ 下面这段是早先的历史记录（当时确实没有任何 OCR 引擎），保留不改，作为「为什么拖到现在」的交代。

试过（sgaiedge17 本机，系统 python3 + /tmp/venv_ft 都试过）：
    tesseract CLI   → 命令不存在
    pytesseract     → ModuleNotFoundError
    easyocr         → ModuleNotFoundError（且首次运行要下载 ~100MB 识别模型 → 违反本次「不许下载模型」）
    paddleocr       → ModuleNotFoundError（paddle 轮子 + 模型共数百 MB）
    cnocr           → ModuleNotFoundError（同样带模型）

为什么不装：
    tesseract 本体在 apt 里（`tesseract-ocr` + `tesseract-ocr-chi-sim` ≈ 15MB），
    但需要 root、且属于「新增系统依赖」，按单子规矩必须先报 PM。
    easyocr / paddleocr / cnocr 都要拉模型权重，与本次「不许下载任何模型」直接冲突。

那这一环给什么
==============
不能假装识别了。给的是**「抽帧对位」的硬证据**，它证明的是「帧取对了、位置对」，
证明不了「字是什么」。两件事必须分清楚：

  ① 文字区「有墨」检查 —— 在已知的文字带（spec.drawtext.y 附近）数亮像素，
     与同一帧的空白对照带比。有墨 ≠ 识别成功，只说明那个位置确实被烧了字。
  ② 方块质心反算 —— 见 probe.py 的 box_alignment：合成片里白方块位置是
     `floor(10 + 80*t)`，用像素质心反算帧号，±3.2px 就能发现错一帧。
     这条比 ① 强得多，是真正的「对位」证据。

启用真正的 OCR 只要两步（等 PM 批准后）：
    apt install -y tesseract-ocr tesseract-ocr-chi-sim   # 需 root
    pip install pytesseract                              # 约 100KB
然后 `python ocr_frames.py --engine pytesseract` 就会走真识别分支，
并把识别结果与 spec 里的已知文本做字符级比对（合成片的好处：标准答案是现成的）。

★ 预处理（后来加的，不是装饰）
====================================
合成片的字是**纯白**、底是**各场景彩色**。原图直送 tesseract：
    蓝底 #1E3A8A / 绿底 #15803D / 深灰底 #1F2937 → 能读（相似度 0.933）
    红底 #B91C1C → 读成 'eit)'（相似度 0.0）
    橄榄底 #A16207 → 读成 '20'（相似度 0.067）
原因：灰度化后红/橄榄底与白字贴太近，Otsu 分不开。
解法：取 **min(R,G,B)** —— 白字三通道都接近 255，min 也高；彩色底总有一两个通道很低，
      min 随之很低。于是白字天然被「抬」出来。再 Otsu 自动阈值 + 20px 白边。
结果：5 帧全部相似度 **1.0**（两个引擎都是）。
`--no-preprocess` 可关掉，用于对照（关掉后红/橄榄底两帧立刻掉回 0.0 / 0.067）。

用法
====
  python ocr_frames.py                        # auto：按 ENGINES 顺序取第一个**真能用**的
  python ocr_frames.py --engine tesseract-cli  # 强制 CLI（补的分支）
  python ocr_frames.py --engine pytesseract    # 强制 Python 壳
  python ocr_frames.py --probe-ink             # 只做「有墨」对位检查，不调 OCR
  python ocr_frames.py --no-preprocess         # 关掉二值化预处理（对照用）

退出码：0 = 识别成功（或按 --probe-ink 正常降级）；1 = 识别全失败；2 = 指定引擎不可用 / 缺 keyframes.json

依赖：cv2 + numpy。OCR 引擎可选，缺了就降级为对位检查。
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FRAMES = OUT / "frames"

# 引擎探测表：(名字, 探测方式, 说明, 安装代价)
ENGINES = [
    ("tesseract-cli", "bin", "tesseract", "系统 apt 包 tesseract-ocr", "≈15MB（含 chi-sim 语言包），需 root"),
    ("pytesseract", "py", "pytesseract", "tesseract 的 Python 壳", "≈100KB + 需先装 tesseract 本体"),
    ("easyocr", "py", "easyocr", "基于 torch 的 OCR", "≈100MB+ 模型权重，首次运行会下载"),
    ("paddleocr", "py", "paddleocr", "PaddleOCR", "数百 MB（paddle 轮子 + 模型）"),
    ("cnocr", "py", "cnocr", "中文 OCR", "数十 MB + 模型权重"),
]


def tess_langs() -> list:
    """`tesseract --list-langs` 的实际结果。拿不到就返回空列表（调用方据此降级）。"""
    exe = shutil.which("tesseract")
    if not exe:
        return []
    try:
        p = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if p.returncode != 0:
        return []
    langs = [ln.strip() for ln in (p.stdout or "").splitlines()[1:]]   # 首行是 "List of available languages (N):"
    return [ln for ln in langs if ln and " " not in ln]


def pick_langs(preferred: tuple = ("chi_sim", "eng")) -> tuple:
    """返回 (langs_arg, notes)。langs_arg 形如 'chi_sim+eng'；一个都没有就返回空串。"""
    have = tess_langs()
    use = [x for x in preferred if x in have]
    notes = f"tessdata 可用语言: {'/'.join(have) if have else '（读不到）'}"
    if not use:
        return "", notes + "；首选语言都不在 → 不传 -l（走 tesseract 默认）"
    miss = [x for x in preferred if x not in have]
    if miss:
        notes += f"；缺 {'/'.join(miss)} → 只用 {'+'.join(use)}"
    return "+".join(use), notes


def probe_engines() -> list:
    """探测每个引擎**是不是真能用**，而不只是「在不在」。

    关键区别（这正是旧版自相矛盾的根）：
      · tesseract-cli：`shutil.which` 命中还不够，要 `tesseract --version` 真跑一次 rc=0
      · pytesseract  ：能 import 还不够，**tesseract 本体必须在**（pytesseract 只是壳，
                       本体不在时 import 成功但一调用就 TesseractNotFoundError）
    """
    rows = []
    have_tess = bool(shutil.which("tesseract"))
    for name, kind, target, desc, cost in ENGINES:
        row = dict(name=name, kind="binary" if kind == "bin" else "python",
                   target=target, desc=desc, install_cost=cost, available=False)
        if kind == "bin":
            path = shutil.which(target)
            row["path"] = path
            if not path:
                row["error"] = f"可执行文件 {target} 不在 PATH"
            else:
                try:
                    p = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
                    if p.returncode == 0:
                        row["available"] = True
                        row["version"] = (p.stdout or "").splitlines()[0].strip()
                    else:
                        row["error"] = f"`{target} --version` rc={p.returncode}: {(p.stderr or '').strip()[:200]}"
                except subprocess.TimeoutExpired:
                    row["error"] = f"`{target} --version` 超时 20s"
                except OSError as e:
                    row["error"] = f"OSError: {e}"
        else:
            try:
                mod = __import__(target)
                row["version"] = getattr(mod, "__version__", "?")
            except Exception as e:      # noqa: BLE001 — 探测就是要把所有失败都记下来
                row["error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                continue
            # 壳类引擎的前置条件：tesseract 本体
            if target == "pytesseract" and not have_tess:
                row["error"] = "模块可 import，但 tesseract 本体不在 PATH → 一调用就失败，不算可用"
            else:
                row["available"] = True
        rows.append(row)
    return rows


def ink_check(img_path: Path, spec: dict | None, frame_no: int) -> dict:
    """在已知文字带数亮像素，并与空白对照带比较。

    只回答「这个位置有没有被烧过字」，不回答「字是什么」。别混淆。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return dict(frame=frame_no, error=f"读不到 {img_path}")

    if spec and spec.get("drawtext"):
        dt = spec["drawtext"]
        y = int(dt["y"])
        fs = int(dt["fontsize"])
        band = (max(0, y - 12), y + fs + 12)          # 含 boxborderw
    else:
        band = (30, 100)

    def bright(roi):
        m = (roi[:, :, 0] > 200) & (roi[:, :, 1] > 200) & (roi[:, :, 2] > 200)
        return m, int(m.sum())

    text_roi = img[band[0]:band[1], :, :]
    m, n_text = bright(text_roi)
    # 对照带：紧贴文字带下方的空白区（合成片里这里是纯背景）
    ctrl_roi = img[band[1] + 20:band[1] + 90, :, :]
    _, n_ctrl = bright(ctrl_roi)

    bbox = None
    if n_text > 0:
        ys, xs = np.nonzero(m)
        bbox = dict(left=int(xs.min()), right=int(xs.max()),
                    top=int(ys.min()) + band[0], bottom=int(ys.max()) + band[0])

    return dict(
        frame=frame_no, frame_path=str(img_path.relative_to(HERE)),
        text_band=list(band), control_band=[band[1] + 20, band[1] + 90],
        bright_px_text_band=n_text, bright_px_control_band=n_ctrl,
        ink_bbox=bbox,
        ratio=round(n_text / max(1, n_ctrl), 2),
        verdict="文字带明显有墨（位置对得上）" if n_text > 100 * max(1, n_ctrl) else "⚠️ 文字带未见明显墨迹",
    )


def binarize_for_ocr(img_path: Path):
    """把「彩色底 + 白字」二值化成「白底黑字」。

    为什么要这一步（实测踩到的）：
      合成片的字是**纯白**，底是**各场景彩色**。直接送 tesseract，
      蓝底(#1E3A8A)/绿底(#15803D)/深灰底(#1F2937) 能读出，
      但红底(#B91C1C)和橄榄底(#A16207)读成 'eit)' / '20' —— 灰度化后前景背景贴太近。
      做法：取 **min(R,G,B)**。白字三个通道都接近 255 → min 也高；
            彩色底总有一两个通道很低 → min 很低。于是白字天然被「抬」出来。
      再加 Otsu 自动阈值 + 20px 白边（tesseract 对贴边文字不友好）。

    返回 (bw, note)；读不到图返回 (None, 原因)。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, f"cv2.imread 读不到 {img_path}"
    mn = img.min(axis=2)
    _, bw = cv2.threshold(mn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return bw, "min-channel + Otsu + 20px 白边"


def _ocr_one_pytesseract(img_path: Path, langs: str, preprocess: bool = True) -> dict:
    try:
        import pytesseract
        from PIL import Image
    except Exception as e:                  # noqa: BLE001
        return dict(error=f"import 失败 {type(e).__name__}: {e}")
    src = Image.open(img_path)
    note = "原图"
    if preprocess:
        bw, note = binarize_for_ocr(img_path)
        if bw is None:
            return dict(error=note)
        src = Image.fromarray(bw)
    try:
        cfg = "-c preserve_interword_spaces=1" + (f" -l {langs}" if langs else "")
        txt = pytesseract.image_to_string(src, config=cfg).strip()
        return dict(text=txt, preprocess=note)
    except Exception as e:                  # noqa: BLE001 — 失败要进 status/error，不抛栈
        return dict(error=f"{type(e).__name__}: {e}")


def _ocr_one_tesseract_cli(img_path: Path, langs: str, preprocess: bool = True) -> dict:
    """tesseract CLI：`tesseract <img> stdout -l chi_sim+eng`。

    超时 60s / OSError / 非零 rc 一律折成 dict(error=...)，由调用方写进 status/error。
    preprocess=True 时先二值化到临时 png（用完即删）。
    """
    target, note, tmpdir = Path(img_path), "原图", None
    if preprocess:
        bw, note = binarize_for_ocr(img_path)
        if bw is None:
            return dict(error=note)
        tmpdir = tempfile.mkdtemp(prefix="ocr_bw_")
        target = Path(tmpdir) / (img_path.stem + "_bw.png")
        cv2.imwrite(str(target), bw)
    try:
        return _run_tesseract(target, langs, note)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_tesseract(target: Path, langs: str, note: str) -> dict:
    cmd = ["tesseract", str(target), "stdout"]
    if langs:
        cmd += ["-l", langs]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return dict(error="TimeoutExpired: tesseract 超过 60s 未返回")
    except FileNotFoundError:
        return dict(error="FileNotFoundError: 找不到 tesseract 可执行文件")
    except OSError as e:
        return dict(error=f"OSError: {e}")
    if p.returncode != 0:
        return dict(error=f"tesseract rc={p.returncode}: {(p.stderr or '').strip()[:300]}")
    return dict(text=(p.stdout or "").strip(), preprocess=note)


def run_real_ocr(name: str, kfs: list, spec: dict | None, preprocess: bool = True) -> dict:
    """真有引擎时才走这里：真识别 + 与 spec 已知文本做字符级比对。"""
    langs, lang_note = pick_langs()
    runners = {
        "tesseract-cli": _ocr_one_tesseract_cli,
        "pytesseract": _ocr_one_pytesseract,
    }
    runner = runners.get(name)
    if runner is None:
        impl = "、".join(runners)
        return dict(status="error",
                    error=f"未实现针对 {name} 的调用分支（已实现的只有 {impl}）")

    results = []
    for kf in kfs:
        img_path = HERE / kf["path"]
        r = dict(frame=kf["frame"])
        r.update(runner(img_path, langs, preprocess))
        results.append(r)

    # 与标准答案比：合成片的文字是 spec 里写死的
    if spec:
        for r in results:
            truth = next((s["text"] for s in spec["scenes"] if s["start_frame"] <= r["frame"] <= s["end_frame"]), "")
            r["truth"] = truth
            if "text" in r:
                r["similarity"] = round(difflib.SequenceMatcher(None, r["text"], truth).ratio(), 3)

    n_ok = sum(1 for r in results if "text" in r)
    out = dict(status="ok" if n_ok else "error",
               engine=name, langs=langs, langs_note=lang_note,
               preprocess=("min-channel+Otsu+20px白边" if preprocess else "关闭（原图直送）"),
               n_ok=n_ok, n_total=len(results), results=results)
    if not n_ok:
        out["error"] = "所有帧都识别失败，见 results[].error"
    return out

    # 与标准答案比：合成片的文字是 spec 里写死的
    if spec:
        for r in results:
            truth = next((s["text"] for s in spec["scenes"] if s["start_frame"] <= r["frame"] <= s["end_frame"]), "")
            r["truth"] = truth
            if "text" in r:
                r["similarity"] = round(difflib.SequenceMatcher(None, r["text"], truth).ratio(), 3)
    return dict(status="ok", engine=name, results=results)


def main() -> int:
    ap = argparse.ArgumentParser(description="关键帧文字提取（无引擎时降级为对位检查）")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--probe-ink", action="store_true", help="只做有墨对位检查")
    ap.add_argument("--spec", default=str(HERE / "data" / "clip_spec.json"))
    ap.add_argument("--no-preprocess", action="store_true",
                    help="不做二值化预处理（默认做：白字彩色底，红/橄榄底原图直送会读错）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    kf_path = OUT / "keyframes.json"
    if not kf_path.exists():
        print("🔴 缺 outputs/keyframes.json —— 先跑 python probe.py", file=sys.stderr)
        return 2
    kfs = json.loads(kf_path.read_text(encoding="utf-8"))["frames"]
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8")) if Path(args.spec).exists() else None

    engines = probe_engines()
    usable = [e for e in engines if e["available"]]
    usable_names = [e["name"] for e in usable]
    print("引擎探测（判据：真能用，不只是『装了』）：")
    for e in engines:
        mark = "✅" if e["available"] else "❌"
        extra = e.get("path") or e.get("version") or e.get("error", "")
        print(f"  {mark} {e['name']:<15} {extra}")
    print(f"可用引擎：{len(usable)} 个" + (f" —— {'、'.join(usable_names)}" if usable else ""))

    doc = dict(probe_time_note="sgaiedge17 本机实测",
               engines=engines, n_engines_available=len(usable),
               usable_engines=usable_names)

    # 显式指定的引擎必须真能用；不能用就明说，不静默换成别的
    if args.engine != "auto" and args.engine not in usable_names:
        why = next((e.get("error") or "未在探测表中" for e in engines if e["name"] == args.engine), "未在探测表中")
        msg = f"指定的引擎 {args.engine} 不可用：{why}（可用：{'、'.join(usable_names) or '无'}）"
        print(f"🔴 {msg}", file=sys.stderr)
        doc.update(status="error", error=msg, ocr_implemented=False)
        (OUT / "ocr.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写出 {OUT/'ocr.json'}")
        return 2

    if usable and not args.probe_ink:
        name = usable_names[0] if args.engine == "auto" else args.engine
        how = "auto → 取第一个可用" if args.engine == "auto" else "显式指定"
        print(f"走真识别分支：{name}（{how}）")
        doc.update(run_real_ocr(name, kfs, spec, preprocess=not args.no_preprocess))
    else:
        if args.probe_ink:
            print("--probe-ink：只做抽帧对位检查，不调用 OCR")
            status, reason = "skipped_ocr", "--probe-ink 模式，按规定不调用 OCR"
        else:
            print("无可用 OCR 引擎 → OCR 环节标记【未实现】，只做抽帧对位检查")
            status, reason = "unavailable", (
                "本机没有任何**真能用**的 OCR 引擎 —— 探测明细见 engines[] 的 error 字段。"
                "注意『装了』≠『能用』：pytesseract 能 import 但 tesseract 本体不在时，仍不算可用。"
            )
        doc.update(
            status=status,
            ocr_implemented=False,
            reason=reason,
            why_not_installed=[
                "tesseract 本体属于系统 apt 依赖（需 root），且按规矩新增依赖要先报 PM",
                "easyocr / paddleocr / cnocr 首次运行都要下载模型权重，与『不许下载任何模型』冲突",
            ],
            enable_steps=["apt install -y tesseract-ocr tesseract-ocr-chi-sim",
                          "pip install pytesseract",
                          "python ocr_frames.py --engine pytesseract"],
            substitute_evidence_note=(
                "下方 ink_check 只证明『文字带里有墨、位置对得上』，证明不了『字是什么』。"
                "真正的对位硬证据在 outputs/groundtruth.json 的 box_alignment（白方块质心反算帧号）。"
            ),
            ink_check=[ink_check(HERE / k["path"], spec, k["frame"])
                       for k in kfs if k.get("path")],
        )

    (OUT / "ocr.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT/'ocr.json'}")

    if doc.get("ink_check"):
        print("\n文字带「有墨」检查（≠ OCR，只证明位置对）：")
        print(f"  {'帧':>5} {'文字带亮像素':>12} {'对照带':>8} {'比值':>7}  判定")
        for r in doc["ink_check"]:
            if "error" in r:
                print(f"  {r['frame']:>5}  🔴 {r['error']}")
                continue
            print(f"  {r['frame']:>5} {r['bright_px_text_band']:>12} "
                  f"{r['bright_px_control_band']:>8} {r['ratio']:>7}  {r['verdict']}")
            if r.get("ink_bbox"):
                b = r["ink_bbox"]
                print(f"        墨迹框 x[{b['left']}..{b['right']}] y[{b['top']}..{b['bottom']}]")
    if doc.get("results"):
        print(f"\n识别结果（引擎 {doc.get('engine')}，langs={doc.get('langs') or '默认'}）：")
        print(f"  {doc.get('langs_note', '')}")
        for r in doc["results"]:
            if "error" in r:
                print(f"  帧 {r['frame']:>5}  🔴 {r['error']}")
                continue
            first = (r["text"].splitlines() or [""])[0]
            sim = r.get("similarity")
            print(f"  帧 {r['frame']:>5}  相似度 {sim if sim is not None else '—'}  ← {first!r}")
    if doc.get("status") == "error":
        print(f"\n🔴 OCR 失败：{doc.get('error')}")
        return 1
    if doc.get("status") == "unavailable":
        print("\n⚠️  OCR 结论：**未实现**。原因见 ocr.json.reason。不许当成『已识别』。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
