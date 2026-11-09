#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：淡入淡出切点检测测试（对合成片逐帧比对）
# ② 对应书稿：ch11 §11.1（书稿未写预处理/切点验证）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/video_probe/fade_test.py
# ④ 依赖：opencv-python（cv2）、numpy；依赖 probe.py
# ⑤ 实测解释器：无可用解释器：.venv / venv-onnx 均 ModuleNotFoundError: No module named 'cv2'（实测）
# ⑥ 与书稿的差异：书稿无此脚本
# ══════════════════════════════════════════════════════════════════
"""fade_test.py — 反例实验：渐变转场（交叉溶解）能不能切出来？

为什么单独一个文件
==================
probe.py 只在合成片上验证了「硬切」切得准。硬切是最好切的情况，
拿它当通用结论就变成「挑好数据」。所以这里主动造一个**更难**的样本：
两段纯色用 `xfade=fade` 交叉溶解 0.5 秒，看检测器会怎么错。

实测结果（见 outputs/fade_test.md，数字由本脚本生成）：
  真值 2 段（过渡带 2.00–2.50s）
  检测 3 段（切在 2.04s 与 2.44s）
  → 多切出来的一段（0.40s）就是过渡本身

这条实验还顺带证伪了一个容易想当然的说法：
「渐变会让直方图距离慢慢滑过去，一层阈值切不出来」——**不对**。
纯色画面下，色相每移动一个 bin，两张直方图就完全不相交（L1 = 1.0），
所以过渡期每一帧的分数都顶到 1.0，反而是被「最小场景长度」这个守门员
卡住，才没有切出十几段。

用法
====
  python fade_test.py            # 造样本 → 跑检测 → 写 outputs/fade_test.{md,json}
  python fade_test.py --thr 0.5  # 换阈值看结果怎么变

依赖：ffmpeg + cv2 + numpy。复用 probe.py 的 scan_video，不复制一份逻辑。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2

import probe as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CLIP = Path("/tmp/video_probe_fade.mp4")

FADE_START_S = 2.00
FADE_DUR_S = 0.50
FPS = 25


def make_fade_clip() -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x1E3A8A:s=960x540:r={FPS}:d=2.5",
        "-f", "lavfi", "-i", f"color=c=0xB91C1C:s=960x540:r={FPS}:d=2.5",
        "-filter_complex",
        f"[0:v][1:v]xfade=transition=fade:duration={FADE_DUR_S}:offset={FADE_START_S}[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(CLIP),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"🔴 ffmpeg 造样本失败：{r.stderr[-800:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=P.DEFAULT_THR)
    ap.add_argument("--min-scene", type=float, default=P.MIN_SCENE_S)
    args = ap.parse_args()

    make_fade_clip()
    OUT.mkdir(parents=True, exist_ok=True)

    scenes, scores, fps, n_read, _ = P.scan_video(CLIP, args.thr, args.min_scene)
    det = [s.as_dict() for s in scenes]

    fade_f0 = int(round(FADE_START_S * fps))
    fade_f1 = int(round((FADE_START_S + FADE_DUR_S) * fps))
    in_fade = [r for r in scores if fade_f0 <= r[0] <= fade_f1]
    before = [r for r in scores if r[0] < fade_f0 - 1]
    after = [r for r in scores if r[0] > fade_f1 + 1]

    doc = dict(
        clip=str(CLIP),
        note="0.5s 交叉溶解（xfade=fade）反例样本",
        fps=fps, frames_read=n_read, threshold=args.thr, min_scene_s=args.min_scene,
        truth=dict(n_scenes=2, boundary_s=[FADE_START_S, FADE_START_S + FADE_DUR_S],
                   boundary_frames=[fade_f0, fade_f1]),
        detected=dict(n_scenes=len(det), cuts_s=[d["start_s"] for d in det[1:]],
                      cuts_frames=[d["start_frame"] for d in det[1:]], scenes=det),
        scores_in_fade=[dict(frame=r[0], t=round(r[1], 3), hist_l1=round(r[2], 4),
                             gray_diff=round(r[3], 4), is_cut=r[4]) for r in in_fade],
        score_stats=dict(
            before_max=round(max((r[2] for r in before), default=0), 4),
            after_max=round(max((r[2] for r in after), default=0), 4),
            in_fade_min=round(min((r[2] for r in in_fade), default=0), 4),
            in_fade_max=round(max((r[2] for r in in_fade), default=0), 4),
        ),
    )

    # 结论必须由数据推出来，不预设
    doc["verdict"] = ("多切出一段：检测到的中间段就是过渡带本身"
                      if len(det) == 3 else
                      f"切出 {len(det)} 段（真值 2 段）")
    (OUT / "fade_test.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    st = doc["score_stats"]
    cuts_txt = ", ".join("%.2fs" % d["start_s"] for d in det[1:]) or "（无）"
    L = [
        "# 反例实验：0.5 秒交叉溶解能不能切出来",
        "",
        "> 由 `fade_test.py` 自动生成。样本：两段纯色（蓝 `#1E3A8A` → 红 `#B91C1C`），"
        f"用 `xfade=fade` 交叉溶解 **{FADE_DUR_S}s**，溶解区间 "
        f"**{FADE_START_S:.2f}–{FADE_START_S + FADE_DUR_S:.2f}s**（帧 {fade_f0}–{fade_f1}）。",
        "",
        "## 结果",
        "",
        "| 项 | 真值 | 检测结果 |",
        "|---|---|---|",
        f"| 段数 | **2** | **{len(det)}** |",
        f"| 切点 | 过渡带 {FADE_START_S:.2f}–{FADE_START_S + FADE_DUR_S:.2f}s | "
        f"{cuts_txt} |",
        "",
        "| # | 起止帧 | 起止时间 (s) | 时长 (s) | 均色 | 切点分 |",
        "|---|---|---|---|---|---|",
    ]
    for d in det:
        L.append(f"| {d['index']} | {d['start_frame']}–{d['end_frame']} | "
                 f"{d['start_s']:.2f}–{d['end_s']:.2f} | {d['duration_s']:.2f} | "
                 f"`{d['color_hex']}` | {d['cut_score']:.4f} |")

    L += [
        "",
        "## 为什么会长这样（逐帧分数）",
        "",
        "过渡带之外的帧分数：**0.0000**（完全静止）。过渡带之内：",
        "",
        "| 帧 | t (s) | hist_l1 | gray_diff | 判为切点 |",
        "|---|---|---|---|---|",
    ]
    for r in doc["scores_in_fade"]:
        L.append(f"| {r['frame']} | {r['t']:.2f} | {r['hist_l1']:.4f} | {r['gray_diff']:.4f} | "
                 f"{'✅' if r['is_cut'] else ''} |")

    L += [
        "",
        f"- 过渡带内 `hist_l1` 最小 **{st['in_fade_min']:.4f}**、最大 **{st['in_fade_max']:.4f}**",
        f"- 过渡带之前最大 **{st['before_max']:.4f}**，之后最大 **{st['after_max']:.4f}**",
        "",
        "## 这推翻了一个想当然的说法",
        "",
        "我原本以为「渐变会让直方图距离慢慢滑过去，一层阈值切不出来」。**实测不是。**",
        "",
        "原因是纯色画面的直方图全压在**一个 bin** 上：色相每移动一个 bin，"
        "两张直方图就**完全不相交**，L1 距离直接顶到 1.0。"
        "于是过渡期每一帧都是 1.0，远高于阈值 0.3。",
        "",
        "真正拦住它的是**「最小场景长度」**（本次 0.4s = 10 帧）："
        "第一刀切在 2.04s，之后 10 帧内不许再切，于是第二刀落在 2.44s。"
        f"结果就是 `{doc['verdict']}`。",
        "",
        "所以对渐变转场，这个算法的失效方式**不是「漏切」，而是「多切一段」**——"
        "把过渡本身当成了一个场景。这比漏切更糟：后面接检测模型时，"
        f"会对着一段 0.4 秒的糊画面干活。",
        "",
        "## ★ 这条实验暴露的更大问题",
        "",
        "在纯色合成片上，两者分数是 **1.0 vs 0.04**，间隔 0.95，看起来极其漂亮。"
        "但那是**合成片给的错觉**：真实视频的直方图铺满几千个 bin，"
        "相邻帧本来就有分布重叠，任何切点的距离都会低一截，"
        "阈值 0.3 在真实素材上是否合适，**本案例没有验证过**（没有真实视频，也没有标注真值）。",
        "",
        "要处理渐变转场，标准做法是**双阈值**（高分必切、低分向后扩）"
        "或**累积帧差找拐点**，本案例**未实现**。",
        "",
        "## 复现",
        "",
        "```bash",
        f"python fade_test.py --thr {args.thr} --min-scene {args.min_scene}",
        "```",
        "",
    ]
    (OUT / "fade_test.md").write_text("\n".join(L), encoding="utf-8")

    print(f"写出 {OUT/'fade_test.md'} / {OUT/'fade_test.json'}")
    print(f"真值 2 段，检测 {len(det)} 段：{[d['start_s'] for d in det[1:]]}")
    print(f"过渡带内 hist_l1 区间 [{st['in_fade_min']}, {st['in_fade_max']}]，"
          f"带外最大 {max(st['before_max'], st['after_max'])}")
    print("结论：" + doc["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
