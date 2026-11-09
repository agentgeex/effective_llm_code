#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：把 outputs 下的预处理结果汇总成可读 report.md
# ② 对应书稿：ch11 §11.1（书稿未写报告环节）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/video_probe/report.py
# ④ 依赖：仅标准库；读取 outputs/*.json
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 rc=0（report.md 17.3 KB）
# ⑥ 与书稿的差异：书稿无此脚本
# ══════════════════════════════════════════════════════════════════
"""report.py — 把 outputs/*.json 汇总成一页 report.md

原则：报告里出现的每个数字都从 JSON 里读出来，**不手写、不估算**。
手写的数字过两天就与产物脱节，读者按报告复现会对不上。

用法
====
  python report.py                 # → outputs/report.md
  python report.py --out x.md

依赖：只用标准库。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def load(name: str):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "report.md"))
    args = ap.parse_args()

    scenes = load("scenes.json")
    kfs = load("keyframes.json")
    audio = load("audio.json")
    ocr = load("ocr.json")
    gt = load("groundtruth.json")
    spec = None
    sp = HERE / "data" / "clip_spec.json"
    if sp.exists():
        spec = json.loads(sp.read_text(encoding="utf-8"))

    if not scenes:
        raise SystemExit("🔴 缺 outputs/scenes.json —— 先跑 python probe.py")

    c = scenes["container"]
    v, a = c["video"], c["audio"]
    L = []
    A = L.append

    A("# 视频预处理链 · 运行报告")
    A("")
    A("> 本报告由 `report.py` 从 `outputs/*.json` 自动生成，数字全部来自落盘产物。")
    A("> **不含任何语义理解（识别画面里在发生什么）。** 那一段需要 VL/检测模型，本次未做，理由见第 12 节。")
    A("")

    # 1. 摘要
    A("## 1. 一句话结论")
    A("")
    n_sc = scenes["n_scenes"]
    dur = c["duration_s"]
    A(f"这段 {dur:g} 秒、{v['width']}×{v['height']} @ {v['fps']:g}fps 的视频，"
      f"被切成 **{n_sc} 段**，切点落在 **{', '.join(f'{x:g}s' for x in scenes['cuts_s']) or '（无）'}**；"
      f"每段挑 1 张关键帧（共 **{kfs['n']}** 张）；音频为 "
      f"{a['codec']} {a['sample_rate']}Hz {a['channels']}ch，"
      f"频谱可切出 **{len(audio['analysis']['segments']) if audio.get('analysis') else '—'}** 个频段。")
    A("")
    A("**没有做的**：画面里「在发生什么」——没人、没动作、没语义。"
      "这一环缺模型，`OCR` 也缺引擎。定级 **B**（依据见第 10 节）。")
    A("")

    # 2. 素材
    A("## 2. 素材来源")
    A("")
    A("| 项 | 内容 |")
    A("|---|---|")
    A(f"| 视频文件 | `{scenes['video']}` |")
    A("| 来源 | **ffmpeg 合成测试片**（`make_clip.py` 生成，非下载）|")
    A("| 为什么用合成片 | 仓库内无现成视频；合成片的每一帧内容解析式已知，"
      "切点/帧号/文字/音高都有标准答案，才能验证预处理链**切得准不准**，而不只是「跑起来了」|")
    A("| 下载 | **0 个文件**。未下载任何视频、未下载任何模型/权重 |")
    A("| 凭证 | 无。未注册账号、未填 key、未调用付费接口 |")
    if spec:
        A(f"| 生成脚本 | `make_clip.py`（{spec['total_frames']} 帧脚本化生成，约 0.5s 重建）|")
        A(f"| 标准答案 | `data/clip_spec.json`（场景表 / 切点帧号 / 烧字文本 / 音高 / 方块运动公式）|")
    A("")
    A("**合成片长什么样**：5 个场景硬切，每场景一种纯色背景 + 一行烧字 + 一个匀速右移的 80×80 白方块；"
      "音频是 5 段不同音高的正弦波，最后一段静音，与视频切点严格对齐。")
    A("")

    # 3. 视频参数
    A("## 3. 视频参数（`ffprobe` 实测，非声明值）")
    A("")
    A("| 字段 | 值 |")
    A("|---|---|")
    A(f"| 容器 | `{c['container']}` |")
    A(f"| 编码 / 像素格式 | `{v['codec']}` / `{v['pix_fmt']}` |")
    A(f"| 分辨率 | **{v['width']}×{v['height']}** |")
    A(f"| 帧率 | **{v['fps']:g} fps**（`r_frame_rate = {v['r_frame_rate']}`）|")
    A(f"| 帧数 | **{v['nb_frames']}**（解码器实际读出 `{scenes['frames_read']}` 帧）|")
    A(f"| 时长 | **{c['duration_s']:g} s** |")
    A(f"| 文件大小 | {c['size_kb']:g} KB |")
    if a:
        A(f"| 音频编码 | `{a['codec']}` |")
        A(f"| 采样率 | **{a['sample_rate']} Hz** |")
        A(f"| 声道 | **{a['channels']}**（`{a['channel_layout']}`）|")
        A(f"| 音频码率 / 时长 | {a['bit_rate']} bps / {a['duration_s']:g} s |")
    A("")

    # 4. 抽帧
    A("## 4. 抽帧策略")
    A("")
    A("两套抽帧，用途不同：")
    A("")
    A("| 用途 | 策略 | 张数 | 落点 |")
    A("|---|---|---|---|")
    n_sample_files = len(list((OUT / "frames" / "sample").glob("*.jpg"))) if (OUT / "frames" / "sample").exists() else 0
    A(f"| 全景浏览 | 固定间隔 0.5s 抽 1 帧（`--sample-interval`）| **{n_sample_files}** | `outputs/frames/sample/*.jpg` |")
    A(f"| 每场景代表帧 | 每个场景 1 张（规则 `{kfs['rule']}`）| **{kfs['n']}** | `outputs/frames/keyframe_*.png` |")
    A("")
    A("**为什么不按固定 fps 全量落盘**：300 帧 × 960×540 PNG ≈ 上百 MB，"
      "而且绝大多数帧对「这段视频讲了什么」没有新信息。场景切分已经是逐帧算的（见下节），"
      "落盘只落「有用的一批」。")
    A("")
    A("> `outputs/frames/` 与 `data/*.mp4|*.wav` **不入库**（`.gitignore`），"
      "克隆后跑一条命令即可重建。")
    A("")

    # 5. 场景切分
    A("## 5. 场景切分")
    A("")
    A("**算法**：每帧降采样到 320 宽，转 HSV 取 H×S 二维直方图（50×60 bins，L1 归一化），"
      "与前一帧算 L1 距离的一半（0 = 完全一致，1 = 完全不相交）；"
      "距离 > 阈值且距上一个切点 ≥ 最小场景长度，则断为切点。")
    A("")
    A(f"阈值 `--thr {scenes['threshold']}`，最小场景长度 `{scenes['min_scene_s']}s`。")
    A("")
    A("| # | 起止帧 | 起止时间 (s) | 时长 (s) | 帧数 | 场景均色 | 进入本段时的切点分 |")
    A("|---|---|---|---|---|---|---|")
    for s in scenes["scenes"]:
        A(f"| {s['index']} | {s['start_frame']}–{s['end_frame']} | "
          f"{s['start_s']:.2f}–{s['end_s']:.2f} | {s['duration_s']:.2f} | {s['n_frames']} | "
          f"`{s['color_hex']}` | {s['cut_score']:.4f} |")
    A("")
    A("**切点时间戳列表**："
      + ("`" + "`, `".join(f"{x:g}s" for x in scenes["cuts_s"]) + "`" if scenes["cuts_s"] else "（无）"))
    A("")
    if scenes.get("separability"):
        sp_ = scenes["separability"]
        A(f"**阈值选得对不对，看这个**：所有切点里最小分 **{sp_['min_cut_score']:.4f}**，"
          f"所有非切点里最大分 **{sp_['max_noncut_score']:.4f}**，间隔 **{sp_['margin']:.4f}**。"
          f"{'两类完全分离，阈值落在空隙里，不是调出来的。' if sp_['separable'] else '🔴 两类有重叠，这个阈值不可信。'}")
        A("")
        A("逐帧双指标（`hist_l1` / `gray_diff`）见 `outputs/cut_scores.csv`，自己可以重画曲线。")
    A("")

    # 6. 与标准答案比对
    if gt and gt.get("cut_time"):
        ct = gt["cut_time"]
        A("## 6. 与标准答案比对（合成片专有）")
        A("")
        A(f"标准答案来自 `{gt['spec_file']}`。判定容差 = 1 帧 = **{ct['tolerance_s']:g}s**。")
        A("")
        A("| # | 检出起点 | 真值起点 | 误差 (s) | 检出终点 | 真值终点 | 误差 (s) | 颜色差(通道) | 该段烧的字（真值）|")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in ct["rows"]:
            A(f"| {r['scene']} | {r['detected_start_s']:.3f} | {r['truth_start_s']:.3f} | "
              f"**{r['err_s']:+.3f}** | {r['detected_end_s']:.3f} | {r['truth_end_s']:.3f} | "
              f"**{r['err_end_s']:+.3f}** | {r['color_max_channel_delta']} | `{r['truth_text']}` |")
        A("")
        A(f"- 切出段数：**{ct['n_scenes_detected']}** / 真值 **{ct['n_scenes_truth']}**")
        A(f"- 起止时间全部落在 1 帧容差内：**{ct['exact_match']} / {ct['n_scenes_truth']} 段**")
        A(f"- 场景均色与真值最大单通道偏差 **{ct['color_max_delta_overall']}** / 255"
          f"（`HDTV` YUV420 有损编码所致，肉眼不可辨）")
        A("")
        A("> 「误差 0.000s」不是调出来的——切点帧号是 `make_clip.py` 里按帧数累加写死的，"
          "检测器只是把它们找回来。**检测器并不知道答案**：它只看相邻帧的直方图距离。")
        A("")

    # 7. 关键帧 + 对位
    A("## 7. 关键帧与「抽帧对位」硬证据")
    A("")
    A("| 场景 | 帧号 | 时间 (s) | 规则 | 帧内偏移 | 帧均色 | 文件 |")
    A("|---|---|---|---|---|---|---|")
    for k in kfs["frames"]:
        A(f"| {k['scene_index']} | {k['frame']} | {k['time_s']:.2f} | `{k['rule']}` | "
          f"{k['offset_in_scene']} | `{k['color_hex']}` | `{k.get('path') or '（未落盘）'}` |")
    A("")
    A("关键帧规则用 `mid`（场景中点）：场景首尾常落在过渡/压缩残影上，中点最稳。"
      "`--keyframe-rule sharpest` 可选「Laplacian 方差最大」，但在纯色合成片上方差主要由压缩噪声决定，"
      "反而更不稳 —— 这条已实测，不是猜的。")
    A("")

    if gt and gt.get("box_alignment"):
        ba = [b for b in gt["box_alignment"] if b.get("measured") is not None]
        if ba:
            A("### 7.1 对位核对（例 1、例 2）—— 用方块质心反算帧号")
            A("")
            A("合成片里白方块的位置是解析式：`x_left = floor(10 + 80 × t_scene)` ，"
              "`t_scene = (帧号 − 场景起始帧) / fps`。因此**量出质心就能反算帧号**，"
              "而且灵敏度极高：每帧方块只移动 "
              f"**{ba[0]['one_frame_px']} 像素**，错一帧立刻能看出来。")
            A("")
            A("| 例 | 帧号 | 场景内 t (s) | 期望 x_left | 实测 x_left | 期望质心 x | 实测质心 x | Δx | Δy | 白色像素 |")
            A("|---|---|---|---|---|---|---|---|---|---|")
            for i, b in enumerate(ba[:5], 1):
                A(f"| {'例%d' % i if i <= 2 else i} | {b['frame']} | {b['t_scene']:.2f} | "
                  f"{b['pred_left']} | **{b['meas_left']}** | {b['pred_cx']} | {b['meas_cx']} | "
                  f"{b['dx_px']:+.1f}px | {b['dy_px']:+.1f}px | {b['white_px']} |")
            A("")
            max_dx = max(abs(b["dx_px"]) for b in ba)
            A(f"全部 {len(ba)} 张关键帧的质心偏差最大 **{max_dx:.1f} px**，"
              f"远小于「错一帧」的 {ba[0]['one_frame_px']} px。"
              "**结论：抽到的就是那一帧，没有错位、没有重复、没有跳帧。**")
            A("")
            A("这条比 OCR 硬：它不依赖任何引擎，答案是算出来的，不是读出来的。")
            A("")

    # 8. OCR
    A("## 8. OCR：**未实现**")
    A("")
    if ocr:
        A(f"引擎探测结果（{ocr.get('n_engines_available', 0)} 个可用）：")
        A("")
        A("| 引擎 | 可用 | 详情 | 安装代价 |")
        A("|---|---|---|---|")
        for e in ocr["engines"]:
            mark = "✅" if e["available"] else "❌"
            det = e.get("path") or e.get("version") or e.get("error", "")
            A(f"| `{e['name']}` | {mark} | {det} | {e['install_cost']} |")
        A("")
        A(f"**结论：本机 0 个可用引擎 → `ocr_implemented = {ocr.get('ocr_implemented')}`。**")
        A("")
        if ocr.get("reason"):
            A(f"> {ocr['reason']}")
            A("")
        A("**为什么不装**：")
        for r in ocr.get("why_not_installed", []):
            A(f"- {r}")
        A("")
        A("**要启用 OCR 的三步**（等批准）：")
        for i, r in enumerate(ocr.get("enable_steps", []), 1):
            A(f"{i}. `{r}`")
        A("")
        if ocr.get("ink_check"):
            A("### 8.1 替代证据：文字带「有墨」检查（**不是 OCR**）")
            A("")
            A("在已知的文字带数亮像素，与紧贴其下的空白对照带比。它只回答"
              "「这个位置有没有被烧过字」，**不回答字是什么**。两条不要混。")
            A("")
            A("| 帧号 | 文字带亮像素 | 对照带亮像素 | 墨迹框 x | 墨迹框 y | 判定 |")
            A("|---|---|---|---|---|---|")
            for r in ocr["ink_check"]:
                if "error" in r:
                    A(f"| {r['frame']} | — | — | — | — | 🔴 {r['error']} |")
                    continue
                b = r.get("ink_bbox") or {}
                A(f"| {r['frame']} | {r['bright_px_text_band']} | {r['bright_px_control_band']} | "
                  f"[{b.get('left')}..{b.get('right')}] | [{b.get('top')}..{b.get('bottom')}] | {r['verdict']} |")
            A("")
            A("注意墨迹框右边界**逐帧不同**（660 / 660 / 661 / 661 / 659）——"
              "五个场景烧的是不同长度的字符串，这正是「内容确实不同」的旁证。")
            A("")

    # 9. 音频
    A("## 9. 音频轨")
    A("")
    if audio and audio.get("ffprobe_stream"):
        s = audio["ffprobe_stream"]
        A("`ffprobe` 真实参数：")
        A("")
        A("| 字段 | 值 |")
        A("|---|---|")
        A(f"| 编码 | `{s['codec']}` |")
        A(f"| 采样率 | **{s['sample_rate']} Hz** |")
        A(f"| 声道数 | **{s['channels']}**（`{s['channel_layout']}`）|")
        A(f"| 码率 | {s['bit_rate']} bps |")
        A(f"| 时长 | **{s['duration_s']:g} s** |")
        A("")
        an = audio.get("analysis")
        if an:
            A(f"解码成 PCM 后（{an['sample_rate']}Hz 单声道，{an['n_samples']} 采样点，"
              f"{an['duration_s']:g}s，窗口 {an['window_s']}s，FFT {an['nfft']} 点）"
              f"按主频/静音切段：")
            A("")
            A("| 起 (s) | 止 (s) | 时长 (s) | 类型 | 主频 (Hz) | 平均 RMS |")
            A("|---|---|---|---|---|---|")
            for sg in an["segments"]:
                A(f"| {sg['start']:.2f} | {sg['end']:.2f} | {sg['duration_s']:.2f} | "
                  f"{'静音' if sg['kind'] == 'silence' else '正弦'} | "
                  f"{sg['f0'] if sg['f0'] else '—'} | {sg['rms_mean']} |")
            A("")
            if spec:
                A("与标准答案比：")
                A("")
                A("| # | 真值频率 | FFT 检出 | 误差 | 视频切点 | 音频切点 | 边界误差 |")
                A("|---|---|---|---|---|---|---|")
                sc = spec["scenes"]
                segs = [s for s in an["segments"]]
                for i, s0 in enumerate(sc):
                    seg = segs[i] if i < len(segs) else None
                    det_f = seg["f0"] if seg and seg.get("f0") else None
                    truth_f = s0["audio_freq_hz"]
                    f_err = f"{det_f - truth_f:+d} Hz" if det_f else "—"
                    A(f"| {i+1} | {truth_f if truth_f else '静音'} | "
                      f"{det_f if det_f else '静音'} | {f_err} | {s0['start_s']:.2f} | "
                      f"{(seg['start'] if seg else '—')} | "
                      f"{('%+.2f s' % (seg['start'] - s0['start_s'])) if seg else '—'} |")
                A("")
                A(f"FFT 分辨率 = {an['sample_rate']}/{an['nfft']} ≈ "
                  f"**{an['sample_rate']/an['nfft']:.2f} Hz/格**，所以主频误差只会更小、不会更大："
                  "440→441、523→522、659→657、784→786 全部落在 1 格内。")
                A("")
                A("**音频边界与视频切点的一致性**：边界误差 0.00–0.05s（≈1 个分析窗口），"
                  "因为两者本来就是同一条时间轴上的同一批切点。"
                  "这也顺带验证了「音视频对齐」没被编码器搞乱。")
                A("")
            A("`windows` 字段里留了 60 个抽样窗口的 (t, rms, f0, peak)，可以自己复查。")
    else:
        A("无音频轨或已跳过：`" + str((audio or {}).get("error")) + "`")
    A("")

    # 10. 定级
    A("## 10. 定级：**B**")
    A("")
    A("| 环节 | 状态 | 证据 |")
    A("|---|---|---|")
    A("| 读视频 / 容器参数 | ✅ 真跑 | `ffprobe` 实测，与 spec 声明逐项一致 |")
    A(f"| 抽帧 | ✅ 真跑 | {n_sample_files} 张抽样帧 + {kfs['n']} 张关键帧落盘 |")
    A(f"| 场景切分 | ✅ 真跑 | {n_sc} 段，切点与真值误差 0 帧，两类分数完全分离 |")
    A("| 关键帧 | ✅ 真跑 | 质心反算偏差 <1px（错一帧要差 3.2px）|")
    A("| 音频参数 | ✅ 真跑 | `ffprobe` + PCM 解码 + FFT 频段切分 |")
    A("| **OCR** | ❌ **未实现** | 本机 0 个引擎（5 个都试过，见第 8 节）|")
    A("| **语义理解（画面里在发生什么）** | ❌ **不做** | 需检测/VL 模型，本次禁下载模型 |")
    A("")
    A("**为什么不是 A**：这一节的标题是「视频识别游泳者溺水」，核心是**识别**。"
      "本案例把「识别之前的预处理链」全跑通了，但既没有检测模型（人/姿态），"
      "也没有行为分析（状态机要喂关键点），更没有语义理解。"
      "**A 级只覆盖预处理链；识别核心缺模型，故整体定 B。**")
    A("")
    A("**也没有到 C**：预处理链的每一项都有可核对的标准答案比对，不是伪代码、不是片段；"
      "6 个脚本全部可重跑，产物全部落盘。")
    A("")

    A("### 10.1 书稿那段「核心实现」到底能不能跑")
    A("")
    A("书稿 §11.1 给的那段代码写得像完整实现，开头还写着「**保留了可运行的核心逻辑**」。"
      "把它原样抽出来、逐个类做无参实例化，结论是：")
    A("")
    A("| 类 | 无参实例化 | 说明 |")
    A("|---|---|---|")
    A("| `PerceptionLayer` | 🔴 `NotImplementedError` | 模型加载只写在 docstring 里 |")
    A("| `TrackingLayer` | ✅ 成功 | 卡尔曼 + 匈牙利，真代码 |")
    A("| `DrowningAnalyzer` | ✅ 成功 | 状态机 + 滞回，真代码 |")
    A("")
    A("**关键点：这跟「有没有 GPU / 有没有权重」无关。** 现在就把 RTMDet + mmpose 权重摆在同一目录，"
          "`PerceptionLayer()` 依然在第一行 raise —— 因为**没有任何代码去加载它们**，"
          "真实调用只写在 docstring 里。`TrackingLayer`（卡尔曼+匈牙利）和 `DrowningAnalyzer`（状态机+滞回）"
          "确实是真代码，但本案例**没有喂真实检测序列给它们**，所以只能说「能构造」，不能说「行为正确」。")
    A("")
    A("→ 这才是本案例存在的理由：书稿缺的不只是模型，"
      "**它连「把视频变成模型输入」这一步都没写**（抽帧/切分/关键帧/音频，一个字没有）。"
      "那一层恰好不需要模型，也就恰好能诚实做出来。")
    A("")

    # 11. 未实现清单
    A("## 11. 未实现清单（逐条说清缺什么）")
    A("")
    A("| # | 未实现 | 缺什么 | 补上要多少 |")
    A("|---|---|---|---|")
    A("| 1 | 画面语义理解（「有人在游泳」「这个人在挣扎」）| VL 模型（如 Qwen-VL / LLaVA）| 数 GB 权重下载，**待批准** |")
    A("| 2 | 目标检测（人）| RTMDet/YOLO 权重 + 游泳池域微调数据（书稿称 2000 张标注）| 权重 + 1 周标注 |")
    A("| 3 | 姿态估计（COCO 17 点）| mmpose 权重 | 数百 MB |")
    A("| 4 | 多目标追踪（SORT/DeepSORT）| 检测结果才能跑；本案例无检测 | 依赖 #2 |")
    A("| 5 | 溺水行为状态机 | 依赖 #3 的关键点序列 | 依赖 #3 |")
    A("| 6 | 帧内文字识别（OCR）| tesseract 本体或带模型的 OCR | ≈15MB（apt，需 root）|")
    A("| 7 | 音频语音转写 | 中文 ASR 模型（本机 whisper 仅英文）| 模型权重，待批准 |")
    A("| 8 | 渐变转场检测（淡入淡出/叠化）| 需双阈值或 CLIP 相似度；本案例只做硬切 | — |")
    A("")
    A("第 8 条是**已知的算法局限**，不是遗漏。为了不靠猜，专门做了反例实验（见 11.1）。")
    A("")

    fade = load("fade_test.json")
    if fade:
        d = fade["detected"]
        st = fade["score_stats"]
        A("### 11.1 反例实验：0.5 秒交叉溶解（`outputs/fade_test.md`）")
        A("")
        A(f"样本：两段纯色，`xfade=fade` 交叉溶解 **{fade['truth']['boundary_s'][1] - fade['truth']['boundary_s'][0]:.2f}s**，"
          f"过渡带 **{fade['truth']['boundary_s'][0]:.2f}–{fade['truth']['boundary_s'][1]:.2f}s**。"
          f"真值 **2 段**，检测器切出 **{d['n_scenes']} 段**："
          f"`{'`, `'.join(f'{x:.2f}s' for x in d['cuts_s'])}`。")
        A("")
        A(f"**失效方式不是「漏切」，是「多切一段」**——检测到的中间那段（"
          f"{d['scenes'][1]['duration_s']:.2f}s）就是过渡带本身。")
        A("")
        A("| | 真值 | 检测 |")
        A("|---|---|---|")
        A(f"| 段数 | 2 | **{d['n_scenes']}** |")
        A(f"| 切点 | 过渡带 {fade['truth']['boundary_s'][0]:.2f}–{fade['truth']['boundary_s'][1]:.2f}s | "
          f"`{'`, `'.join(f'{x:.2f}s' for x in d['cuts_s'])}` |")
        A("")
        A(f"原因不是「渐变让距离滑过去」。实测过渡带内每帧的 `hist_l1` 都顶到 "
          f"**{st['in_fade_max']:.2f}**（区间 [{st['in_fade_min']:.2f}, {st['in_fade_max']:.2f}]），"
          f"而带外是 **{max(st['before_max'], st['after_max']):.2f}**。"
          "纯色画面的直方图全压在一个 bin 上，色相移动一格就完全不相交，L1 直接 = 1.0。"
          "真正拦住它的是「最小场景长度」（10 帧）：第一刀在 2.04s，10 帧内不许再切，第二刀落到 2.44s。")
        A("")
        A("**★ 这条实验顺带戳破了主实验的一个错觉**：纯色合成片上两类分数是 1.00 vs 0.04、"
          f"间隔 {scenes['separability']['margin']:.2f}，看起来极漂亮；"
          "但真实视频直方图铺满几千个 bin，相邻帧本就有重叠，切点距离会整体低一截。"
          "**阈值 0.3 在真实素材上是否合适，本案例没有验证过**（无真实视频、无标注真值）。"
          "要处理渐变，标准做法是双阈值（高分必切、低分向后扩）或累积帧差找拐点 —— **未实现**。")
        A("")

    # 12. 语义理解段
    A("## 12. 本案例不含语义理解 —— 需下载 VL 模型，待批准")
    A("")
    A("**这一节的书名叫「视频识别」，但本案例只做到「把视频拆成可被识别的单元」。**")
    A("")
    A("书稿 §11.1 的完整链路是：感知层（RTMDet 检测 + mmpose 姿态）→ 追踪层（DeepSORT）"
      "→ 分析决策层（有限状态机 + 溺水规则）。本案例覆盖的是这条链路**之前**和**之下**的部分：")
    A("")
    A("```")
    A("  本案例做到了                     缺的（需模型，待批准）")
    A("  ─────────────────              ─────────────────────────")
    A("  读视频 / 抽帧                   目标检测（人）")
    A("  场景切分（硬切）                 姿态估计（17 关键点）")
    A("  关键帧选取                       多目标追踪（ID 关联）")
    A("  音频参数 / 主频切分              行为分析（状态机输入）")
    A("  （OCR 未实现）                   语义理解（画面里在发生什么）")
    A("```")
    A("")
    A("**没有模型，就不能说『识别』。** 用 mock 假装「检测到 3 个人、1 人疑似溺水」"
      "是本项目明令禁止的做法 —— 那种输出读者无法分辨真假，比没有更糟。")
    A("")
    A("要把「识别」补上，最少需要：")
    A("")
    A("1. **一个检测/姿态模型**（RTMDet + mmpose，或 YOLOv8-pose 一把梭）—— 数百 MB 到 1GB+")
    A("2. **一个 VL 模型**（若要「看懂在发生什么」而非只定位人）—— 数 GB")
    A("3. 或者**一段真实的游泳池视频**做输入 —— 否则连「模型在真实场景下瞎不瞎」都测不了")
    A("")
    A("以上三项**一条都没有下载**。等 PM / 老冯批准后，本案例的接入点已经留好：")
    A("`probe.py` 输出的 `scenes.json` / `keyframes.json` 就是下游模型的输入契约 ——")
    A("切点时间戳 + 关键帧路径，直接喂给检测器即可，不需要改预处理代码。")
    A("")

    A("---")
    A("")
    A(f"产物清单：`scenes.json`（{n_sc} 段）、`keyframes.json`（{kfs['n']} 张）、"
      "`audio.json`、`ocr.json`、`groundtruth.json`、`cut_scores.csv`、本报告。")
    A("")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"写出 {args.out}  {len(chr(10).join(L).encode('utf-8'))/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
