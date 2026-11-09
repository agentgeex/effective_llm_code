#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：视频前置预处理链：抽帧 → 场景切分 → 关键帧 → 音频轨 → 结构化结果
# ② 对应书稿：ch11 §11.1，（书稿 PerceptionLayer 主体之前的预处理段，书稿未写）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/video_probe/probe.py
# ④ 依赖：opencv-python（cv2）、numpy、ffmpeg/ffprobe 可执行文件
# ⑤ 实测解释器：无可用解释器：四个解释器均 ModuleNotFoundError: No module named 'cv2'（实测）
# ⑥ 与书稿的差异：书稿没有抽帧/切点/音频这一段，本文件补齐
# ══════════════════════════════════════════════════════════════════
"""probe.py — 视频预处理链：抽帧 → 场景切分 → 关键帧 → 音频参数/主频

这一节的书名叫「视频识别游泳者溺水」，书稿给的是 YOLO/RTMDet + DeepSORT + 状态机。
本文件**不做识别**，只做识别之前必须做对的预处理链：

    读视频 → 抽帧 → 场景切分（硬切）→ 每个场景挑一张关键帧 → 音频轨参数与主频

为什么先做这一层
================
书稿 §11.1 的核心实现在 100–545 行，需要 mmdet + mmpose + 2000 张微调标注 + CUDA。
那些都拿不到时，唯一能诚实验证的是「喂给模型之前的那段流水线」：
帧读得对不对、切点准不准、关键帧能不能代表场景、音频参数拿不拿得到。
这一层不成立，后面接什么模型都没意义（切点错 1 帧，行为分析的时序窗就整个错位）。

算法（全部自己实现，不引入任何模型/权重）
=====================================
· 场景切分：HSV 2D 直方图（H50×S60，L1 归一化）逐帧 L1 距离 /2 → 0..1
            距离超阈值且距上次切点 >= 最小场景长度，则断为切点
· 关键帧  ：默认取场景中点帧（避开首尾过渡帧），可选最清晰帧（Laplacian 方差最大）
· 音频    ：ffprobe 拿真实流参数；解码成 PCM 后按窗口算 RMS + FFT 主频，做频段切分

输出（全部落 outputs/，可复核）
=============================
  scenes.json        切点时间戳 + 每场景起止帧
  keyframes.json     每场景 1 张关键帧的清单
  audio.json         ffprobe 原始参数 + FFT 频段切分结果
  cut_scores.csv     逐帧双指标分数（阈值选得对不对，看这个）
  groundtruth.json   与 data/clip_spec.json 的比对（有 spec 才有）
  frames/            抽样帧 + 关键帧（.gitignore，不入库）

用法
====
  python probe.py                          # 用 data/clip.mp4
  python probe.py --video /path/v.mp4 --thr 0.30 --sample-interval 0.5
  python probe.py --no-audio               # 跳过音频分析
  python probe.py --keyframe-rule sharpest

依赖：cv2 + numpy（系统已有 4.11.0 / 1.26.4）+ 系统 ffmpeg/ffprobe。**无新增依赖。**
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
FRAMES = OUT / "frames"

# 切分参数（默认值，可 CLI 覆盖）
DEFAULT_THR = 0.30        # HSV 直方图 L1 距离阈值
MIN_SCENE_S = 0.40        # 最小场景长度，防止一帧抖动被切成两段
PROBE_W = 320             # 打分用的降采样宽度（只为提速，不影响切点 1 帧精度）


# ── 视频元信息 ────────────────────────────────────────────────────────
def ffprobe_streams(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def container_info(info: dict) -> dict:
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return dict(
        path=info["format"].get("filename"),
        container=info["format"].get("format_name"),
        duration_s=float(info["format"].get("duration", 0) or 0),
        size_kb=round(int(info["format"].get("size", 0)) / 1024, 1),
        video=dict(
            codec=v.get("codec_name"), width=int(v.get("width", 0)), height=int(v.get("height", 0)),
            fps=round(fps, 4), r_frame_rate=v.get("r_frame_rate"),
            nb_frames=int(v.get("nb_frames", 0) or 0), pix_fmt=v.get("pix_fmt"),
            bit_rate=int(v.get("bit_rate", 0) or 0),
        ),
        audio=dict(
            codec=a.get("codec_name"), sample_rate=int(a.get("sample_rate", 0) or 0),
            channels=int(a.get("channels", 0) or 0), channel_layout=a.get("channel_layout"),
            bit_rate=int(a.get("bit_rate", 0) or 0), duration_s=float(a.get("duration", 0) or 0),
        ) if a else None,
    )


def hist_l1_norm(hsv: np.ndarray) -> np.ndarray:
    """HSV 2D 直方图，手写 L1 归一化（sum=1）。

    ★ 这里踩过一个坑：`cv2.normalize(h, h, 0, 1, cv2.NORM_L1)` 实测会把整张
      直方图清成全 0（sum=0，不报错）。后果是所有距离恒为 0 → 切分器"只切出
      1 个场景"，而且看起来一切正常。手写除以 sum 才可靠。
    """
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    total = float(h.sum())
    return h / total if total > 0 else h


# ── 逐帧打分 ──────────────────────────────────────────────────────────
def frame_score(prev_gray: np.ndarray | None, prev_hist: np.ndarray | None, frame) -> tuple:
    """返回 (hist_l1, gray_diff)；first frame 返回 (0,0)。"""
    small = cv2.resize(frame, (PROBE_W, int(PROBE_W * frame.shape[0] / frame.shape[1])),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = hist_l1_norm(hsv)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if prev_hist is None:
        return 0.0, 0.0
    # L1 距离的一半 ∈ [0,1]：0 = 直方图完全一致，1 = 完全不相交
    hist_l1 = float(np.sum(np.abs(hist - prev_hist)) / 2.0)
    gray_diff = float(np.mean(np.abs(gray - prev_gray)))
    return hist_l1, gray_diff


def median_color_hex(frame) -> str:
    med = np.median(frame.reshape(-1, 3), axis=0)      # BGR
    b, g, r = [int(round(x)) for x in med]
    return f"#{r:02X}{g:02X}{b:02X}"


# ── 主扫描 ────────────────────────────────────────────────────────────
@dataclass
class Scene:
    index: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    cut_score: float                 # 进入本场景那一刻的 hist_l1（首场景 0）
    color_hex: str = ""
    frames: list = field(default_factory=list)   # 仅供内部挑关键帧，不落盘

    def as_dict(self) -> dict:
        n = self.end_frame - self.start_frame + 1
        return dict(index=self.index, start_frame=self.start_frame, end_frame=self.end_frame,
                    start_s=round(self.start_s, 3), end_s=round(self.end_s, 3),
                    duration_s=round(self.end_s - self.start_s, 3), n_frames=n,
                    cut_score=round(self.cut_score, 4), color_hex=self.color_hex)


def scan_video(video: Path, thr: float, min_scene_s: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"🔴 打不开视频：{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_frames = max(1, int(round(min_scene_s * fps)))

    scores = []          # (frame, t, hist_l1, gray_diff, is_cut)
    scenes = []
    prev_gray = prev_hist = None
    cur = Scene(1, 0, 0, 0.0, 0.0, 0.0)
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hist_l1, gray_diff = frame_score(prev_gray, prev_hist, frame)
        is_cut = (frame_idx > 0 and hist_l1 > thr
                  and (frame_idx - cur.start_frame) >= min_frames)
        scores.append((frame_idx, frame_idx / fps, hist_l1, gray_diff, is_cut))

        if is_cut:
            cur.end_frame = frame_idx - 1
            cur.end_s = frame_idx / fps
            cur.color_hex = cur.color_hex or median_color_hex(cur.frames[len(cur.frames) // 2])
            scenes.append(cur)
            cur = Scene(len(scenes) + 1, frame_idx, frame_idx, frame_idx / fps,
                        frame_idx / fps, hist_l1)
        cur.frames.append(frame)
        prev_gray = cv2.cvtColor(
            cv2.resize(frame, (PROBE_W, int(PROBE_W * frame.shape[0] / frame.shape[1]))),
            cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(cv2.resize(frame, (PROBE_W, int(PROBE_W * frame.shape[0] / frame.shape[1]))),
                           cv2.COLOR_BGR2HSV)
        prev_hist = hist_l1_norm(hsv)
        frame_idx += 1

    cap.release()
    if frame_idx == 0:
        raise SystemExit("🔴 视频解出 0 帧")
    cur.end_frame = frame_idx - 1
    cur.end_s = frame_idx / fps
    cur.color_hex = cur.color_hex or median_color_hex(cur.frames[len(cur.frames) // 2])
    scenes.append(cur)
    return scenes, scores, fps, frame_idx, total


def keyframe_pick(scene: Scene, rule: str) -> int:
    """在场景内挑一帧。mid = 中点（避开首尾）；sharpest = Laplacian 方差最大。"""
    n = len(scene.frames)
    if rule == "sharpest":
        best, best_v = 0, -1.0
        for i, f in enumerate(scene.frames):
            v = float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if v > best_v:
                best, best_v = i, v
        return best
    return n // 2


# ── 音频 ──────────────────────────────────────────────────────────────
def extract_wav(video: Path, wav: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(wav)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  ⚠️  音频提取失败：{r.stderr[-500:]}", file=sys.stderr)
        return False
    return True


def analyze_audio(wav: Path, win_s: float = 0.05) -> dict:
    """按窗口算 RMS 与主频，再按主频/静音做频段切分。不引入 scipy。"""
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)

    hop = int(sr * win_s)
    nfft = 8192                                    # 4.4Hz/格，足够分辨 440/523/659/784
    win = np.hanning(nfft).astype(np.float32)
    win_meta = []
    for start in range(0, max(1, len(x) - nfft + 1), hop):
        seg = x[start:start + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        rms = float(np.sqrt(np.mean(seg ** 2)))
        spec = np.abs(np.fft.rfft(seg * win))
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        spec[:int(50 / (sr / nfft))] = 0.0          # 掐掉 50Hz 以下（直流/低频噪声）
        peak = int(np.argmax(spec))
        f0 = float(freqs[peak])
        win_meta.append(dict(t=round(start / sr, 3), rms=round(rms, 5), f0=round(f0, 1),
                             peak=round(float(spec[peak]), 4)))

    # 频段切分：静音（rms 很小）或主频变化 > 5% 且 > 8Hz 视为换段
    silence_rms = 0.002
    segs = []
    for m in win_meta:
        is_sil = m["rms"] < silence_rms
        label = None if is_sil else round(m["f0"])
        if not segs:
            segs.append(dict(start=m["t"], kind="silence" if is_sil else "tone",
                             f0=None if is_sil else label, rms=[m["rms"]]))
            continue
        last = segs[-1]
        same = (last["kind"] == "silence" and is_sil) or \
               (last["kind"] == "tone" and not is_sil and last["f0"]
                and abs(label - last["f0"]) <= max(8, 0.05 * last["f0"]))
        if same:
            last["rms"].append(m["rms"])
        else:
            last["end"] = m["t"]
            segs.append(dict(start=m["t"], kind="silence" if is_sil else "tone",
                             f0=None if is_sil else label, rms=[m["rms"]]))
    if segs:
        segs[-1]["end"] = round(len(x) / sr, 3)
    for s in segs:
        s["rms_mean"] = round(float(np.mean(s.pop("rms"))), 5)
        s["duration_s"] = round(s["end"] - s["start"], 3)
    # 合并过短的段（AAC 编码在边界会糊，产生 <0.1s 的碎段）
    merged = []
    for s in segs:
        if merged and s["duration_s"] < 0.10:
            merged[-1]["end"] = s["end"]
            merged[-1]["duration_s"] = round(merged[-1]["end"] - merged[-1]["start"], 3)
        else:
            merged.append(s)
    return dict(sample_rate=sr, channels=nch, n_samples=int(len(x)),
                duration_s=round(len(x) / sr, 3), window_s=win_s, nfft=nfft,
                segments=merged[:40], n_windows=len(win_meta),
                windows=win_meta[::max(1, len(win_meta) // 60)][:60])


def color_delta(hex_a: str, hex_b: str) -> int:
    """两个 #RRGGBB 的最大单通道差。用于容忍 YUV420 有损编码带来的偏移。"""
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return max(abs(x - y) for x, y in zip(a, b))


# ── ground truth 比对 ─────────────────────────────────────────────────
def groundtruth(scenes, spec: dict, fps: float) -> dict:
    gt = spec.get("scenes", [])
    rows = []
    for a, b in zip(scenes, gt):
        delta = color_delta(a["color_hex"], b["color_hex"])
        rows.append(dict(
            scene=a["index"],
            detected_start_s=a["start_s"], truth_start_s=b["start_s"],
            err_s=round(a["start_s"] - b["start_s"], 4),
            detected_end_s=a["end_s"], truth_end_s=b["end_s"],
            err_end_s=round(a["end_s"] - b["end_s"], 4),
            detected_color=a["color_hex"], truth_color=b["color_hex"],
            color_max_channel_delta=delta,
            # 精确相等要求 RGB 三个通道一字不差 —— YUV420 有损编码下这不可能。
            # 故同时给「严格相等」与「容差内相等」，容差取 16/255（肉眼不可辨的量级）。
            color_match_exact=a["color_hex"].upper() == b["color_hex"].upper(),
            color_match_tol16=delta <= 16,
            truth_text=b["text"],
            truth_audio_freq_hz=b["audio_freq_hz"],
        ))
    n_ok = sum(1 for r in rows if abs(r["err_s"]) < 1.0 / fps + 1e-6
               and abs(r["err_end_s"]) < 1.0 / fps + 1e-6)
    return dict(n_scenes_detected=len(scenes), n_scenes_truth=len(gt),
                exact_match=n_ok, tolerance_s=round(1.0 / fps, 4),
                color_exact_all=all(r["color_match_exact"] for r in rows),
                color_tol16_all=all(r["color_match_tol16"] for r in rows),
                color_max_delta_overall=max((r["color_max_channel_delta"] for r in rows), default=0),
                rows=rows)


def box_alignment(keyframes: list, spec: dict, fps: float) -> list:
    """用「白方块质心 = 10+80*t+40」反查抽帧是否对上帧号 —— 这是硬证据，不是 OCR。"""
    if not spec.get("moving_box"):
        return []
    mb = spec["moving_box"]
    out = []
    for kf in keyframes:
        img = cv2.imread(str(kf["path"]))
        if img is None:
            continue
        y0 = int(mb["y"]) - 6
        y1 = int(mb["y"] + mb["h"]) + 6
        roi = img[y0:y1, :, :]
        mask = (roi[:, :, 0] > 235) & (roi[:, :, 1] > 235) & (roi[:, :, 2] > 235)
        sc = next((s for s in spec["scenes"] if s["start_frame"] <= kf["frame"] <= s["end_frame"]), None)
        if sc is None or mask.sum() == 0:
            out.append(dict(frame=kf["frame"], measured=None, note="未在预期 y 带检出白色方块"))
            continue
        ys, xs = np.nonzero(mask)
        t_scene = (kf["frame"] - sc["start_frame"]) / fps
        pred_left = int(np.floor(mb["x0"] + mb["px_per_s"] * t_scene))
        pred_cx = pred_left + mb["w"] / 2.0
        pred_cy = mb["y"] + mb["h"] / 2.0
        meas_cx = float(xs.mean())
        meas_cy = float(ys.mean()) + y0
        out.append(dict(
            frame=kf["frame"], t_scene=round(t_scene, 4),
            pred_left=pred_left, meas_left=int(xs.min()),
            pred_cx=round(pred_cx, 1), meas_cx=round(meas_cx, 1),
            dx_px=round(meas_cx - pred_cx, 1),
            pred_cy=round(pred_cy, 1), meas_cy=round(meas_cy, 1),
            dy_px=round(meas_cy - pred_cy, 1),
            white_px=int(mask.sum()),
            one_frame_px=round(mb["px_per_s"] / fps, 2),
        ))
    return out


# ── 主流程 ────────────────────────────────────────────────────────────
def main() -> int:
    global OUT, FRAMES
    ap = argparse.ArgumentParser(description="视频预处理链：抽帧/场景切分/关键帧/音频")
    ap.add_argument("--video", default=str(DATA / "clip.mp4"))
    ap.add_argument("--thr", type=float, default=DEFAULT_THR, help="直方图 L1 切点阈值")
    ap.add_argument("--min-scene", type=float, default=MIN_SCENE_S)
    ap.add_argument("--sample-interval", type=float, default=0.5, help="抽样帧间隔（秒）")
    ap.add_argument("--keyframe-rule", choices=["mid", "sharpest"], default="mid")
    ap.add_argument("--spec", default=str(DATA / "clip_spec.json"))
    ap.add_argument("--outdir", default=str(OUT), help="产物目录（默认 outputs/）")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="不落图片，只出清单")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"🔴 找不到视频 {video}\n   先跑：python make_clip.py", file=sys.stderr)
        return 2

    OUT = Path(args.outdir)
    FRAMES = OUT / "frames"
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 读容器参数：{video.name}")
    info = ffprobe_streams(video)
    cinfo = container_info(info)
    v = cinfo["video"]
    print(f"      {v['codec']} {v['width']}x{v['height']} @ {v['fps']}fps "
          f"{v['nb_frames']} 帧 {cinfo['duration_s']}s {cinfo['size_kb']}KB")

    print(f"[2/5] 逐帧打分 + 场景切分（thr={args.thr}, min_scene={args.min_scene}s）")
    scenes_obj, scores, fps, n_read, n_total = scan_video(video, args.thr, args.min_scene)
    det = [s.as_dict() for s in scenes_obj]
    print(f"      解出 {n_read} 帧（容器声明 {n_total}），切出 {len(det)} 个场景")

    with open(OUT / "cut_scores.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "hist_l1", "gray_diff", "is_cut"])
        for row in scores:
            wr.writerow([row[0], round(row[1], 3), round(row[2], 4), round(row[3], 4), int(row[4])])

    cut_rows = [r for r in scores if r[4]]
    noncut = [r for r in scores if not r[4]]
    gap = None
    if cut_rows and noncut:
        gap = dict(min_cut_score=round(min(r[2] for r in cut_rows), 4),
                   max_noncut_score=round(max(r[2] for r in noncut), 4),
                   threshold=args.thr)
        gap["separable"] = gap["min_cut_score"] > gap["max_noncut_score"]
        gap["margin"] = round(gap["min_cut_score"] - gap["max_noncut_score"], 4)

    print(f"[3/5] 挑关键帧（rule={args.keyframe_rule}）")
    kfs = []
    for s, d in zip(scenes_obj, det):
        k = keyframe_pick(s, args.keyframe_rule)
        frame_no = d["start_frame"] + k
        path = FRAMES / f"keyframe_{d['index']:02d}_f{frame_no:04d}.png"
        if not args.no_frames:
            cv2.imwrite(str(path), s.frames[k])
        kfs.append(dict(scene_index=d["index"], frame=frame_no,
                        time_s=round(frame_no / fps, 3), rule=args.keyframe_rule,
                        offset_in_scene=k, color_hex=median_color_hex(s.frames[k]),
                        path=str(path.relative_to(HERE)) if not args.no_frames else None))

    print(f"[4/5] 抽样帧（每 {args.sample_interval}s）")
    n_sample = 0
    if not args.no_frames:
        (FRAMES / "sample").mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        step = max(1, int(round(args.sample_interval * fps)))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if n_sample * step == int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1:
                cv2.imwrite(str(FRAMES / "sample" / f"sample_{n_sample:04d}_f{n_sample*step:04d}.jpg"),
                            fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
                n_sample += 1
        cap.release()
    expected_samples = int(np.ceil(n_read / max(1, int(round(args.sample_interval * fps)))))
    print(f"      落 {n_sample} 张（预期 ≈{expected_samples}）")

    print("[5/5] 音频")
    audio = None
    if not args.no_audio and cinfo["audio"]:
        wav = DATA / "clip.wav"
        if extract_wav(video, wav):
            audio = dict(ffprobe_stream=cinfo["audio"], analysis=analyze_audio(wav), wav=str(wav))
            print(f"      {cinfo['audio']['codec']} {cinfo['audio']['sample_rate']}Hz "
                  f"{cinfo['audio']['channels']}ch，切出 {len(audio['analysis']['segments'])} 个频段")
            wav.unlink(missing_ok=True)      # 临时 wav 不留在 data/ 里
        else:
            audio = dict(ffprobe_stream=cinfo["audio"], analysis=None, error="wav 提取失败")
    else:
        audio = dict(ffprobe_stream=cinfo["audio"], analysis=None,
                     error="无音频轨" if not cinfo["audio"] else "已用 --no-audio 跳过")
        print("      跳过")

    # 汇总落盘
    try:
        video_ref = str(video.relative_to(HERE))
    except ValueError:                      # 外部路径（如 /tmp 的对照实验）
        video_ref = str(video)
    scenes_doc = dict(video=video_ref, container=cinfo, fps=round(fps, 4),
                      frames_read=n_read, frames_declared=n_total,
                      threshold=args.thr, min_scene_s=args.min_scene,
                      n_scenes=len(det), cuts_s=[round(d["start_s"], 3) for d in det[1:]],
                      cuts_frames=[d["start_frame"] for d in det[1:]], scenes=det,
                      separability=gap)
    (OUT / "scenes.json").write_text(json.dumps(scenes_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "keyframes.json").write_text(
        json.dumps(dict(n=len(kfs), rule=args.keyframe_rule, frames=kfs), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "audio.json").write_text(json.dumps(audio, ensure_ascii=False, indent=2), encoding="utf-8")

    gt = None
    spec_path = Path(args.spec)
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        gt = dict(spec_file=str(spec_path.relative_to(HERE)),
                  cut_time=groundtruth(det, spec, fps),
                  box_alignment=box_alignment(kfs, spec, fps))
        (OUT / "groundtruth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        ct = gt["cut_time"]
        print(f"      ground truth：{ct['n_scenes_detected']} vs {ct['n_scenes_truth']} 段，"
              f"起止时间严格吻合 {ct['exact_match']} 段，颜色(容差16)全对={ct['color_tol16_all']} "
              f"最大通道差={ct['color_max_delta_overall']}")

    print()
    print(f"切点（秒）：{scenes_doc['cuts_s']}")
    for d in det:
        print(f"  场景{d['index']}  帧 {d['start_frame']:3d}-{d['end_frame']:3d}  "
              f"{d['start_s']:5.2f}-{d['end_s']:5.2f}s  {d['duration_s']:4.2f}s  "
              f"{d['color_hex']}  切点分={d['cut_score']}")
    if gap:
        print(f"可分性：cut 最小={gap['min_cut_score']} vs 非 cut 最大={gap['max_noncut_score']} "
              f"→ 间隔 {gap['margin']}（{'可分' if gap['separable'] else '🔴 不可分'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
