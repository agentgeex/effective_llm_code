#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：用 ffmpeg 合成带已知切点的测试片 clip.mp4 + 标准答案 clip_spec.json
# ② 对应书稿：ch11 §11.1（书稿未给素材；本文件造可核对的标准答案）
# ③ 怎么跑：cd <仓库根> && python ch11/cases/video_probe/make_clip.py
# ④ 依赖：ffmpeg（系统已装）；仅标准库 subprocess/json
# ⑤ 实测解释器：.venv/bin/python（Python 3.10.12），实测 rc=0（960x540@25fps，300 帧，5 场景）
# ⑥ 与书稿的差异：书稿无此脚本；本文件为预处理链提供 ground truth
# ══════════════════════════════════════════════════════════════════
"""make_clip.py — 生成「已知答案」的合成测试片

为什么要有这个文件
==================
视频识别的预处理链（抽帧 / 场景切分 / 关键帧 / 音频）要验证，就必须有**标准答案**。
真实视频拿不到 ground truth，你就没法判断"切在 4.40s"是切对了还是切歪了。

所以这里用 ffmpeg 合成一段测试片，它的每一帧内容都是**解析式可算的**：

  · 5 个场景，纯色背景硬切（切点帧号精确已知）
  · 每个场景烧一行文字（drawtext，内容已知）
  · 每个场景有一个匀速右移的白色方块（位置 = 10 + 80*t 像素，t 为该场景内秒数）
  · 5 段不同频率的正弦音（最后一段静音），与视频切点**严格对齐**

于是后面 probe.py 报出来的每个数字都能被反算：
  切点时间 → 与 spec 比误差
  关键帧 → 方块质心与 10+80*t+40 比误差（±3.2px 就能发现错了一帧）
  音频段 → FFT 主频与 spec 里的频率比

这不是"假数据"，是**带解析解的测试信号**。生成脚本本身也在仓库里，谁都能重跑。

用法
====
  python make_clip.py                    # → data/clip.mp4 + data/clip_spec.json
  python make_clip.py --out /tmp/x.mp4   # 换输出路径
  python make_clip.py --verify           # 生成后再用 ffprobe 回读，核对参数与 spec 一致

依赖：只用系统 ffmpeg（4.4.2）+ ffprobe。无 Python 第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

W, H = 960, 540
FPS = 25
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ── 场景脚本（唯一真源）───────────────────────────────────────────────
# frames：该场景占多少帧；起点由前面累加得出，切点因此是精确的整数帧。
# freq  ：该场景音频主频，0 表示静音。
SCENES = [
    dict(idx=1, frames=50,  color="0x1E3A8A", freq=440, text="SCENE 1 CH-01 FRAMES 000-049"),
    dict(idx=2, frames=60,  color="0xB91C1C", freq=523, text="SCENE 2 CH-02 FRAMES 050-109"),
    dict(idx=3, frames=75,  color="0x15803D", freq=659, text="SCENE 3 CH-03 FRAMES 110-184"),
    dict(idx=4, frames=40,  color="0xA16207", freq=784, text="SCENE 4 CH-04 FRAMES 185-224"),
    dict(idx=5, frames=75,  color="0x1F2937", freq=0,   text="SCENE 5 CH-05 FRAMES 225-299"),
]

BOX_W = BOX_H = 80
BOX_Y = 200
BOX_X0 = 10.0      # 起点
BOX_V = 80.0       # 像素/秒


def build_spec() -> dict:
    """把 SCENES 展开成带绝对帧号的 spec —— 这就是全部 ground truth。"""
    scenes, cur = [], 0
    for s in SCENES:
        n = s["frames"]
        scenes.append(
            dict(
                index=s["idx"],
                start_frame=cur,
                end_frame=cur + n - 1,
                start_s=round(cur / FPS, 4),
                end_s=round((cur + n) / FPS, 4),
                duration_s=round(n / FPS, 4),
                n_frames=n,
                color_hex="#%s" % s["color"][2:],
                text=s["text"],
                audio_freq_hz=s["freq"],
            )
        )
        cur += n
    return dict(
        generator="ch11/cases/video_probe/make_clip.py",
        note="合成测试片：解析式已知答案，用于验证抽帧/切分/关键帧/音频链路",
        width=W, height=H, fps=FPS,
        total_frames=cur,
        duration_s=round(cur / FPS, 4),
        moving_box=dict(
            w=BOX_W, h=BOX_H, y=BOX_Y, x0=BOX_X0, px_per_s=BOX_V,
            formula="x_left = floor(x0 + px_per_s * t_scene) ; centroid_x = x_left + w/2 ; centroid_y = y + h/2 ; t_scene = (frame - scene_start_frame)/fps",
            color="white (#FFFFFF)",
        ),
        drawtext=dict(
            font=FONT, fontsize=34, x=40, y=40,
            box="black@0.55, borderw=10",
        ),
        cuts=[s["start_frame"] for s in scenes[1:]],   # 硬切点（帧号）
        scenes=scenes,
    )


def _textfile(tmpdir: Path, idx: int, text: str) -> Path:
    """drawtext 的 text 里带空格/等号要转义，用 textfile= 最省事。"""
    p = tmpdir / f"s{idx}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def build_cmd(out: Path, tmpdir: Path) -> list:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    n = len(SCENES)

    # 输入段（共 3n 路）：
    #   0..n-1   每场景一路纯色视频
    #   n..2n-1  每场景一路 80x80 白方块源，由 overlay 的 x 表达式驱动它右移
    #   2n..3n-1 每场景一路音频（正弦 / 静音）
    # ★ 这里踩过一个坑：ffmpeg 4.4 的 drawbox 虽文档写 x 支持 t，
    #   实测带 t 的表达式会让整个 box 静默消失（不报错、不绘制）。
    #   改用 overlay + x='...t...' 才真正逐帧移动，所以多出这 n 路白方块输入。
    for s in SCENES:
        d = s["frames"] / FPS
        cmd += ["-f", "lavfi", "-i", f"color=c={s['color']}:s={W}x{H}:r={FPS}:d={d:.4f}"]
    for s in SCENES:
        d = s["frames"] / FPS
        cmd += ["-f", "lavfi", "-i", f"color=c=white:s={BOX_W}x{BOX_H}:r={FPS}:d={d:.4f}"]
    for s in SCENES:
        d = s["frames"] / FPS
        if s["freq"] > 0:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency={s['freq']}:sample_rate=44100:duration={d:.4f}"]
        else:
            cmd += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={d:.4f}"]

    # 滤波图：逐场景画方块 + 烧字 + 烧帧号，然后视频音频各自 concat
    parts = []
    for i, s in enumerate(SCENES):
        tf = _textfile(tmpdir, s["idx"], s["text"])
        parts.append(
            "[{i}:v][{box}:v]overlay=x='{x0}+{v}*t':y={by}:shortest=1,"
            "drawtext=fontfile={font}:textfile={tf}:x=40:y=40:fontsize=34:"
            "fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10,"
            "drawtext=fontfile={font}:text='%{{n}}':x=890:y=492:fontsize=22:fontcolor=white,"
            "format=yuv420p,setpts=PTS-STARTPTS[v{i}]".format(
                i=i, box=n + i, x0=int(BOX_X0), v=int(BOX_V), by=BOX_Y,
                font=FONT, tf=tf,
            )
        )
        parts.append(
            "[{a}:a]aresample=44100,aformat=channel_layouts=mono,"
            "asetpts=PTS-STARTPTS[a{i}]".format(a=2 * n + i, i=i)
        )

    concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
    parts.append(f"{concat_in}concat=n={n}:v=1:a=1[vout][aout]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-g", str(FPS),          # 每 1s 一个关键帧，切片/定位都方便
        "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "44100",
        "-movflags", "+faststart",
        str(out),
    ]
    return cmd


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def verify(path: Path, spec: dict) -> bool:
    """回读真实容器参数，和 spec 声明值逐项比对。不一致就报错，不静默。"""
    info = ffprobe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)

    checks = [
        ("width",  int(v["width"]),  spec["width"]),
        ("height", int(v["height"]), spec["height"]),
        ("r_frame_rate", v["r_frame_rate"], f"{FPS}/1"),
        ("nb_frames", int(v.get("nb_frames", -1)), spec["total_frames"]),
        ("duration", round(float(info["format"]["duration"]), 2), round(spec["duration_s"], 2)),
    ]
    if a:
        checks += [
            ("audio_sample_rate", int(a["sample_rate"]), 44100),
            ("audio_channels", int(a["channels"]), 2),
            ("audio_codec", a["codec_name"], "aac"),
        ]

    ok = True
    print("  参数回读核对（ffprobe 实测 vs spec 声明）：")
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"    {'✅' if good else '🔴'} {name:<20} 实测={got!r:<24} 声明={want!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成测试片（已知答案）")
    ap.add_argument("--out", default=str(DATA / "clip.mp4"))
    ap.add_argument("--spec", default=str(DATA / "clip_spec.json"))
    ap.add_argument("--verify", action="store_true", help="生成后用 ffprobe 回读核对参数")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("🔴 找不到 ffmpeg，无法生成素材", file=sys.stderr)
        return 2
    if not Path(FONT).exists():
        print(f"🔴 缺字体 {FONT}，drawtext 会失败", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = build_spec()

    print(f"生成 {out.name}：{spec['width']}x{spec['height']} @ {FPS}fps "
          f"{spec['total_frames']} 帧 = {spec['duration_s']}s，{len(SCENES)} 个场景")
    with tempfile.TemporaryDirectory() as td:
        cmd = build_cmd(out, Path(td))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("🔴 ffmpeg 失败：", r.stderr[-3000:], file=sys.stderr)
            return 1

    Path(args.spec).parent.mkdir(parents=True, exist_ok=True)
    Path(args.spec).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    size = out.stat().st_size
    print(f"  ✅ 写出 {out}  {size/1024:.0f} KB")
    print(f"  ✅ 写出 {args.spec}")
    print(f"  切点（帧号）：{spec['cuts']}  → 时间 {[round(f/FPS,2) for f in spec['cuts']]} s")
    print(f"  ⚠️  本文件 >1MB 阈值时不会入库（见 .gitignore），只入库生成脚本 + spec")

    if args.verify:
        print()
        ok = verify(out, spec)
        print("  核对结果：" + ("✅ 全部一致" if ok else "🔴 有不一致项"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
