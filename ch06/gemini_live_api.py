# ─────────────────────────────────────────────────────────
# ① 这是什么：Gemini Live API（WebSocket 双向流）最小示例，重点是 interrupted 打断语义。
# ② 对应书稿：第 6 章 §6.2.6「Live API 与打断语义」。
# ③ 怎么跑：
#       .venv/bin/python ch06/gemini_live_api.py        # 自检（打印缺失清单，rc=0）
#    真跑需：GEMINI_API_KEY + 麦克风采集 + 音频播放 + 音频设备（见文件末 4 步）。
# ④ 依赖：google-genai（本机 `.venv` 实测 2.33.0）；GEMINI_API_KEY；设备侧未实现。
# ⑤ 实测解释器：python（google-genai 2.33.0 / Python 3.10.12）
# ⑥ 与书稿的差异：见下方 4 条（key 出码 + 书稿未定义函数改为显式桩）。
#
# 书稿出处：ch06 §6.2.6 Google Gemini（Live API）
# 本文件相对书稿的改动：
#   1. `read_mic_chunk()` —— 书里没定义（注释自己写了「伪代码」）
#      本文件 → 保留同名桩函数，显式 raise NotImplementedError，并说明要接什么
#   2. `stop_playback()` 与 `play_audio(...)` —— 书里没定义
#      本文件 → 同样保留桩函数
#   3. `genai.Client(api_key="...")` 写死 key → 改为读 GEMINI_API_KEY
#   4. 其余保持书稿写法：LiveConnectConfig(response_modalities=["AUDIO"]) /
#      client.aio.live.connect(...) / session.send_realtime_input(...) / session.receive()
# 定级：C（示意代码，不可直接运行）—— 缺麦克风采集、缺播放、缺 key；这些都与书稿讲解的
#       「打断语义」无关，属于设备侧，本仓库没有音频设备可验证，所以不假装能跑。
# ─────────────────────────────────────────────────────────
"""
Gemini Live API（WebSocket 双向流 / 打断语义）—— 示意代码，不可直接运行

书里这段要教的是「打断（interruption）语义」：服务端检测到用户开口就停止生成，
并向下行推 interrupted 标记。真正跑起来还需要：
  · GEMINI_API_KEY
  · 麦克风采集（16kHz PCM 分片）
  · 音频播放（PCM 下行 → 扬声器）
  · 本机/服务器有可用音频设备

本文件把「设备侧」抽成 3 个桩函数，Live API 的调用序列保持书稿原样。
直接 `python ecosystem/gemini_demo/gemini_live_api.py` 会打印缺失清单并退出。
"""

import asyncio
import os
import sys

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"   # 书稿原值（示意）


# ── 书稿未定义的三个设备侧函数：这里显式留桩，避免"看起来能跑" ──────────
async def read_mic_chunk():
    """需要你接：麦克风采集，返回 16kHz / 单声道 / 16bit PCM 的一小段 bytes。"""
    raise NotImplementedError("read_mic_chunk 未实现：需要接麦克风采集（如 sounddevice / pyaudio）")


def stop_playback():
    """需要你接：收到 interrupted 后立即停止播放并清空缓冲。"""
    raise NotImplementedError("stop_playback 未实现：需要接音频播放器")


def play_audio(data: bytes):
    """需要你接：把下行 PCM 分片丢给扬声器。"""
    raise NotImplementedError("play_audio 未实现：需要接音频播放器")


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[跳过] 未设置 GEMINI_API_KEY。", file=sys.stderr)
        sys.exit(2)

    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(response_modalities=["AUDIO"])

    async with client.aio.live.connect(model=MODEL, config=config) as session:

        async def pump_microphone():
            while True:
                chunk = await read_mic_chunk()
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                )

        async for message in session.receive():
            content = message.server_content
            if content is None:
                continue
            # 用户打断：立刻停止播放并清空缓冲
            if content.interrupted:
                stop_playback()
                continue
            if content.model_turn:
                for part in content.model_turn.parts:
                    if part.inline_data:
                        play_audio(part.inline_data.data)


if __name__ == "__main__":
    print(__doc__)
    print("缺失清单：GEMINI_API_KEY / 麦克风采集 / 音频播放 / 音频设备")
    print("本文件定级 C：Live API 调用序列可直接抄，设备侧需你自己实现。")
    raise SystemExit(0)


# ─────────────────────────────────────────────────
# 实测记录
#   解释器：python（Python 3.10.12 / google-genai 2.33.0）
#   1) python -c "import google.genai" → 成功，__file__ 指向 site-packages（不是仓库同名目录）
#   2) 本文件 __main__ 自检 → rc=0（打印缺失清单，不建连接）
#   3) 未调用 main()：无 GEMINI_API_KEY，且麦克风/播放器与音频设备均不具备；
#      书稿本身也把设备侧标为「伪代码」，故定级 C，不假装可跑。
# ─────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
# 想自己接真麦克风？只需改这 4 个地方
#
# 本文件把设备侧留成桩（`NotImplementedError`），因为打断语义（interrupted）的讲解
# 不依赖设备，而设备侧在不同平台上写法差异很大、没法给出一个通用实现。
# 你要真的跑起来，按顺序补这 4 块：
#
#   1. read_mic_chunk()  —— 麦克风采集
#        Linux/树莓派：arecord -f S16_LE -r 16000 -c 1 -t raw | 直接读 stdout
#        macOS：sox -d -r 16000 -c 1 -b 16 -e signed-integer -t raw -
#        或 Python：sounddevice.RawInputStream(samplerate=16000, channels=1,
#                  dtype='int16', blocksize=1600, callback=...)  # 100ms/块
#        要求：16kHz、单声道、16bit、PCM，块大小 100ms 量级（太小会刷爆 websocket）
#
#   2. play_audio(bytes) —— 播放模型回传的 PCM
#       最简单：sounddevice.RawOutputStream(samplerate=24000, channels=1,
#               dtype='int16').write(numpy.frombuffer(data, dtype='int16'))
#       注意采样率跟 Gemini 返回的 mime_type 走（audio/pcm;rate=24000 是常见值）
#
#   3. stop_playback() —— 「打断」的真正实现：清空播放缓冲
#       play_audio 若在独立线程/队列里播，这里要能立刻清队列 + 关流，
#       否则模型被打断了、声音还在播 —— 读者会以为 interrupted 没生效
#
#   4. GEMINI_API_KEY —— 环境变量，不写进代码
#         export GEMINI_API_KEY=...
#
# 另外两点：
#   · session 里除了 send_realtime_input，还可以发 text 输入（client_content），
#     书稿只讲了音频路径，需要文字打断可以自己加
#   · 回声消除（AEC）没做：外放时模型会听到自己的声音。要么戴耳机，
#     要么接 webrtcvad / speexdsp 之类做 AEC，这是 Live API 落地的真实门槛
# ─────────────────────────────────────────────────────────
