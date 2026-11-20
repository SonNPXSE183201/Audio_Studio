"""Hậu kỳ audio chương bằng FFmpeg: lọc ù, chuẩn độ to 2 lượt, chặn đỉnh.

Thay cho loudnorm 1 lượt (chế độ dynamic) ở bước ghép video: chế độ đó liên tục kéo
gain lên xuống theo câu -> tiếng "phồng xẹp", nhiễu nền bị kéo to trong khoảng nghỉ.
Ở đây đo độ to cả chương trước (lượt 1), rồi áp MỘT mức gain cố định + limiter chặn
đỉnh (lượt 2): âm sắc và nhịp nghỉ giữ nguyên, độ to đạt chuẩn -16 LUFS.
"""
import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, Optional

from core.utils import log

_LIMITER_LATENCY: Optional[bool] = None


def _ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _limiter_supports_latency(ffmpeg: str) -> bool:
    """alimiter có tùy chọn latency (bù trễ) từ FFmpeg 5.1; bản Ubuntu cũ thì không."""
    global _LIMITER_LATENCY
    if _LIMITER_LATENCY is None:
        res = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "lavfi", "-i", "anullsrc=d=0.1", "-af", "alimiter=latency=1",
             "-f", "null", "-"],
            capture_output=True,
        )
        _LIMITER_LATENCY = res.returncode == 0
    return _LIMITER_LATENCY


def limiter_filter(ffmpeg: str, true_peak_db: float = -1.5) -> str:
    # Limiter chặn theo đỉnh MẪU; đỉnh thật (giữa hai mẫu) cao hơn ~0,3-0,5 dB -> chừa thêm 0,5 dB.
    limit = 10 ** ((true_peak_db - 0.5) / 20.0)
    flt = f"alimiter=limit={limit:.4f}:attack=5:release=60:level=0"
    if _limiter_supports_latency(ffmpeg):
        flt += ":latency=1"
    return flt


def measure_loudness(path: str, pre_filter: str = "anull") -> Optional[Dict[str, float]]:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return None
    res = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", path, "-af",
         f"{pre_filter},loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr)
    if res.returncode != 0 or not match:
        return None
    data = json.loads(match.group(0))
    try:
        return {k: float(v) for k, v in data.items() if k.startswith(("input_", "target_"))}
    except ValueError:
        return None


def master_wav(src: str, dst: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Ghi bản đã hậu kỳ của `src` ra `dst` (cùng sample rate). Trả về False nếu không làm được."""
    cfg = cfg or {}
    ffmpeg = _ffmpeg()
    if not ffmpeg or not cfg.get("enabled", True):
        return False
    lufs = float(cfg.get("lufs", -16.0))
    tp = float(cfg.get("true_peak_db", -1.5))
    pre = []
    if cfg.get("highpass_hz", 70):
        pre.append(f"highpass=f={float(cfg.get('highpass_hz', 70))}:poles=2")
    if cfg.get("compressor", True):
        # Nén nhẹ 2:1 như sách nói: chỗ to vọt / nhỏ hụt trong câu được làm đều, không "bơm" tiếng.
        pre.append("acompressor=threshold=-24dB:ratio=2:attack=15:release=250:knee=6:makeup=1")
    if cfg.get("deesser", False):
        pre.append("deesser=i=0.25")
    pre_filter = ",".join(pre) or "anull"
    stats = measure_loudness(src, pre_filter)
    if not stats or stats.get("input_i", -70.0) <= -69.0:
        log("Không đo được độ to audio chương; giữ nguyên bản gốc.", "Cảnh báo")
        return False
    gain = max(-20.0, min(20.0, lufs - stats["input_i"]))
    chain = f"{pre_filter},volume={gain:.2f}dB,{limiter_filter(ffmpeg, tp)}"
    tmp = f"{dst}.tmp.wav"
    res = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-i", src, "-af", chain, "-c:a", "pcm_s24le", tmp],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if res.returncode != 0:
        log(f"Hậu kỳ audio lỗi, giữ bản gốc: {res.stderr[-400:]}", "Cảnh báo")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, dst)
    log(f"Hậu kỳ audio: {stats['input_i']:.1f} LUFS -> {lufs:.1f} LUFS (gain {gain:+.1f} dB, đỉnh <= {tp} dBTP)", "TTS")
    return True
