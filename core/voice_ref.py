"""Chuẩn bị giọng mẫu cho VieNeu-TTS: cắt ĐÚNG CHỖ NGẮT CÂU, không quá 8 giây.

VieNeu chỉ lấy 8 giây ĐẦU của clip mẫu (cắt cứng, kể cả khi đang giữa chữ) rồi khử nhiễu và
trích hồ sơ giọng (speaker embedding + mã codec). Giọng mẫu 18 s mà đưa thẳng vào sẽ bị cắt
giữa từ -> model học một đoạn kết thúc cụt. Ở đây tìm các đoạn câu trọn vẹn 3-7,8 s, bắt đầu
và kết thúc ở chỗ ngừng sạch; engine sẽ thử từng đoạn và giữ đoạn clone giống giọng gốc nhất.
"""
import os
import shutil
import subprocess
from typing import List, Tuple

import numpy as np

from core.audio_utils import read_wav, trim_silence

# VieNeu cắt cứng ở 8,0 s sau khi tự bỏ lặng hai đầu: chừa 0,2 s để không bao giờ chạm mốc đó.
MAX_SECONDS = 7.8
MIN_SECONDS = 3.0


def load_reference(src_audio: str) -> Tuple[np.ndarray, int]:
    """Đọc giọng mẫu (wav/mp3/m4a...) thành mono float, giữ nguyên tần số lấy mẫu gốc."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and not src_audio.lower().endswith((".wav", ".flac")):
        tmp = f"{src_audio}.tmp.wav"
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src_audio,
                        "-ac", "1", "-c:a", "pcm_f32le", tmp], check=True)
        try:
            return read_wav(tmp)
        finally:
            os.remove(tmp)
    return read_wav(src_audio)


def find_pauses(samples: np.ndarray, sr: int, min_pause: float = 0.15) -> List[Tuple[float, float, float]]:
    """Các chỗ ngừng bên trong clip: (giữa, độ dài giây, độ sạch dB so với đỉnh — càng âm càng sạch)."""
    frame = int(sr * 0.02)
    n = samples.size // frame
    if n == 0:
        return []
    energy = np.array([20 * np.log10(np.sqrt(np.mean(samples[i * frame:(i + 1) * frame] ** 2)) + 1e-9)
                       for i in range(n)])
    peak = float(energy.max())
    quiet = energy < peak - 35
    pauses, start = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            length = (i - start) * 0.02
            if length >= min_pause and start > 0 and i < n:          # bỏ lặng đầu/cuối file
                floor = float(np.median(energy[start:i])) - peak
                pauses.append(((start + i) / 2 * 0.02, length, floor))
            start = None
    return pauses


def candidate_segments(samples: np.ndarray, sr: int, max_seconds: float = MAX_SECONDS,
                       min_seconds: float = MIN_SECONDS, limit: int = 4) -> List[Tuple[float, float]]:
    """Các đoạn câu trọn vẹn tốt nhất làm giọng mẫu, xếp từ tốt đến kém.

    Ranh giới mạnh = chỗ ngừng dài và sạch (hết câu). Đầu/cuối file chỉ là ranh giới khi ở
    đó im lặng thật (file hay bị cắt cụt giữa lời). Ưu tiên đoạn dài vừa phải và có chỗ ngắt
    hơi bên trong (model học được nhịp ngắt câu của người đọc).
    """
    total = samples.size / sr
    if total <= max_seconds:
        return [(0.0, total)]
    pauses = find_pauses(samples, sr)
    edge = int(sr * 0.1)
    peak_db = 20 * np.log10(float(np.max(np.abs(samples))) + 1e-9)

    def quiet(seg: np.ndarray) -> bool:
        return 20 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-9) < peak_db - 35

    bounds = ([(0.0, 0.4)] if quiet(samples[:edge]) else []) \
        + [(t, length + max(0.0, (-floor - 40) / 100)) for t, length, floor in pauses] \
        + ([(total, 0.4)] if quiet(samples[-edge:]) else [])
    scored = []
    for i, (a, sa) in enumerate(bounds):
        for b, sb in bounds[i + 1:]:
            if min_seconds <= b - a <= max_seconds:
                inner = sum(1 for t, _, _ in pauses if a < t < b)
                scored.append((2.0 * sa + 2.0 * sb + 0.1 * min(inner, 3) + 0.05 * (b - a), a, b))
    out: List[Tuple[float, float]] = []
    for _, a, b in sorted(scored, reverse=True):
        if all(abs(a - x) > 0.5 or abs(b - y) > 0.5 for x, y in out):
            out.append((a, b))
        if len(out) >= limit:
            break
    return out


def segment(samples: np.ndarray, sr: int, a: float, b: float) -> np.ndarray:
    """Cắt đoạn [a, b] giây, fade 30 ms hai mép, giữ lại chút lặng tự nhiên hai đầu."""
    cut = samples[int(a * sr): int(b * sr)].copy()
    fade = min(int(sr * 0.03), cut.size // 2)
    if fade > 0:
        cut[:fade] *= np.linspace(0.0, 1.0, fade)
        cut[-fade:] *= np.linspace(1.0, 0.0, fade)
    return trim_silence(cut, sr, threshold_db=-45, lead_ms=60, tail_ms=150)
