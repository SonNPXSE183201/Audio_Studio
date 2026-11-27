"""Xử lý audio bằng numpy (không phụ thuộc pydub/audioop)."""
import os
import wave
from typing import Tuple

import numpy as np


def to_mono_float(samples) -> np.ndarray:
    """Đưa mọi kiểu dữ liệu audio về float32 mono.

    Chỉ chia theo dải số nguyên khi dữ liệu THẬT SỰ là kiểu số nguyên. Bản cũ đoán theo
    biên độ (đỉnh > 1.5 => int16) nên một chunk float hơi to bị chia 32768 và gần như câm.
    """
    if hasattr(samples, "detach"):
        samples = samples.detach().float().cpu().numpy()
    raw = np.asarray(samples)
    if np.issubdtype(raw.dtype, np.integer):
        scale = float(np.iinfo(raw.dtype).max) + 1.0
        data = raw.astype(np.float32) / scale
    else:
        data = raw.astype(np.float32, copy=False)
    data = np.squeeze(data)
    if data.ndim == 2:
        # (channels, n) hoặc (n, channels)
        data = data.mean(axis=0 if data.shape[0] < data.shape[1] else 1)
    return np.nan_to_num(data.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_mono_float(data), int(sr)
    except ImportError:
        pass
    with wave.open(path, "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"Chỉ hỗ trợ WAV 16-bit khi thiếu soundfile: {path}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def peak_limit(samples: np.ndarray, max_peak: float = 0.97) -> np.ndarray:
    """Hạ cả đoạn xuống khi đỉnh vượt ngưỡng (không cắt cứng -> không méo)."""
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak > max_peak:
        return (samples * (max_peak / peak)).astype(np.float32)
    return samples


def write_wav(path: str, samples: np.ndarray, sr: int, float32: bool = False) -> None:
    """Ghi WAV mono. float32=True giữ nguyên độ phân giải (dùng cho cache trung gian)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = peak_limit(to_mono_float(samples), 0.999)
    tmp = f"{path}.tmp.wav"
    if float32:
        import soundfile as sf

        sf.write(tmp, data, int(sr), subtype="FLOAT", format="WAV")
    else:
        pcm = np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(tmp, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sr))
            handle.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(src_sr), int(dst_sr))
        return resample_poly(samples, int(dst_sr) // g, int(src_sr) // g).astype(np.float32)
    except ImportError:
        pass
    duration = samples.size / float(src_sr)
    target_len = max(1, int(round(duration * dst_sr)))
    src_t = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


def _frame_db(samples: np.ndarray, sr: int, frame_ms: int = 10) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    count = samples.size // frame
    if count == 0:
        return np.array([-120.0])
    frames = samples[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def trim_silence(
    samples: np.ndarray,
    sr: int,
    threshold_db: float = -42.0,
    lead_ms: int = 30,
    tail_ms: int = 60,
) -> np.ndarray:
    """Cắt khoảng lặng đầu/cuối do TTS tự sinh để nhịp nghỉ do tool kiểm soát."""
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-6:
        return samples[:0]
    db = _frame_db(samples, sr) - 20.0 * np.log10(peak)  # tương đối so với đỉnh
    voiced = np.where(db > threshold_db)[0]
    if voiced.size == 0:
        return samples
    frame = max(1, int(sr * 10 / 1000))
    start = max(0, voiced[0] * frame - int(sr * lead_ms / 1000))
    end = min(samples.size, (voiced[-1] + 1) * frame + int(sr * tail_ms / 1000))
    return samples[start:end]
