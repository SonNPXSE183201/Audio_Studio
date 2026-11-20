"""Xử lý audio bằng numpy (không phụ thuộc pydub/audioop)."""
import os
import wave
from typing import List, Optional, Tuple

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


def silence(ms: int, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(sr * ms / 1000.0)), dtype=np.float32)


def room_tone(ms: float, sr: int, level_db: Optional[float] = -66.0, seed: int = 0) -> np.ndarray:
    """Khoảng nghỉ có "không khí phòng" rất nhỏ thay cho im lặng số tuyệt đối.

    Im lặng tuyệt đối giữa hai câu (trong khi câu có nhiễu nền) khiến tai nghe thấy
    chỗ cắt. Nhiễu hồng ở -66 dBFS gần như không nghe được nhưng lấp chỗ hụt.
    """
    n = max(0, int(round(sr * ms / 1000.0)))
    if n == 0 or level_db is None:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(spectrum.size, dtype=np.float64)
    freqs[0] = 1.0
    pink = np.fft.irfft(spectrum / np.sqrt(freqs), n)
    pink -= pink.mean()
    rms = float(np.sqrt(np.mean(pink ** 2) + 1e-12))
    out = (pink * (10 ** (level_db / 20.0) / rms)).astype(np.float32)
    return apply_fade(out, sr, 20, 20)


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


def apply_fade(samples: np.ndarray, sr: int, fade_ms: float = 10, fade_out_ms: Optional[float] = None) -> np.ndarray:
    """Fade vào/ra dạng nửa cos (mượt hơn đường thẳng, không tạo tiếng click)."""
    if samples.size == 0:
        return samples
    out = samples.astype(np.float32, copy=True)
    fade_out_ms = fade_ms if fade_out_ms is None else fade_out_ms
    for ms, head in ((fade_ms, True), (fade_out_ms, False)):
        length = min(out.size // 2, int(sr * ms / 1000))
        if length <= 1:
            continue
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, length))).astype(np.float32)
        if head:
            out[:length] *= ramp
        else:
            out[-length:] *= ramp[::-1]
    return out


def crossfade_concat(a: np.ndarray, b: np.ndarray, sr: int, ms: float = 25.0) -> np.ndarray:
    """Nối a và b chồng lên nhau `ms` mili giây với fade equal-power."""
    n = min(a.size, b.size, int(sr * ms / 1000))
    if n <= 1:
        return np.concatenate([a, b]).astype(np.float32)
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    mixed = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], mixed, b[n:]]).astype(np.float32)


def normalize_rms(samples: np.ndarray, target_db: float = -20.0, max_peak: float = 0.95) -> np.ndarray:
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
    gain = (10 ** (target_db / 20.0)) / max(rms, 1e-6)
    out = samples * gain
    peak = float(np.max(np.abs(out)))
    if peak > max_peak:
        out *= max_peak / peak
    return out.astype(np.float32)


def voiced_rms_db(samples: np.ndarray, sr: int, gate_db: float = -40.0) -> Optional[float]:
    """RMS (dBFS) chỉ tính trên các frame có tiếng nói, bỏ khoảng lặng."""
    if samples.size == 0:
        return None
    db = _frame_db(samples, sr, 20)
    peak_db = float(db.max())
    voiced = db[db > peak_db + gate_db]
    if voiced.size == 0:
        return None
    power = np.mean(10 ** (voiced / 10.0))
    return float(10.0 * np.log10(power + 1e-12))


def normalize_voiced(
    samples: np.ndarray,
    sr: int,
    target_db: float = -20.0,
    max_gain_db: float = 9.0,
    max_peak: float = 0.95,
) -> np.ndarray:
    """Đưa độ to phần tiếng nói của mỗi chunk về cùng một mức để các câu không to nhỏ lệch nhau."""
    level = voiced_rms_db(samples, sr)
    if level is None:
        return samples
    gain_db = float(np.clip(target_db - level, -max_gain_db, max_gain_db))
    out = samples * (10 ** (gain_db / 20.0))
    return peak_limit(out.astype(np.float32), max_peak)


def longest_internal_gap_ms(samples: np.ndarray, sr: int, threshold_db: float = -45.0) -> int:
    """Khoảng câm dài nhất NẰM GIỮA phần có tiếng (dấu hiệu kẹt/nuốt chữ giữa câu)."""
    if samples.size == 0:
        return 0
    db = _frame_db(samples, sr, 10)
    rel = db - float(db.max())
    voiced = np.where(rel > threshold_db)[0]
    if voiced.size < 2:
        return 0
    gaps = np.diff(voiced) - 1
    return int(gaps.max() * 10) if gaps.size else 0


def tail_level_db(samples: np.ndarray, sr: int, ms: int = 30) -> float:
    """Độ to của `ms` cuối so với đỉnh. Cao => câu bị cắt cụt giữa chừng."""
    n = int(sr * ms / 1000)
    if n == 0 or samples.size < n * 2:
        return -120.0
    peak = float(np.max(np.abs(samples))) + 1e-12
    tail = samples[-n:]
    return float(20.0 * np.log10(np.sqrt(np.mean(tail ** 2) + 1e-12) / peak))


def f0_contour(samples: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 450.0,
               frame_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    """Cao độ (Hz) các frame có tiếng nói, tự tương quan tính bằng FFT cho cả loạt frame."""
    n, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if samples.size < n * 2:
        return np.zeros(0, dtype=np.float32)
    count = 1 + (samples.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[idx].astype(np.float32)
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    frames = frames[energy > 0.05 * (energy.max() + 1e-9)]
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    frames = (frames - frames.mean(axis=1, keepdims=True)) * np.hanning(n)[None, :]
    spec = np.fft.rfft(frames, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / fmax), min(n - 1, int(sr / fmin))
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(ac.shape[0]), lags]
    return (sr / lags[strength > 0.5]).astype(np.float32)


def breath_regions(samples: np.ndarray, sr: int, min_ms: float = 150.0) -> List[Tuple[int, int]]:
    """Tìm tiếng lấy hơi (hít vào): đoạn KHÔNG có cao độ, nhỏ hơn giọng nói nhưng to hơn im lặng,
    dài >= min_ms, đứng ngay sau khoảng lặng (hoặc ở đầu câu), năng lượng không dồn ở tần số cao
    (phụ âm s/x/ph có trọng tâm phổ > 3.5 kHz và ngắn nên không bị coi là hơi thở).
    Trả về danh sách (mẫu bắt đầu, mẫu kết thúc).
    """
    n, hop = int(sr * 0.03), int(sr * 0.01)
    if samples.size < n * 3:
        return []
    count = 1 + (samples.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[idx].astype(np.float32)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    win = (frames - frames.mean(axis=1, keepdims=True)) * np.hanning(n)[None, :]
    spec = np.abs(np.fft.rfft(win, 2 * n, axis=1))
    ac = np.fft.irfft(spec ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / 450), min(n - 1, int(sr / 70))
    voiced = ac[:, lo:hi].max(axis=1) > 0.5
    freqs = np.fft.rfftfreq(2 * n, 1 / sr)
    centroid = (spec * freqs[None, :]).sum(axis=1) / (spec.sum(axis=1) + 1e-9)
    if not voiced.any():
        return []
    speech = float(np.median(rms_db[voiced]))
    quiet = rms_db < speech - 38
    cand = (~voiced) & (rms_db >= speech - 38) & (rms_db <= speech - 8) & (centroid < 3500)
    out: List[Tuple[int, int]] = []
    i = 0
    while i < count:
        if not cand[i]:
            i += 1
            continue
        j = i
        while j < count and cand[j]:
            j += 1
        dur_ms = (j - i) * 10
        after_silence = i == 0 or quiet[max(0, i - 5):i].any()   # trước đó <= 50 ms có lặng
        if dur_ms >= min_ms and after_silence:
            out.append((i * hop, min(samples.size, j * hop + n)))
        i = j
    return out


def suppress_breaths(samples: np.ndarray, sr: int, reduce_db: float = -30.0, min_ms: float = 150.0) -> np.ndarray:
    """Giảm tiếng lấy hơi (mặc định -30 dB, gần như mất hẳn) với fade 10 ms để không click; giữ nguyên nhịp."""
    regions = breath_regions(samples, sr, min_ms)
    if not regions:
        return samples
    out = samples.astype(np.float32).copy()
    gain_lo = 10 ** (reduce_db / 20.0)
    ramp = int(sr * 0.01)
    for a, b in regions:
        env = np.full(b - a, gain_lo, dtype=np.float32)
        r = min(ramp, (b - a) // 2)
        if r > 0:
            env[:r] = np.linspace(1.0, gain_lo, r)
            env[-r:] = np.linspace(gain_lo, 1.0, r)
        out[a:b] *= env
    return out


def median_f0(samples: np.ndarray, sr: int) -> Optional[float]:
    f0 = f0_contour(samples, sr)
    return float(np.median(f0)) if f0.size >= 10 else None


def semitones(a: float, b: float) -> float:
    return float(12.0 * np.log2(a / b))


def clipping_ratio(samples: np.ndarray, threshold: float = 0.999) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= threshold))


def duration_ms(samples: np.ndarray, sr: int) -> int:
    return int(round(samples.size * 1000.0 / sr))
