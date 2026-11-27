"""Chấm chất lượng giọng nói bằng DNSMOS (Microsoft, thang MOS 1-5) qua torchmetrics.

  sig   độ sạch của CHÍNH giọng nói: thấp = méo, sước, rè, tiếng kim loại
  bak   nền: thấp = còn nhiễu / nhạc nền
  ovrl  tổng thể
Dùng để lọc clip train (Demucs đôi khi để lại méo; model học cả tiếng méo) và chấm audio sinh ra.
Không có torchmetrics/onnxruntime thì trả None (bỏ qua bước chấm, không lỗi).
"""
from typing import Dict, Optional

import numpy as np

_FAILED = False


def dnsmos(samples: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
    global _FAILED
    if _FAILED:
        return None
    try:
        import torch
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
    except Exception:  # noqa: BLE001 - thiếu gói thì bỏ qua chấm điểm
        _FAILED = True
        return None
    x = torch.from_numpy(np.asarray(samples, dtype=np.float32).reshape(-1))
    score = deep_noise_suppression_mean_opinion_score(x, sr, personalized=False).tolist()
    return {"p808": round(score[0], 2), "sig": round(score[1], 2), "bak": round(score[2], 2),
            "ovrl": round(score[3], 2)}
