"""Nghe lại audio TTS bằng ASR (PhoWhisper) để phát hiện vấp, nuốt chữ, đọc sượt."""
import re
import unicodedata
from typing import Any, Dict, Iterable, Optional

import numpy as np

from core.audio_utils import resample
from core.utils import log


def normalize_compare(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", "", text)


def _drop_terms(text: str, terms: Iterable[str]) -> str:
    """Bỏ tên riêng khỏi chuỗi so sánh: ASR hay viết sai tên Hán Việt dù giọng đọc đúng."""
    text = unicodedata.normalize("NFC", text or "").lower()
    for term in sorted({t.strip().lower() for t in terms if t and t.strip()}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(unicodedata.normalize('NFC', term))}(?!\w)", " ", text)
    return text


def repeated_words(hypothesis: str, reference: str, n: int = 3) -> bool:
    """ASR nghe ra cụm n từ lặp liền nhau mà văn bản gốc không có -> giọng bị vấp/lặp."""
    def grams(text: str):
        words = re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", text or "").lower()).split()
        return words

    hyp, ref = grams(hypothesis), " ".join(grams(reference))
    for size in range(1, n + 1):
        for i in range(len(hyp) - 2 * size + 1):
            a, b = hyp[i:i + size], hyp[i + size:i + 2 * size]
            if a == b and len("".join(a)) >= 2 and " ".join(a + b) not in ref:
                return True
    return False


def char_error_rate(reference: str, hypothesis: str, ignore: Iterable[str] = ()) -> float:
    if ignore:
        reference, hypothesis = _drop_terms(reference, ignore), _drop_terms(hypothesis, ignore)
    ref = normalize_compare(reference)
    hyp = normalize_compare(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (rc != hc),
            )
        previous = current
    return previous[-1] / float(len(ref))


class ASRChecker:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.model_id = cfg.get("model_id", "vinai/PhoWhisper-large")
        self.pipe = None
        self._failed = False

    def load(self) -> bool:
        if not self.enabled or self._failed:
            return False
        if self.pipe is not None:
            return True
        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            dtype = torch.float16 if device == 0 else torch.float32
            log(f"Nạp ASR '{self.model_id}' để kiểm tra giọng đọc...", "ASR")
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device,
                torch_dtype=dtype,
            )
            return True
        except Exception as exc:
            self._failed = True
            log(f"Không nạp được ASR ({exc}). Bỏ qua bước kiểm tra giọng.", "Cảnh báo")
            return False

    def transcribe(self, samples: np.ndarray, sr: int) -> Optional[str]:
        if not self.load():
            return None
        audio = resample(samples, sr, 16000)
        try:
            result = self.pipe({"raw": audio, "sampling_rate": 16000})
            return str(result.get("text", "")).strip()
        except Exception as exc:
            log(f"ASR lỗi: {exc}", "Cảnh báo")
            return None

    def transcribe_batch(self, items, batch_size: int = 16):
        """Nghe lại nhiều đoạn một lượt (nhanh hơn nhiều so với từng đoạn)."""
        if not items or not self.load():
            return [None] * len(items)
        inputs = [{"raw": resample(s, sr, 16000), "sampling_rate": 16000} for s, sr in items]
        try:
            results = self.pipe(inputs, batch_size=int(self.cfg.get("batch_size", batch_size)))
            return [str(r.get("text", "")).strip() for r in results]
        except Exception as exc:
            log(f"ASR batch lỗi ({exc}); nghe từng đoạn.", "Cảnh báo")
            return [self.transcribe(s, sr) for s, sr in items]

    def unload(self) -> None:
        if self.pipe is None:
            return
        import torch

        del self.pipe
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
