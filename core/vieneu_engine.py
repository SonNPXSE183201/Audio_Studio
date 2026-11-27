"""VieNeu-TTS v3 Turbo: clone giọng từ giọng mẫu, đọc cả chương theo đúng logic của VieNeu.

Logic đọc giống hệt ``Vieneu.infer()`` (gói ``vieneu`` 3.8.5):
  1. ``normalize_to_chunks_v3_with_gaps``: chuẩn hóa + tách câu bằng sea-g2p, đóng gói câu
     thành chunk <= max_chars, nhớ loại ranh giới giữa các chunk (đoạn / câu / trong câu).
  2. Sinh mọi chunk (GPU: gộp batch theo độ dài phoneme; CPU: tuần tự).
  3. ``join_audio_chunks`` với khoảng nghỉ theo loại ranh giới (0,70 / 0,50 / 0,30 s).

Khác ``infer()`` duy nhất ở chỗ engine trả về TỪNG chunk trước khi ghép, để bước nghe lại
(nếu bật) đọc lại đúng chunk lỗi. Không được đưa chunk đã chuẩn hóa qua ``infer_batch()``:
chuẩn hóa lần 2 đổi dấu phẩy cuối chunk thành dấu chấm -> giọng hạ xuống như hết câu.
"""
import os
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from core import voice_ref
from core.audio_utils import write_wav
from core.utils import log

VOICE_NAME = "audio_studio_voice"
SAMPLE_RATE = 48_000


def chunk_text(text: str, max_chars: int = 256) -> Tuple[List[str], List[str]]:
    """Văn bản chương -> (chunk đã chuẩn hóa, loại ranh giới giữa các chunk) theo VieNeu."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

    return normalize_to_chunks_v3_with_gaps(text, max_chars=int(max_chars))


def join_chunks(wavs: List[np.ndarray], gaps: List[str]) -> np.ndarray:
    from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks

    if not wavs:
        return np.zeros(1, dtype=np.float32)
    return join_audio_chunks(wavs, SAMPLE_RATE, silence_ps=gaps_to_silence(gaps)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class VieneuEngine:
    sample_rate = SAMPLE_RATE

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tts = None
        self.voice_ready = False
        self.voice_name = VOICE_NAME

    # ------------------------------------------------------------ model
    def load(self) -> None:
        if self.tts is not None:
            return
        from vieneu import Vieneu

        kwargs: Dict[str, Any] = {
            "mode": "v3turbo",
            "backend": str(self.cfg.get("backend", "auto")),
            "max_batch_size": int(self.cfg.get("batch_size", 32)),
        }
        if self.cfg.get("precision"):
            kwargs["precision"] = str(self.cfg["precision"])
        if self.cfg.get("backbone_repo"):
            # Model đã fine-tune (finetune/train_voice): chỉ chạy GPU/PyTorch, ONNX dùng đồ thị của model gốc.
            kwargs["backbone_repo"] = str(self.cfg["backbone_repo"])
            if kwargs["backend"] == "auto":
                kwargs["backend"] = "pytorch"
        started = time.time()
        log(f"Nạp VieNeu-TTS v3 Turbo (backend {kwargs['backend']})...", "TTS")
        self.tts = Vieneu(**kwargs)
        log(f"VieNeu sẵn sàng: backend {self.tts.backend}, {time.time() - started:.0f}s", "TTS")
        self.voice_ready = False

    def unload(self) -> None:
        if self.tts is None:
            return
        try:
            self.tts.close()
        except Exception:  # noqa: BLE001 - giải phóng hết mức có thể
            pass
        self.tts = None
        self.voice_ready = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log("Đã giải phóng VieNeu.", "TTS")

    @property
    def backend(self) -> str:
        return getattr(self.tts, "backend", "?") if self.tts is not None else "?"

    # ------------------------------------------------------------ voice
    def set_voice(self, ref_wav: str) -> None:
        """Đăng ký giọng mẫu (đã cắt sẵn <= 7,8 s): khử nhiễu + hồ sơ giọng, làm một lần."""
        self.load()
        self.tts.add_voice(VOICE_NAME, ref_wav, denoise=bool(self.cfg.get("denoise", True)))
        self.voice_name, self.voice_ready = VOICE_NAME, True

    def use_preset(self, name: str) -> None:
        """Dùng giọng có sẵn trong model (giọng đã fine-tune đóng gói bằng make_voice.py)."""
        self.load()
        resolved = self.tts.resolve_voice_name(name)
        if not resolved:
            names = [label for label, _ in self.tts.list_preset_voices()]
            raise RuntimeError(f"Model không có giọng '{name}'. Có: {', '.join(names)}")
        self.voice_name, self.voice_ready = resolved, True

    def speaker_embedding(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Hồ sơ giọng (x-vector) của một đoạn audio, dùng để so độ giống giọng."""
        self.load()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav(path, samples, sr, float32=True)
            emb, _ = self.tts.engine.prepare_reference(path, denoise=False, use_ref_codes=False)
        finally:
            os.remove(path)
        return np.asarray(emb, dtype=np.float32)

    def reference_profile(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Hồ sơ giọng của CẢ giọng mẫu gốc: trung bình các cửa sổ 6 s (VieNeu chỉ nhận <= 8 s)."""
        win, hop = int(6 * sr), int(3 * sr)
        starts = list(range(0, max(1, samples.size - win + 1), hop)) or [0]
        embs = [self.speaker_embedding(samples[s:s + win], sr) for s in starts]
        mean = np.mean([e / (np.linalg.norm(e) + 1e-12) for e in embs], axis=0)
        return mean

    def audition(self, options: List[Dict[str, Any]], full: np.ndarray, sr: int,
                 texts: List[str]) -> Dict[str, Any]:
        """Thử giọng: mỗi đoạn mẫu đọc vài câu thật, chọn đoạn có giọng sinh ra GIỐNG giọng gốc nhất.

        Độ giống = cosine giữa hồ sơ giọng của audio sinh ra và hồ sơ giọng của CẢ file mẫu
        (cùng bộ trích đặc trưng mà VieNeu dùng để clone).
        """
        target = self.reference_profile(full, sr)
        best, best_score = options[0], -2.0
        for opt in options:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                write_wav(path, opt["cut"], sr, float32=True)
                self.set_voice(path)
            finally:
                os.remove(path)
            wavs, _ = self.synthesize_chunks(texts)
            sims = [cosine(self.speaker_embedding(w, SAMPLE_RATE), target) for w in wavs if w.size]
            opt["similarity"] = float(np.mean(sims)) if sims else -1.0
            log(f"  đoạn {opt['a']:.1f}-{opt['b']:.1f}s ({opt['cut'].size / sr:.1f}s): "
                f"giống giọng gốc {opt['similarity']:.3f}", "Giọng")
            if opt["similarity"] > best_score:
                best, best_score = opt, opt["similarity"]
        self.voice_ready = False   # người gọi đăng ký lại đoạn thắng
        return best

    def pick_reference(self, src_audio: str, out_wav: str, test_texts: List[str]) -> Dict[str, Any]:
        """Giọng mẫu gốc -> đoạn câu trọn vẹn <= 7,8 s clone tốt nhất, ghi ra ``out_wav``."""
        samples, sr = voice_ref.load_reference(src_audio)
        total = samples.size / sr
        cands = voice_ref.candidate_segments(samples, sr, limit=int(self.cfg.get("ref_candidates", 4)))
        if not cands:
            log(f"Giọng mẫu {total:.1f}s không có chỗ ngừng nào để cắt câu: dùng {voice_ref.MAX_SECONDS}s đầu.",
                "Cảnh báo")
            cands = [(0.0, min(total, voice_ref.MAX_SECONDS))]
        options = []
        for a, b in cands:
            cut = samples if (a, b) == (0.0, total) else voice_ref.segment(samples, sr, a, b)
            options.append({"a": a, "b": b, "cut": cut})
        best = options[0]
        if len(options) > 1 and test_texts:
            log(f"Thử giọng: {len(options)} đoạn mẫu x {len(test_texts)} câu...", "Giọng")
            best = self.audition(options, samples, sr, test_texts)
        write_wav(out_wav, best["cut"], sr, float32=True)
        log(f"Giọng mẫu {total:.1f}s -> đoạn {best['a']:.1f}-{best['b']:.1f}s "
            f"({best['cut'].size / sr:.1f}s)", "Giọng")
        return {"segment": [round(best["a"], 2), round(best["b"], 2)],
                "seconds": round(best["cut"].size / sr, 2),
                "similarity": best.get("similarity"),
                "candidates": [{"segment": [round(o["a"], 2), round(o["b"], 2)], "similarity": o.get("similarity")}
                               for o in options]}

    # ------------------------------------------------------------ synthesis
    def sampling(self) -> Dict[str, Any]:
        c = self.cfg
        return {
            "temperature": float(c.get("temperature", 0.8)),
            "top_k": int(c.get("top_k", 25)),
            "top_p": float(c.get("top_p", 0.95)),
            "max_new_frames": int(c.get("max_new_frames", 300)),
            "repetition_penalty": float(c.get("repetition_penalty", 1.2)),
        }

    def synthesize_chunks(self, chunks: List[str]) -> Tuple[List[np.ndarray], float]:
        """Sinh từng chunk ĐÃ chuẩn hóa (từ ``chunk_text``). Trả (audio 48 kHz từng chunk, giây chạy).

        Dùng đúng hàm nội bộ mà ``Vieneu.infer()`` gọi (vieneu==3.8.5 được ghim trong requirements):
        chunk được phiên âm, gộp batch theo độ dài trên GPU, có chống "nói thêm" ở chunk ngắn.
        """
        if not self.voice_ready:
            raise RuntimeError("Chưa đăng ký giọng mẫu (set_voice).")
        if not chunks:
            return [], 0.0
        speaker_emb, ref_codes = self.tts._resolve_ref(self.voice_name, None, True, True)
        started = time.time()
        wavs = self.tts._infer_chunks(list(chunks), speaker_emb, ref_codes, ref_codes is not None,
                                      self.tts.max_batch_size, self.sampling())
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs], time.time() - started
