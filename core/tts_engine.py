"""OmniVoice TTS: clone giọng, đọc theo batch, giọng mẫu mã hóa một lần.

Tăng tốc so với cách gọi từng câu:
  * voice_clone_prompt tạo MỘT lần cho cả phiên (bản cũ mã hóa lại giọng mẫu ở mỗi câu).
  * Nhiều câu được sinh trong một lần gọi generate(text=[...]) -> GPU chạy đầy tải.
"""
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.audio_utils import normalize_rms, read_wav, to_mono_float, trim_silence, write_wav
from core.utils import log


def _install_torch_accelerator_polyfill(torch) -> None:
    """transformers mới gọi torch.accelerator; PyTorch < 2.6 không có."""
    if hasattr(torch, "accelerator"):
        return

    class _PolyfillAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        @staticmethod
        def is_available():
            return torch.cuda.is_available()

        @staticmethod
        def device_count():
            return torch.cuda.device_count()

    torch.accelerator = _PolyfillAccelerator()


class TTSEngine:
    OPTIONAL_KWARGS = ("num_step", "guidance_scale", "speed", "t_shift", "denoise", "language",
                       "position_temperature", "class_temperature")

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_id = cfg.get("model_id", "k2-fsa/OmniVoice")
        self.sample_rate = int(cfg.get("sample_rate", 24000))
        self.model = None
        self._unsupported: set = set()
        self._voice_key: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._voice_prompt = None
        self._batch_ok = True

    # ------------------------------------------------------------ model
    def load(self) -> None:
        if self.model is not None:
            return
        import torch

        _install_torch_accelerator_polyfill(torch)
        from omnivoice import OmniVoice

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        if device == "cuda":
            wanted = str(self.cfg.get("dtype", "auto")).lower()
            name = torch.cuda.get_device_name(0)
            if wanted == "auto":
                # fp16 như AI_Video_Studio (giọng sạch). bf16 kém chính xác khi giải mã âm thanh -> rè, sước.
                # Dòng GTX 16xx tính fp16 hay lỗi -> fp32.
                wanted = "fp32" if "GTX 16" in name else "fp16"
            if wanted == "bf16" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif wanted in ("fp32", "float32"):
                dtype = torch.float32
            else:
                dtype = torch.float16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        log(f"Nạp OmniVoice '{self.model_id}' trên {device.upper()} ({str(dtype).replace('torch.', '')})...", "TTS")
        self.model = OmniVoice.from_pretrained(self.model_id, device_map=device, dtype=dtype)
        model_sr = getattr(self.model, "sampling_rate", None) or getattr(self.model, "sample_rate", None)
        if model_sr:
            self.sample_rate = int(model_sr)
        self._voice_key, self._voice_prompt = None, None

    def unload(self) -> None:
        if self.model is None:
            return
        import torch

        del self.model
        self.model = None
        self._voice_key, self._voice_prompt = None, None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("Đã giải phóng OmniVoice khỏi VRAM.", "TTS")

    # ------------------------------------------------------------ generation
    def _gen_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for key in self.OPTIONAL_KWARGS:
            value = self.cfg.get(key)
            if value is not None and key not in self._unsupported:
                kwargs[key] = value
        return kwargs

    def _voice_kwargs(self, ref_audio: Optional[str], ref_text: Optional[str]) -> Dict[str, Any]:
        """Mã hóa giọng mẫu một lần rồi dùng lại cho mọi câu."""
        if not ref_audio or not os.path.exists(ref_audio):
            return {}
        key = (ref_audio, ref_text)
        if self._voice_key != key:
            self._voice_key, self._voice_prompt = key, None
            builder = getattr(self.model, "create_voice_clone_prompt", None)
            if builder is not None:
                try:
                    # preprocess_prompt=False: KHÔNG để OmniVoice xóa các khoảng ngừng > 200 ms trong
                    # giọng mẫu (làm model học nhịp đọc dồn dập, không ngắt hơi -> đọc vội, giật).
                    # Tool đã tự cắt lặng đầu/cuối và chuẩn hóa giọng mẫu ở prepare_reference.
                    self._voice_prompt = builder(
                        ref_audio=ref_audio,
                        ref_text=self._ref_text_punctuated(ref_text),
                        preprocess_prompt=bool(self.cfg.get("preprocess_prompt", False)),
                    )
                    log("Đã mã hóa giọng mẫu (dùng lại cho mọi câu).", "TTS")
                except Exception as exc:
                    log(f"Không tạo được voice prompt ({exc}); truyền giọng mẫu từng lần.", "TTS")
        if self._voice_prompt is not None:
            return {"voice_clone_prompt": self._voice_prompt}
        kwargs: Dict[str, Any] = {"ref_audio": ref_audio}
        if ref_text:
            kwargs["ref_text"] = self._ref_text_punctuated(ref_text)
        return kwargs

    @staticmethod
    def _ref_text_punctuated(ref_text: Optional[str]) -> Optional[str]:
        if not ref_text:
            return None
        text = " ".join(ref_text.split()).rstrip(",;:- ")
        return text if text[-1:] in ".!?…" else text + "."

    def _generate(self, texts: List[str], voice: Dict[str, Any], seed: int) -> List[np.ndarray]:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        batch = len(texts) > 1
        kwargs: Dict[str, Any] = {"text": texts if batch else texts[0], **voice, **self._gen_kwargs()}
        if batch and "voice_clone_prompt" in kwargs:
            kwargs["voice_clone_prompt"] = [kwargs["voice_clone_prompt"]] * len(texts)
        while True:
            try:
                with torch.inference_mode():
                    audio = self.model.generate(**kwargs)
                break
            except TypeError as exc:
                bad = next((k for k in self.OPTIONAL_KWARGS if k in kwargs and k in str(exc)), None)
                if not bad:
                    raise
                log(f"OmniVoice không hỗ trợ tham số '{bad}', bỏ qua.", "TTS")
                self._unsupported.add(bad)
                kwargs.pop(bad, None)
        outputs = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        if len(outputs) != len(texts):
            raise RuntimeError(f"OmniVoice trả {len(outputs)} audio cho {len(texts)} câu")
        return [to_mono_float(o) for o in outputs]

    def synthesize_batch(
        self,
        texts: List[str],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        seed: int,
    ) -> List[Optional[np.ndarray]]:
        """Sinh nhiều câu một lượt. Phần tử None = câu đó lỗi (người gọi sẽ đọc lại)."""
        self.load()
        voice = self._voice_kwargs(ref_audio, ref_text)
        if len(texts) > 1 and self._batch_ok:
            try:
                return list(self._generate(texts, voice, seed))
            except Exception as exc:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if "out of memory" in str(exc).lower():
                    log(f"Batch {len(texts)} câu hết VRAM, chia nhỏ.", "TTS")
                    half = len(texts) // 2
                    return (self.synthesize_batch(texts[:half], ref_audio, ref_text, seed)
                            + self.synthesize_batch(texts[half:], ref_audio, ref_text, seed))
                log(f"Đọc theo batch lỗi ({exc}); chuyển sang đọc từng câu.", "TTS")
                self._batch_ok = False
        results: List[Optional[np.ndarray]] = []
        for text in texts:
            try:
                results.append(self._generate([text], voice, seed)[0])
            except Exception as exc:
                log(f"Lỗi sinh câu '{text[:40]}...': {exc}", "TTS")
                results.append(None)
        return results

    def synthesize(self, text: str, ref_audio: Optional[str], ref_text: Optional[str], seed: int) -> Tuple[np.ndarray, int]:
        result = self.synthesize_batch([text], ref_audio, ref_text, seed)[0]
        if result is None:
            raise RuntimeError(f"Không sinh được câu: {text[:60]}")
        return result, self.sample_rate

    # ------------------------------------------------------------ reference
    @staticmethod
    def find_pauses(samples: np.ndarray, sr: int, min_pause: float = 0.15) -> List[Tuple[float, float, float]]:
        """Các chỗ ngừng: (giữa, độ dài giây, độ sạch dB so với đỉnh — càng âm càng sạch)."""
        frame = int(sr * 0.02)
        n = samples.size // frame
        energy = np.array([20 * np.log10(np.sqrt(np.mean(samples[i * frame:(i + 1) * frame] ** 2)) + 1e-9)
                           for i in range(n)])
        peak = float(energy.max()) if n else 0.0
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

    @staticmethod
    def cut_at_pause(samples: np.ndarray, sr: int, max_seconds: float = 10.0,
                     min_seconds: float = 4.0) -> Optional[np.ndarray]:
        """Chọn ĐOẠN CÂU TRỌN VẸN đẹp nhất (≤ max_seconds) làm giọng mẫu (None nếu đã đủ ngắn/không tìm được).

        Ranh giới tốt = chỗ ngừng dài (hết câu) và sạch. Đầu file bị coi là ranh giới yếu (audio
        hay bị cắt giữa câu), cuối file là ranh giới tốt. Ưu tiên đoạn dài vừa phải, có ngắt hơi bên trong.
        """
        total = samples.size / sr
        if total <= max_seconds:
            return None
        pauses = TTSEngine.find_pauses(samples, sr)
        # Ranh giới: (thời điểm, độ mạnh). Mạnh = ngừng dài + sạch.
        bounds = [(0.0, 0.15)] + [(t, length + max(0.0, (-floor - 40) / 100)) for t, length, floor in pauses]
        bounds.append((total, 0.5))
        best, best_score = None, -1e9
        for i, (a, sa) in enumerate(bounds):
            for b, sb in bounds[i + 1:]:
                dur = b - a
                if dur < min_seconds or dur > max_seconds:
                    continue
                inner = sum(1 for t, _, _ in pauses if a < t < b)      # có chỗ ngắt hơi bên trong = tự nhiên
                score = 2.0 * sa + 2.0 * sb + 0.1 * min(inner, 3) + 0.05 * dur
                if score > best_score:
                    best, best_score = (a, b), score
        if best is None:
            return None
        a, b = best
        cut = samples[int(a * sr): int(b * sr)].copy()
        fade = int(sr * 0.03)
        cut[:fade] *= np.linspace(0.0, 1.0, fade)
        cut[-fade:] *= np.linspace(1.0, 0.0, fade)
        return trim_silence(cut, sr, threshold_db=-45, lead_ms=60, tail_ms=150)

    @staticmethod
    def match_transcript(full_text: str, heard: str) -> Tuple[str, float]:
        """Đoạn liền mạch của lời mẫu (giữ chính tả, dấu câu) khớp nhất với điều ASR nghe được.

        Tìm cả điểm bắt đầu lẫn kết thúc: file .txt hay dư chữ ở đầu/cuối so với audio
        (vd audio bị cắt mất 'Ba ngày trước, ...'), lời lệch audio làm OmniVoice thừa/mất chữ đầu câu.
        """
        from core.asr_checker import char_error_rate

        words = full_text.split()
        best_text, best_err = full_text, char_error_rate(full_text, heard)
        heard_len = max(1, len(heard))
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                candidate = " ".join(words[i:j])
                if len(candidate) > heard_len * 1.6:
                    break
                if len(candidate) < heard_len * 0.5:
                    continue
                err = char_error_rate(candidate, heard)
                if err < best_err - 1e-9:
                    best_text, best_err = candidate, err
        return best_text, best_err

    @staticmethod
    def prepare_reference(src_audio: str, dst_wav: str, max_seconds: float = 20.0) -> str:
        """Đổi giọng mẫu (wav/mp3...) sang WAV mono 24 kHz, cắt lặng, chuẩn hóa âm lượng."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            tmp = f"{dst_wav}.src.wav"
            os.makedirs(os.path.dirname(dst_wav) or ".", exist_ok=True)
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src_audio,
                            "-ac", "1", "-ar", "24000", "-c:a", "pcm_f32le", tmp], check=True)
            samples, sr = read_wav(tmp)
            os.remove(tmp)
        else:
            samples, sr = read_wav(src_audio)
        samples = trim_silence(samples, sr, threshold_db=-45, lead_ms=50, tail_ms=120)
        if samples.size > sr * max_seconds:
            log(
                f"Giọng mẫu dài {samples.size / sr:.1f}s > {max_seconds}s; nên dùng mẫu 6-12s "
                "và sửa file .txt cho khớp.",
                "Cảnh báo",
            )
        samples = normalize_rms(samples, -20.0)
        write_wav(dst_wav, samples, sr, float32=True)
        return dst_wav
