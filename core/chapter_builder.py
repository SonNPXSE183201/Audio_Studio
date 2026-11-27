"""Đọc cả chương bằng VieNeu: chunk -> sinh -> (tùy chọn) nghe lại -> ghép -> hậu kỳ.

Chunk, khoảng nghỉ và cách ghép đều theo VieNeu (xem core/vieneu_engine.py). Phần riêng
của tool chỉ còn:
  * làm sạch file chương + thay từ theo glossary.json (dữ liệu của bộ truyện);
  * nghe lại bằng ASR (asr.enabled): chunk sai chữ / lặp từ thì sinh lại, giữ bản tốt nhất;
  * hậu kỳ cả chương (lọc ù, chuẩn LUFS, limiter) như trước.
"""
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.asr_checker import ASRChecker, char_error_rate, repeated_words
from core.audio_master import master_wav
from core.audio_utils import peak_limit, write_wav
from core.utils import fingerprint, log, read_json, write_json
from core.vieneu_engine import VieneuEngine, chunk_text, join_chunks

BUILD_VERSION = 1


class ChapterBuilder:
    def __init__(self, tts: VieneuEngine, asr: ASRChecker, tts_cfg: Dict[str, Any],
                 asr_cfg: Dict[str, Any], master_cfg: Dict[str, Any]):
        self.tts = tts
        self.asr = asr
        self.cfg = tts_cfg
        self.asr_cfg = asr_cfg
        self.master_cfg = master_cfg
        self.ignore_terms: List[str] = []
        self.force = False

    # ------------------------------------------------------------ check
    def _verdict(self, text: str, hyp: Optional[str]) -> Dict[str, Any]:
        if hyp is None:
            return {"ok": True, "cer": None, "asr_missing": self.asr.enabled}
        cer = char_error_rate(text, hyp, self.ignore_terms)
        problems = []
        if cer > float(self.asr_cfg.get("max_cer", 0.12)):
            problems.append(f"sai {cer:.0%} ký tự")
        if repeated_words(hyp, text):
            problems.append("lặp từ")
        return {"ok": not problems, "cer": round(cer, 4), "asr": hyp, "problems": problems}

    @staticmethod
    def _better(new: Dict[str, Any], old: Dict[str, Any]) -> bool:
        if new["ok"] != old["ok"]:
            return new["ok"]
        return (new.get("cer") or 0.0) < (old.get("cer") or 0.0)

    def render(self, chunks: Sequence[str]) -> Dict[str, Any]:
        """Sinh mọi chunk; ASR bật thì đọc lại chunk lỗi (tối đa asr.max_attempts lượt)."""
        wavs, tts_sec = self.tts.synthesize_chunks(list(chunks))
        stats = {"tts_first_sec": tts_sec, "tts_sec": tts_sec, "asr_sec": 0.0, "regenerated": 0, "attempts": 1}
        checks: List[Dict[str, Any]] = [{"ok": True, "cer": None} for _ in chunks]
        first_wavs = list(wavs)
        if not self.asr.enabled:
            return {"wavs": wavs, "checks": checks, "first_wavs": first_wavs, "first_checks": list(checks),
                    "stats": stats}
        first_checks: List[Dict[str, Any]] = []
        todo = list(range(len(chunks)))
        candidates: Dict[int, np.ndarray] = {}
        attempts = max(1, int(self.asr_cfg.get("max_attempts", 3)))
        for attempt in range(1, attempts + 1):
            started = time.time()
            hyps = self.asr.transcribe_batch([(candidates.get(i, wavs[i]), self.tts.sample_rate) for i in todo])
            stats["asr_sec"] += time.time() - started
            failed = []
            for i, hyp in zip(todo, hyps):
                verdict = self._verdict(chunks[i], hyp)
                verdict["attempt"] = attempt
                if attempt == 1 or self._better(verdict, checks[i]):
                    if attempt > 1:
                        wavs[i] = candidates[i]
                    checks[i] = verdict
                if not checks[i]["ok"]:
                    failed.append(i)
            if attempt == 1:
                first_checks = [dict(c) for c in checks]
            if not failed or attempt == attempts:
                break
            stats["attempts"] = attempt + 1
            stats["regenerated"] += len(failed)
            log(f"  lượt {attempt + 1}: đọc lại {len(failed)} chunk nghi lỗi", "TTS")
            new_wavs, tts_sec = self.tts.synthesize_chunks([chunks[i] for i in failed])
            stats["tts_sec"] += tts_sec
            candidates = dict(zip(failed, new_wavs))
            todo = failed
        return {"wavs": wavs, "checks": checks, "first_wavs": first_wavs, "first_checks": first_checks,
                "stats": stats}

    # ------------------------------------------------------------ chapter
    def build(self, text: str, voice_fp: str, out_wav: str, timeline_path: str) -> Dict[str, Any]:
        chunks, gaps = chunk_text(text, int(self.cfg.get("max_chars", 256)))
        audio_fp = fingerprint(BUILD_VERSION, voice_fp, self.tts.sampling(), self.master_cfg,
                               bool(self.asr.enabled), chunks)
        cached = read_json(timeline_path)
        if cached and cached.get("audio_fp") == audio_fp and os.path.exists(out_wav) and not self.force:
            log(f"Audio chương đã có ({cached['total_duration_sec']:.1f}s), dùng cache.", "TTS")
            return cached
        if not chunks:
            raise RuntimeError("Chương không còn chữ nào để đọc sau khi chuẩn hóa.")
        log(f"{len(chunks)} chunk (tối đa {self.cfg.get('max_chars', 256)} ký tự)", "TTS")
        result = self.render(chunks)
        wavs, checks = result["wavs"], result["checks"]
        empty = [i for i, w in enumerate(wavs) if w.size == 0]
        if empty:
            raise RuntimeError(f"VieNeu trả audio rỗng cho {len(empty)} chunk (vd: «{chunks[empty[0]][:60]}»)")
        full = join_chunks(wavs, gaps)
        sr = self.tts.sample_rate
        raw_out = f"{out_wav}.raw.wav"
        write_wav(raw_out, full, sr, float32=True)
        if not master_wav(raw_out, out_wav, self.master_cfg):
            write_wav(out_wav, peak_limit(full, 0.89), sr)
        os.remove(raw_out)
        warnings = [{"chunk": i, "text": chunks[i], **{k: v for k, v in c.items() if k != "ok"}}
                    for i, c in enumerate(checks) if not c.get("ok", True)]
        timeline = {
            "audio_fp": audio_fp,
            "engine": f"vieneu-v3turbo/{self.tts.backend}",
            "sample_rate": sr,
            "total_duration_sec": full.size / float(sr),
            "chunks": len(chunks),
            "stats": {k: round(v, 2) if isinstance(v, float) else v for k, v in result["stats"].items()},
            "cer_mean": _mean([c.get("cer") for c in checks]),
            "warnings": warnings,
        }
        write_json(timeline_path, timeline)
        if warnings:
            log(f"{len(warnings)} chunk vẫn nghi lỗi sau khi đọc lại (xem {timeline_path}).", "Cảnh báo")
        return timeline


def _mean(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None
