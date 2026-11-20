"""Sinh giọng đọc cả chương: đọc theo batch, nghe lại để đọc lại câu vấp, nối mượt, hậu kỳ.

Mỗi lượt:
  1. Gom các câu chưa đạt thành batch (xếp theo độ dài để ít padding) -> OmniVoice sinh một lượt.
  2. Hậu kỳ nhẹ từng câu (cắt lặng nhẹ tay, cân bằng độ to phần tiếng nói).
  3. Kiểm tra: tốc độ đọc, khoảng câm giữa câu, đuôi câu cụt; ASR nghe lại theo batch
     (sai chữ, lặp từ).
  4. Câu chưa đạt sang lượt sau với seed khác. Hết lượt mà vẫn hỏng thì tách đôi câu.
Không bao giờ bỏ qua một câu: lỗi hẳn thì dừng chương, không xuất audio thiếu câu.

Ghép: fade nửa cos ở hai đầu câu, khoảng nghỉ lấp "không khí phòng" rất nhỏ thay vì im lặng
tuyệt đối, khoảng nghỉ ngắn thì chồng phần lặng của hai câu lên nhau. Cuối cùng hậu kỳ cả
chương (lọc ù, chuẩn LUFS bằng gain cố định + limiter).
"""
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.asr_checker import ASRChecker, char_error_rate, repeated_words
from core.audio_master import master_wav
from core.audio_utils import (
    apply_fade,
    duration_ms,
    longest_internal_gap_ms,
    median_f0,
    normalize_voiced,
    peak_limit,
    read_wav,
    resample,
    room_tone,
    semitones,
    suppress_breaths,
    tail_level_db,
    trim_silence,
    write_wav,
)
from core.tts_engine import TTSEngine
from core.utils import fingerprint, log, read_json, write_json
from core.vi_normalizer import speakable_length

POST_DEFAULTS: Dict[str, Any] = {
    "trim_silence_db": -50,   # ngưỡng cắt lặng so với đỉnh (thấp = giữ phụ âm nhẹ, đuôi thanh điệu)
    "lead_ms": 80,            # giữ trước tiếng đầu tiên
    "tail_ms": 180,           # giữ sau tiếng cuối cùng
    "target_db": -20,         # độ to phần tiếng nói của mọi câu
    "max_gain_db": 6,
    "fade_in_ms": 15,
    "fade_out_ms": 40,
    "room_tone_db": None,     # None = khoảng nghỉ im lặng tuyệt đối (như AI_Video_Studio)
    "suppress_breaths": True,  # giảm tiếng lấy hơi model tự sinh (nhiều khi speed < 1)
    "breath_reduce_db": -30,
}


class AudioBuilder:
    SYNTH_VERSION = 4   # đổi khi cách sinh/lưu câu thô thay đổi (sẽ đọc lại toàn bộ)
    POST_VERSION = 1    # đổi khi cách ghép thay đổi (chỉ ghép lại, không đọc lại)

    def __init__(
        self,
        tts: TTSEngine,
        asr: ASRChecker,
        tts_cfg: Dict[str, Any],
        asr_cfg: Dict[str, Any],
        cache_dir: str,
        master_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.tts = tts
        self.asr = asr
        self.cfg = tts_cfg
        self.asr_cfg = asr_cfg
        self.master_cfg = master_cfg or {}
        self.cache_dir = os.path.join(cache_dir, "tts")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.force = False                  # True: bỏ qua cache câu, đọc lại toàn bộ
        self.ignore_terms: List[str] = []   # tên riêng không tính vào tỷ lệ sai của ASR
        self.post = {k: self.cfg.get(k, v) for k, v in POST_DEFAULTS.items()}
        self.ref_f0: Optional[float] = None   # cao độ trung vị của giọng mẫu (Hz)
        self._ref_f0_path: Optional[str] = None

    def set_reference(self, ref_audio: Optional[str]) -> None:
        """Đo cao độ giọng mẫu: câu đọc lệch giọng mẫu quá nhiều sẽ được đọc lại."""
        if ref_audio == self._ref_f0_path:
            return
        self._ref_f0_path = ref_audio
        self.ref_f0 = None
        if ref_audio and os.path.exists(ref_audio):
            # Mức giọng chung đo lúc thử giọng (cao độ trung vị các câu model đọc ra). So với mức này
            # thì "lệch giọng" = câu cao/thấp hơn các câu khác, không phạt cả giọng cao hơn mẫu đều đặn.
            meta = read_json(os.path.splitext(ref_audio)[0] + ".json") or {}
            if meta.get("pitch_target_hz"):
                self.ref_f0 = float(meta["pitch_target_hz"])
                source = "mức giọng chung khi thử giọng"
            else:
                samples, sr = read_wav(ref_audio)
                self.ref_f0 = median_f0(samples, sr)
                source = "giọng mẫu"
            if self.ref_f0:
                log(f"Cao độ chuẩn: {self.ref_f0:.0f} Hz ({source}; câu lệch quá "
                    f"{self.asr_cfg.get('max_pitch_dev_st', 1.5)} bán cung sẽ được đọc lại)", "TTS")

    # ------------------------------------------------------------ fingerprints
    def _synth_fp(self, voice_fp: str) -> str:
        keys = ("model_id", "num_step", "guidance_scale", "speed", "t_shift", "denoise", "seed", "dtype", "language",
                "preprocess_prompt", "position_temperature", "class_temperature")
        return fingerprint(self.SYNTH_VERSION, voice_fp, {k: self.cfg.get(k) for k in keys})

    def _paths(self, synth_fp: str, text: str) -> Tuple[str, str]:
        key = fingerprint(synth_fp, text)
        return os.path.join(self.cache_dir, f"{key}.wav"), os.path.join(self.cache_dir, f"{key}.json")

    # ------------------------------------------------------------ chunk audio
    def postprocess(self, samples: np.ndarray, sr: int) -> np.ndarray:
        p = self.post
        if p.get("suppress_breaths", True):
            # Trước khi cắt lặng: tiếng hít vào ở đầu câu thành im lặng -> bị cắt bỏ luôn.
            samples = suppress_breaths(samples, sr, float(p.get("breath_reduce_db", -30)))
        out = trim_silence(samples, sr, float(p["trim_silence_db"]), int(p["lead_ms"]), int(p["tail_ms"]))
        return normalize_voiced(out, sr, float(p["target_db"]), float(p["max_gain_db"]))

    def _signal_check(self, text: str, raw: np.ndarray, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        dur_s = samples.size / float(sr) if sr else 0.0
        chars = speakable_length(text)
        cps = chars / dur_s if dur_s > 0 else 0.0
        problems: List[str] = []
        if dur_s <= 0.25:
            problems.append("rỗng")
        cps_min = float(self.asr_cfg.get("cps_min", 7))
        cps_max = float(self.asr_cfg.get("cps_max", 28))
        if chars >= 8 and not (cps_min <= cps <= cps_max):
            problems.append(f"tốc độ {cps:.1f} ký tự/s")
        gap = longest_internal_gap_ms(samples, sr)
        if gap > int(self.asr_cfg.get("max_gap_ms", 1300)):
            problems.append(f"khoảng câm {gap}ms giữa câu")
        if tail_level_db(raw, sr) > float(self.asr_cfg.get("max_tail_db", -18)):
            problems.append("đuôi câu bị cụt")
        pitch_dev = None
        f0 = median_f0(samples, sr) if self.ref_f0 else None
        if f0:
            pitch_dev = round(semitones(f0, self.ref_f0), 2)
            limit = float(self.asr_cfg.get("max_pitch_dev_st", 1.5))
            if text.rstrip(" \"'”’»)").endswith(("?", "!")):
                limit += 1.0   # câu hỏi/cảm thán vốn cao giọng hơn
            if chars >= 12 and abs(pitch_dev) > limit:
                problems.append(f"lệch giọng {pitch_dev:+.1f} bán cung")
        return {"duration_s": round(dur_s, 3), "cps": round(cps, 2), "max_gap_ms": gap,
                "pitch_dev_st": pitch_dev, "cer": None, "asr": None, "problems": problems}

    def _asr_check(self, text: str, hyp: Optional[str], check: Dict[str, Any]) -> Dict[str, Any]:
        if hyp is not None:
            cer = char_error_rate(text, hyp, self.ignore_terms)
            check["cer"], check["asr"] = round(cer, 3), hyp
            if cer > float(self.asr_cfg.get("max_cer", 0.12)):
                check["problems"].append(f"sai {cer:.0%} ký tự")
            if repeated_words(hyp, text):
                check["problems"].append("lặp từ")
        elif self.asr.enabled:
            # ASR lỗi (vd hết VRAM): câu này CHƯA được nghe lại -> ghi cảnh báo để kiểm tra, không coi là đạt hoàn toàn.
            check["asr_missing"] = True
        check["ok"] = not check["problems"]
        return check

    @staticmethod
    def _score(check: Dict[str, Any]) -> float:
        """Thấp hơn = tốt hơn: đạt trước, rồi ít lỗi, ít sai chữ, gần cao độ giọng mẫu."""
        cer = check.get("cer")
        pitch = abs(check.get("pitch_dev_st") or 0.0)
        return (0 if check.get("ok") else 10) + len(check.get("problems", []))             + (cer if cer is not None else 0.5) + 0.3 * pitch

    def _batches(self, texts: Sequence[str]) -> List[List[int]]:
        """Chia chỉ số câu thành batch theo số câu và tổng ký tự (giới hạn VRAM)."""
        size = max(1, int(self.cfg.get("batch_size", 8)))
        max_chars = max(1, int(self.cfg.get("batch_max_chars", 1800)))
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batches: List[List[int]] = []
        current: List[int] = []
        total = 0
        for i in order:
            if current and (len(current) >= size or total + len(texts[i]) > max_chars):
                batches.append(current)
                current, total = [], 0
            current.append(i)
            total += len(texts[i])
        if current:
            batches.append(current)
        return batches

    def synthesize_texts(self, texts: Sequence[str], ref_audio, ref_text, label: str = "") -> List[Optional[Tuple[np.ndarray, Dict[str, Any]]]]:
        """Sinh nhiều câu, đọc lại câu lỗi qua nhiều lượt. Trả về (audio thô, kết quả kiểm tra) mỗi câu."""
        attempts = max(1, int(self.asr_cfg.get("max_attempts", 3)))
        base_seed = int(self.cfg.get("seed", 1234))
        sr = self.tts.sample_rate
        self.set_reference(ref_audio)
        best: List[Optional[Tuple[np.ndarray, Dict[str, Any]]]] = [None] * len(texts)
        pending = list(range(len(texts)))
        for attempt in range(attempts):
            if not pending:
                break
            seed = base_seed + attempt * 7919
            produced: List[Tuple[int, np.ndarray, np.ndarray, Dict[str, Any]]] = []
            for batch in self._batches([texts[i] for i in pending]):
                ids = [pending[b] for b in batch]
                outputs = self.tts.synthesize_batch([texts[i] for i in ids], ref_audio, ref_text, seed)
                sr = self.tts.sample_rate
                for i, raw in zip(ids, outputs):
                    if raw is None or raw.size == 0:
                        continue
                    post = self.postprocess(raw, sr)
                    produced.append((i, raw, post, self._signal_check(texts[i], raw, post, sr)))
            hyps = self.asr.transcribe_batch([(p[2], sr) for p in produced]) if self.asr.enabled else [None] * len(produced)
            still: List[int] = []
            for (i, raw, _post, check), hyp in zip(produced, hyps):
                check = self._asr_check(texts[i], hyp, check)
                check.update({"attempt": attempt + 1, "seed": seed})
                if best[i] is None or self._score(check) < self._score(best[i][1]):
                    best[i] = (raw, check)
            for i in pending:
                if best[i] is None or not best[i][1]["ok"]:
                    still.append(i)
            if still:
                # Thống kê lý do (sai chữ / lệch giọng / tốc độ...) để biết vì sao phải đọc lại.
                reasons: Dict[str, int] = {}
                for i in still:
                    for p in (best[i][1].get("problems") if best[i] else ["không sinh được"]) or []:
                        key = re.sub(r"[-+]?\d+(\.\d+)?%?", "#", p)
                        reasons[key] = reasons.get(key, 0) + 1
                top = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:5])
                log(f"{label}Lượt {attempt + 1}: {len(still)}/{len(texts)} câu chưa đạt ({top})", "TTS")
                for i in still[:3]:
                    c = best[i][1] if best[i] else {}
                    log(f"   ví dụ: «{texts[i][:70]}» -> ASR «{str(c.get('asr'))[:70]}» {c.get('problems')}", "TTS")
            if still and attempt + 1 < attempts:
                log(f"{label}Đọc lại {len(still)} câu, lượt {attempt + 2} (seed khác)...", "TTS")
            pending = still
        return best

    @staticmethod
    def split_half(text: str) -> Optional[Tuple[str, str]]:
        """Tách câu làm đôi ở dấu câu gần giữa nhất (hoặc khoảng trắng)."""
        middle = len(text) // 2
        cuts = [m.end() for m in re.finditer(r"[,;:.!?…]\s+", text)] or [m.start() for m in re.finditer(r"\s", text)]
        cuts = [c for c in cuts if 8 <= c <= len(text) - 8]
        if not cuts:
            return None
        cut = min(cuts, key=lambda c: abs(c - middle))
        left, right = text[:cut].strip(), text[cut:].strip()
        if not re.search(r"[,;:.!?…]$", left):
            left += ","
        return left, right

    def _rescue(self, text: str, current, ref_audio, ref_text, depth: int = 0):
        """Câu đọc nhiều lần vẫn lỗi: tách đôi rồi đọc từng nửa, ghép bằng nhịp nghỉ dấu phẩy."""
        if depth >= 2 or len(text) < 50:
            return current
        halves = self.split_half(text)
        if not halves:
            return current
        parts = self.synthesize_texts(list(halves), ref_audio, ref_text, label="  [tách đôi] ")
        fixed = []
        for half, part in zip(halves, parts):
            if part is not None and not part[1]["ok"]:
                part = self._rescue(half, part, ref_audio, ref_text, depth + 1)
            fixed.append(part)
        if any(p is None for p in fixed):
            return current
        sr = self.tts.sample_rate
        a = apply_fade(self.postprocess(fixed[0][0], sr), sr, 10, 30)
        b = apply_fade(self.postprocess(fixed[1][0], sr), sr, 10, 30)
        joined = np.concatenate([a, np.zeros(int(sr * 0.16), dtype=np.float32), b])
        check = {
            "ok": all(p[1]["ok"] for p in fixed),
            "problems": [x for p in fixed for x in p[1].get("problems", [])],
            "cer": max((p[1]["cer"] for p in fixed if p[1].get("cer") is not None), default=None),
            "asr": " | ".join(str(p[1].get("asr") or "") for p in fixed),
            "split": True,
            "duration_s": round(joined.size / float(sr), 3),
        }
        if current is not None and not check["ok"] and self._score(current[1]) <= self._score(check):
            return current
        return joined, check

    def render_chunks(self, chunks: List[Dict[str, Any]], ref_audio, ref_text, synth_fp: str) -> List[Dict[str, Any]]:
        """Bảo đảm mọi câu có file audio thô trong cache. Trả về thông tin từng câu theo thứ tự."""
        infos: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        todo: List[int] = []
        for index, chunk in enumerate(chunks):
            wav_path, meta_path = self._paths(synth_fp, chunk["tts"])
            meta = None if self.force else read_json(meta_path)
            if meta and os.path.exists(wav_path):
                infos[index] = {"path": wav_path, "meta_path": meta_path, **meta}
            else:
                todo.append(index)
        if todo:
            log(f"Đọc {len(todo)}/{len(chunks)} câu (batch {self.cfg.get('batch_size', 8)})...", "TTS")
            window = max(1, int(self.cfg.get("save_every", 64)))
            for start in range(0, len(todo), window):
                part = todo[start:start + window]
                texts = [chunks[i]["tts"] for i in part]
                results = self.synthesize_texts(texts, ref_audio, ref_text)
                for i, text, result in zip(part, texts, results):
                    if result is not None and not result[1]["ok"]:
                        log(f"Câu {chunks[i]['id']} vẫn lỗi ({', '.join(result[1]['problems'])}), thử tách đôi.", "TTS")
                        result = self._rescue(text, result, ref_audio, ref_text)
                    if result is None:
                        raise RuntimeError(f"Không sinh được audio cho câu {chunks[i]['id']}: {text[:80]}")
                    raw, check = result
                    wav_path, meta_path = self._paths(synth_fp, text)
                    write_wav(wav_path, raw, self.tts.sample_rate, float32=True)
                    meta = dict(check, text=text)
                    write_json(meta_path, meta)
                    infos[i] = {"path": wav_path, "meta_path": meta_path, **meta}
                log(f"  {min(start + window, len(todo))}/{len(todo)} câu", "TTS")
        return [info for info in infos if info is not None]

    # ------------------------------------------------------------ chapter
    def build(
        self,
        chunks: List[Dict[str, Any]],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        voice_fp: str,
        out_wav: str,
        timeline_path: str,
    ) -> Dict[str, Any]:
        synth_fp = self._synth_fp(voice_fp)
        audio_fp = fingerprint(
            synth_fp, self.POST_VERSION, self.post, self.master_cfg,
            [(c["tts"], c["pause_after_ms"]) for c in chunks],
        )
        cached = read_json(timeline_path)
        if cached and cached.get("audio_fp") == audio_fp and os.path.exists(out_wav) and not self.force:
            log(f"Audio chương đã có ({cached['total_duration_sec']:.1f}s), dùng cache.", "TTS")
            return cached

        infos = self.render_chunks(chunks, ref_audio, ref_text, synth_fp)
        target_sr: Optional[int] = None
        rendered = []
        for chunk, info in zip(chunks, infos):
            raw, sr = read_wav(info["path"])
            if target_sr is None:
                target_sr = sr
            rendered.append((chunk, self.postprocess(resample(raw, sr, target_sr), target_sr), info))
        sr = target_sr or self.tts.sample_rate
        full, timeline_chunks, warnings = self.assemble(rendered, sr)
        raw_out = f"{out_wav}.raw.wav"
        write_wav(raw_out, full, sr, float32=True)
        if not master_wav(raw_out, out_wav, self.master_cfg):
            write_wav(out_wav, peak_limit(full, 0.89), sr)
        os.remove(raw_out)
        timeline = {
            "audio_fp": audio_fp,
            "sample_rate": sr,
            "total_duration_ms": duration_ms(full, sr),
            "total_duration_sec": full.size / float(sr),
            "chunks": timeline_chunks,
            "warnings": warnings,
            "cache_files": sorted({p for info in infos for p in (info["path"], info["meta_path"])}),
        }
        write_json(timeline_path, timeline)
        if warnings:
            log(f"{len(warnings)} câu vẫn có dấu hiệu đọc sai sau khi thử lại (xem {timeline_path}).", "Cảnh báo")
        log(f"Audio chương: {timeline['total_duration_sec']:.1f}s", "TTS")
        return timeline

    def assemble(self, rendered, sr: int):
        """Nối các câu đã hậu kỳ thành audio chương + mốc thời gian từng câu."""
        p = self.post
        lead_ms, tail_ms = int(p["lead_ms"]), int(p["tail_ms"])
        room_db = p.get("room_tone_db")
        pieces: List[np.ndarray] = []
        entries: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        cursor = 0   # vị trí (mẫu) trên trục thời gian chương
        overlap = 0  # số mẫu câu sau được chồng lên phần đuôi lặng của câu trước
        for chunk, samples, info in rendered:
            samples = apply_fade(samples, sr, float(p["fade_in_ms"]), float(p["fade_out_ms"]))
            start = cursor
            if overlap and pieces:
                prev = pieces[-1]
                n = min(overlap, prev.size, samples.size)
                prev[-n:] = prev[-n:] + samples[:n]   # cộng hai phần lặng, không mất tiếng nào
                samples = samples[n:]
                start = cursor - n
            pieces.append(samples.copy())
            cursor += samples.size
            entries.append({
                "id": chunk["id"],
                "para": chunk.get("para"),
                "display": chunk.get("display", ""),
                "start_ms": int(round(start * 1000.0 / sr)),
                "end_ms": int(round(cursor * 1000.0 / sr)),
            })
            if not info.get("ok", True) or info.get("asr_missing"):
                problems = list(info.get("problems") or [])
                if info.get("asr_missing"):
                    problems.append("chưa nghe lại được (ASR lỗi)")
                warnings.append({
                    "id": chunk["id"], "time_sec": round(start / float(sr), 1), "text": chunk.get("display", ""),
                    "problems": problems, "asr": info.get("asr"),
                })
            # Câu đã giữ sẵn ~tail_ms lặng ở cuối, câu sau giữ ~lead_ms ở đầu.
            gap_ms = int(chunk["pause_after_ms"]) - tail_ms - lead_ms
            if gap_ms > 0:
                gap = room_tone(gap_ms, sr, room_db, seed=int(chunk["id"]))
                pieces.append(gap)
                cursor += gap.size
                overlap = 0
            else:
                overlap = int(sr * min(-gap_ms, lead_ms) / 1000)
        full = np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(1, dtype=np.float32)
        return full, entries, warnings

    @staticmethod
    def cleanup_cache(timeline: Dict[str, Any]) -> int:
        """Xóa câu thô của chương đã xong (tiết kiệm ổ đĩa pod). Trả về số byte giải phóng."""
        freed = 0
        for path in timeline.get("cache_files", []):
            if os.path.exists(path):
                freed += os.path.getsize(path)
                os.remove(path)
        return freed
