#!/usr/bin/env python3
"""Dựng dataset fine-tune LoRA cho VieNeu-TTS từ audio dài của MỘT người đọc.

Đầu vào: các thư mục Demucs (mỗi nguồn có vocals.wav + no_vocals.wav), ví dụ
    python -m demucs -n htdemucs --two-stems vocals -o demucs raw/*.wav
Các bước:
  1. Cắt giọng (vocals.wav) thành clip 3-15 s ở chỗ ngắt câu (khoảng lặng >= 0,25 s).
  2. Bỏ clip còn nhạc nền: đo mức no_vocals so với vocals trong cùng đoạn (--max-music-db).
  3. Khử nhiễu bằng bộ khử nhiễu của VieNeu (giống lúc VieNeu clone giọng).
  4. Bỏ clip KHÔNG phải giọng này (intro, người khác): so hồ sơ giọng với giọng mẫu (--min-sim).
  5. PhoWhisper chép lời; bỏ clip đọc quá nhanh/chậm so với số chữ (thường là chép sai/sót).
  6. Chọn đều giữa các nguồn tới --target-minutes, ghi:
        <out>/raw_audio/*.wav, <out>/metadata.csv (file_name|text: định dạng finetune/prepare_dataset.py
        của VieNeu), <out>/review.csv (thêm điểm số để soát lời: sai tên riêng thì sửa thẳng metadata.csv).

    python finetune/build_voice_dataset.py --demucs demucs/htdemucs --ref voice_reference_clean.wav --out dataset
"""
import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.asr_checker import ASRChecker  # noqa: E402
from core.utils import log, read_json, write_json  # noqa: E402
from core.vieneu_engine import VieneuEngine, cosine  # noqa: E402

FRAME = 0.02


def mono(path: str) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    f = int(sr * FRAME)
    n = x.size // f
    rms = np.sqrt(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def split_clips(x: np.ndarray, sr: int, min_s: float, max_s: float) -> List[Tuple[int, int]]:
    """Ranh giới clip = giữa các khoảng lặng >= 0,25 s. Ưu tiên cắt ở chỗ ngừng dài (hết câu)."""
    db = frame_db(x, sr)
    speech = float(np.percentile(db, 90))
    quiet = db < speech - 30
    pauses = []          # (giữa khoảng lặng [giây], độ dài [giây])
    start = None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            length = (i - start) * FRAME
            if length >= 0.25:
                pauses.append(((start + i) / 2 * FRAME, length))
            start = None
    clips = []
    if not pauses:
        return clips
    cur = pauses[0][0]
    candidates: List[Tuple[float, float]] = []
    for t, length in pauses[1:]:
        dur = t - cur
        if dur > max_s:
            # Quá dài: cắt ở chỗ ngừng dài nhất đã gặp (nếu đủ min_s), không thì bỏ đoạn này.
            good = [c for c in candidates if c[0] - cur >= min_s]
            if good:
                end = max(good, key=lambda c: c[1])[0]
                clips.append((cur, end))
                cur = end
                candidates = [c for c in candidates if c[0] > end]
                if t - cur <= max_s:
                    candidates.append((t, length))
                    continue
            cur, candidates = t, []
            continue
        candidates.append((t, length))
        if length >= 0.45 and dur >= max(min_s, 5.0):     # hết câu, độ dài đẹp -> cắt luôn
            clips.append((cur, t))
            cur, candidates = t, []
    return [(int(a * sr), int(b * sr)) for a, b in clips if min_s <= b - a <= max_s]


def trim(x: np.ndarray, sr: int, keep: float = 0.12) -> np.ndarray:
    db = frame_db(x, sr)
    voiced = np.flatnonzero(db > float(np.max(db)) - 40)
    if voiced.size == 0:
        return x
    f = int(sr * FRAME)
    a = max(0, voiced[0] * f - int(keep * sr))
    b = min(x.size, (voiced[-1] + 1) * f + int(keep * sr))
    out = x[a:b].copy()
    n = min(int(0.01 * sr), out.size // 2)
    out[:n] *= np.linspace(0, 1, n)
    out[-n:] *= np.linspace(1, 0, n)
    return out


def finish_text(text: str) -> str:
    text = " ".join(text.split()).strip(" -")
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?…" else text + "."


def filter_clips(args, clip_dir: str):
    """Cắt + lọc nhạc + khử nhiễu + lọc giọng; mỗi clip đạt ghi ngay ra clip_dir (không giữ trong RAM)."""
    engine = VieneuEngine({"backend": "onnx"})
    engine.load()
    ref, ref_sr = mono(args.ref)
    target = engine.reference_profile(ref, ref_sr)
    denoiser = None if args.no_denoise else getattr(engine.tts.engine, "denoiser", None)
    stats: Dict[str, int] = {"tong": 0, "nhac": 0, "giong_khac": 0}
    kept: List[Dict] = []
    sources = sorted(d for d in os.listdir(args.demucs) if os.path.exists(os.path.join(args.demucs, d, "vocals.wav")))
    for src in sources:
        vocals, sr = mono(os.path.join(args.demucs, src, "vocals.wav"))
        music, _ = mono(os.path.join(args.demucs, src, "no_vocals.wav"))
        clips = split_clips(vocals, sr, args.min_sec, args.max_sec)
        log(f"{src}: {vocals.size / sr / 60:.1f} phút -> {len(clips)} clip", "Dataset")
        for idx, (a, b) in enumerate(clips):
            stats["tong"] += 1
            v, m = vocals[a:b], music[a:b]
            music_db = 20 * np.log10((np.sqrt(np.mean(m ** 2)) + 1e-9) / (np.sqrt(np.mean(v ** 2)) + 1e-9))
            if music_db > args.max_music_db:
                stats["nhac"] += 1
                continue
            wav, wsr = trim(v, sr), sr
            if denoiser is not None:
                wav, wsr = denoiser.denoise(wav, sr), 44100
            sim = cosine(engine.speaker_embedding(wav, wsr), target)
            if sim < args.min_sim:
                stats["giong_khac"] += 1
                continue
            name = f"{src}_{idx:04d}.wav"
            sf.write(os.path.join(clip_dir, name), np.asarray(wav, dtype=np.float32), wsr, subtype="PCM_16")
            kept.append({"name": name, "src": src, "idx": idx, "dur": round(len(wav) / wsr, 2),
                         "music_db": round(float(music_db), 1), "sim": round(sim, 3)})
        del vocals, music
    engine.unload()
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demucs", required=True, help="thư mục chứa <nguồn>/vocals.wav + no_vocals.wav")
    ap.add_argument("--ref", required=True, help="giọng mẫu sạch của đúng người đọc (để lọc giọng khác)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--max-music-db", type=float, default=-20.0, help="nhạc còn lại tối đa so với giọng (dB)")
    ap.add_argument("--min-sim", type=float, default=0.6, help="độ giống giọng mẫu tối thiểu")
    ap.add_argument("--target-minutes", type=float, default=30.0)
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-large")
    ap.add_argument("--no-denoise", action="store_true")
    args = ap.parse_args()

    audio_dir = os.path.join(args.out, "raw_audio")
    os.makedirs(audio_dir, exist_ok=True)
    # Bước 1-4 (cắt, lọc nhạc, khử nhiễu, lọc giọng) -> clip lưu ra đĩa ngay; chạy lại thì bỏ qua.
    clip_dir = os.path.join(args.out, "_clips")
    index_path = os.path.join(clip_dir, "index.json")
    os.makedirs(clip_dir, exist_ok=True)
    if os.path.exists(index_path):
        kept, stats = read_json(index_path)["clips"], read_json(index_path)["stats"]
        log(f"Dùng lại {len(kept)} clip đã lọc ở {clip_dir}", "Dataset")
    else:
        kept, stats = filter_clips(args, clip_dir)
        write_json(index_path, {"clips": kept, "stats": stats})
    stats["loi_chep"] = 0

    # Bước 5: chép lời từng lô nhỏ, ghi dần ra đĩa (bị dừng giữa chừng thì chạy lại làm tiếp).
    trans_path = os.path.join(clip_dir, "transcripts.json")
    texts: Dict[str, str] = read_json(trans_path, {}) or {}
    todo = [k for k in kept if k["name"] not in texts]
    if todo:
        log(f"Chép lời {len(todo)}/{len(kept)} clip bằng {args.asr_model}...", "Dataset")
        asr = ASRChecker({"model_id": args.asr_model, "batch_size": 4})
        for i in range(0, len(todo), 16):
            part = todo[i:i + 16]
            hyps = asr.transcribe_batch([mono(os.path.join(clip_dir, k["name"])) for k in part])
            for k, h in zip(part, hyps):
                if h is not None:
                    texts[k["name"]] = h
            write_json(trans_path, texts)
            log(f"  {min(i + 16, len(todo))}/{len(todo)}", "Dataset")
        asr.unload()
    good = []
    for k in kept:
        text = finish_text(texts.get(k["name"], ""))
        cps = len(text.replace(" ", "")) / k["dur"]
        if not text or not 7 <= cps <= 25:
            stats["loi_chep"] += 1
            continue
        good.append(dict(k, text=text, cps=round(cps, 1)))

    # Chia đều giữa các nguồn (nhiều buổi thu khác nhau), clip giống giọng nhất trước.
    by_src: Dict[str, List[Dict]] = {}
    for k in sorted(good, key=lambda k: -k["sim"]):
        by_src.setdefault(k["src"], []).append(k)
    chosen, total = [], 0.0
    while total < args.target_minutes * 60 and any(by_src.values()):
        for src in list(by_src):
            if by_src[src]:
                k = by_src[src].pop(0)
                chosen.append(k)
                total += k["dur"]
    chosen.sort(key=lambda k: (k["src"], k["idx"]))

    meta = os.path.join(args.out, "metadata.csv")
    review = os.path.join(args.out, "review.csv")
    with open(meta, "w", encoding="utf-8", newline="\n") as fm, \
            open(review, "w", encoding="utf-8-sig", newline="") as fr:
        w = csv.writer(fr)
        w.writerow(["file_name", "text", "giay", "giong_giong_mau", "nhac_con_lai_db", "ky_tu_moi_giay"])
        for k in chosen:
            name = k["name"]
            wav, wsr = mono(os.path.join(clip_dir, name))
            sf.write(os.path.join(audio_dir, name), wav, wsr, subtype="PCM_16")
            fm.write(f"{name}|{k['text']}\n")
            w.writerow([name, k["text"], k["dur"], k["sim"], k["music_db"], k["cps"]])
    log(f"Clip: {stats['tong']} | bỏ vì còn nhạc {stats['nhac']}, giọng khác {stats['giong_khac']}, "
        f"chép lời lỗi {stats['loi_chep']} | còn {len(good)} đạt, chọn {len(chosen)} = {total / 60:.1f} phút", "Dataset")
    log(f"Soát lời: {review} (sửa trong {meta})", "Dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
