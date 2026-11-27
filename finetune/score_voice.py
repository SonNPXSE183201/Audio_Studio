#!/usr/bin/env python3
"""Đo độ giống giọng với mốc là CHÍNH người đọc trong dataset (không phải một file mẫu lẻ).

Hồ sơ giọng mốc = trung bình hồ sơ của một nửa số clip dataset (chọn xen kẽ). Nửa còn lại là
"giọng thật chưa thấy" -> cho ra TRẦN: giọng thật tự giống chính nó bao nhiêu với cùng cách đo.
Mỗi file cần chấm được cắt cửa sổ 6 s (như clip dataset), báo trung bình / thấp nhất.

    python finetune/score_voice.py --dataset dataset --ref voice_reference_clean.wav eval/*.wav
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.audio_utils import read_wav  # noqa: E402
from core.vieneu_engine import VieneuEngine, cosine  # noqa: E402


def windows(x: np.ndarray, sr: int, sec: float = 6.0):
    n, hop = int(sec * sr), int(sec * sr)
    if x.size <= n:
        return [x]
    return [x[i:i + n] for i in range(0, x.size - n + 1, hop)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="thư mục dataset (raw_audio/*.wav)")
    ap.add_argument("--ref", help="file giọng mẫu lẻ để kiểm tra có cùng người với dataset không")
    ap.add_argument("--max-clips", type=int, default=80)
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    eng = VieneuEngine({"backend": "onnx"})
    eng.load()

    def emb(x, sr):
        e = eng.speaker_embedding(x, sr)
        return e / (np.linalg.norm(e) + 1e-12)

    clips = sorted(glob.glob(os.path.join(args.dataset, "raw_audio", "*.wav")))
    step = max(1, len(clips) // args.max_clips)
    clips = clips[::step][: args.max_clips]
    embs = [emb(*read_wav(c)) for c in clips]
    profile = np.mean(embs[0::2], axis=0)
    held = [cosine(e, profile) for e in embs[1::2]]
    print(f"Mốc: {len(embs[0::2])} clip dataset. TRẦN (giọng thật chưa thấy vs mốc): "
          f"TB {np.mean(held):.3f}, thấp nhất {np.min(held):.3f}")
    if args.ref:
        x, sr = read_wav(args.ref)
        sims = [cosine(emb(w, sr), profile) for w in windows(x, sr)]
        print(f"Giọng mẫu lẻ {os.path.basename(args.ref)} vs mốc: TB {np.mean(sims):.3f} (cùng người thì gần TRẦN)")
    print(f"\n{'file':<34}{'giống (TB)':>12}{'thấp nhất':>11}{'% trần':>9}")
    for path in args.wavs:
        x, sr = read_wav(path)
        sims = [cosine(emb(w, sr), profile) for w in windows(x, sr)]
        print(f"{os.path.basename(path):<34}{np.mean(sims):>12.3f}{np.min(sims):>11.3f}"
              f"{100 * np.mean(sims) / np.mean(held):>8.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
