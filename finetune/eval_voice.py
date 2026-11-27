#!/usr/bin/env python3
"""So sánh giọng: clone tức thì (model gốc) vs các model đã fine-tune, trên cùng các câu truyện thật.

Chỉ số (cao = tốt, trừ CER):
  giong_mau   cosine hồ sơ giọng của từng câu sinh ra vs hồ sơ giọng của file mẫu (giọng thật tự so: ~0,89)
  dong_deu    cosine trung bình giữa các câu với nhau (câu này có cùng giọng với câu kia không)
  cer         tỷ lệ sai ký tự khi PhoWhisper nghe lại (đọc sai / nuốt chữ / lặp)
Mỗi model ghi một file <nhãn>.wav (các câu nối nhau) để nghe so sánh.

    python finetune/eval_voice.py --ref voice.wav --chapter Chuong_0484.txt --out eval \\
        --model goc=clone:voice.wav --model e3=/path/merged:"Giọng đọc truyện"
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.asr_checker import ASRChecker, char_error_rate  # noqa: E402
from core.audio_utils import read_wav, write_wav  # noqa: E402
from core.text_processor import chapter_text  # noqa: E402
from core.utils import log  # noqa: E402
from core.vieneu_engine import VieneuEngine, chunk_text, cosine, join_chunks  # noqa: E402
from finetune.speech_quality import dnsmos  # noqa: E402


def parse_model(spec: str):
    label, _, rest = spec.partition("=")
    repo, _, voice = rest.partition(":")
    if repo == "clone":
        return {"label": label, "clone": voice}
    return {"label": label, "backbone_repo": repo, "voice": voice}


def pick_best(results):
    """Model nên dùng: giống giọng nhất trong các model KHÔNG kém bản đầu tiên (clone tức thì) về
    độ rõ chữ (CER tăng <= 1 điểm %) và độ sạch giọng (SIG thấp nhất giảm <= 0,1). Không có thì None."""
    if not results:
        return None
    base = results[0]
    ok = []
    for r in results[1:]:
        if r["cer"] > base["cer"] + 0.01:
            continue
        if r.get("sig_thap_nhat") is not None and base.get("sig_thap_nhat") is not None \
                and r["sig_thap_nhat"] < base["sig_thap_nhat"] - 0.1:
            continue
        if r["giong_mau"] > base["giong_mau"]:
            ok.append(r)
    return max(ok, key=lambda r: r["giong_mau"])["label"] if ok else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="file giọng thật (làm mốc độ giống giọng)")
    ap.add_argument("--chapter", required=True, help="chương truyện .txt để lấy câu thử")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="nhãn=clone:<giọng mẫu.wav>  hoặc  nhãn=<thư mục model merged>:<tên giọng>")
    ap.add_argument("--sentences", type=int, default=12)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-large")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.chapter, encoding="utf-8") as handle:
        chunks, _ = chunk_text(chapter_text(handle.read()))
    chunks = [c for c in chunks if len(c) >= 60][: args.sentences]
    ref, ref_sr = read_wav(args.ref)

    results, audio = [], {}
    target = None
    for spec in args.model:
        m = parse_model(spec)
        cfg = {"backend": args.backend}
        if "backbone_repo" in m:
            cfg["backbone_repo"] = m["backbone_repo"]
        engine = VieneuEngine(cfg)
        engine.load()
        if target is None:     # mốc giọng thật: đo bằng bộ trích đặc trưng của model gốc lẫn fine-tune (giống nhau)
            target = engine.reference_profile(ref, ref_sr)
        if "clone" in m:
            engine.set_voice(m["clone"])
        else:
            engine.use_preset(m["voice"])
        engine.synthesize_chunks(chunks[:1])          # khởi động
        started = time.time()
        wavs, _ = engine.synthesize_chunks(chunks)
        sec = time.time() - started
        embs = [engine.speaker_embedding(w, engine.sample_rate) for w in wavs]
        sims = [cosine(e, target) for e in embs]
        inter = [cosine(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
        audio_sec = sum(w.size for w in wavs) / engine.sample_rate
        qs = [q for q in (dnsmos(w, engine.sample_rate) for w in wavs) if q]
        write_wav(os.path.join(args.out, f"{m['label']}.wav"),
                  join_chunks(wavs, ["sentence"] * (len(wavs) - 1)), engine.sample_rate)
        audio[m["label"]] = wavs
        results.append({"label": m["label"], "giong_mau": round(float(np.mean(sims)), 3),
                        "giong_mau_thap_nhat": round(float(np.min(sims)), 3),
                        "dong_deu": round(float(np.mean(inter)), 3),
                        "nhanh_gap_x": round(audio_sec / max(sec, 1e-6), 1), "backend": engine.backend,
                        "sig": round(float(np.mean([q["sig"] for q in qs])), 2) if qs else None,
                        "sig_thap_nhat": round(float(np.min([q["sig"] for q in qs])), 2) if qs else None,
                        "ovrl": round(float(np.mean([q["ovrl"] for q in qs])), 2) if qs else None})
        log(json.dumps(results[-1], ensure_ascii=False), "Eval")
        engine.unload()

    asr = ASRChecker({"model_id": args.asr_model, "batch_size": 4})
    for r in results:
        hyps = asr.transcribe_batch([(w, 48000) for w in audio[r["label"]]])
        cers = [char_error_rate(t, h) if h is not None else 1.0 for t, h in zip(chunks, hyps)]
        r["cer"] = round(float(np.mean(cers)), 4)
        r["cer_cau_te_nhat"] = round(float(np.max(cers)), 4)
    asr.unload()

    ref_q = dnsmos(ref, ref_sr)
    best = pick_best(results)
    with open(os.path.join(args.out, "ket_qua.json"), "w", encoding="utf-8") as handle:
        json.dump({"cau": chunks, "ket_qua": results, "dnsmos_giong_mau": ref_q, "goi_y": best}, handle,
                  ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "goi_y.txt"), "w", encoding="utf-8") as handle:
        handle.write((best or "") + "\n")
    print(f"\n{'model':<14}{'giống giọng':>12}{'thấp nhất':>11}{'đồng đều':>10}{'CER':>8}"
          f"{'SIG':>7}{'SIG min':>9}{'OVRL':>7}")
    for r in results:
        sig = f"{r['sig']:>7.2f}{r['sig_thap_nhat']:>9.2f}{r['ovrl']:>7.2f}" if r.get("sig") else ""
        print(f"{r['label']:<14}{r['giong_mau']:>12.3f}{r['giong_mau_thap_nhat']:>11.3f}{r['dong_deu']:>10.3f}"
              f"{r['cer']:>8.1%}{sig}")
    print("(mốc: giọng thật tự so với chính nó ~0,89"
          + (f"; DNSMOS giọng mẫu SIG {ref_q['sig']:.2f} OVRL {ref_q['ovrl']:.2f}" if ref_q else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
