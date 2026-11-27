#!/usr/bin/env python3
"""Đóng gói giọng từ hồ sơ giọng TRUNG BÌNH của cả dataset (thay vì một clip lẻ).

Model fine-tune học "phiên âm + speaker embedding -> giọng". Embedding của một clip lẻ mang theo
đặc điểm riêng của câu đó (cảm xúc, độ to, nhiễu); trung bình của cả dataset là "giọng gốc" ổn
định hơn -> các câu sinh ra đồng đều hơn. Độ lớn vector được giữ bằng độ lớn trung bình của các
clip (model chỉ thấy embedding cỡ đó lúc train).

    python finetune/make_mean_voice.py --data dataset/train.parquet --name "Giọng TB" --out output/run/merged
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="train.parquet (cột speaker_embedding)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="thư mục model merged (ghi voices_v3_turbo.json)")
    ap.add_argument("--default", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    embs = np.asarray(pq.read_table(args.data, columns=["speaker_embedding"]).column(0).to_pylist(),
                      dtype=np.float64)
    norms = np.linalg.norm(embs, axis=1)
    mean = embs.mean(axis=0)
    mean *= norms.mean() / (np.linalg.norm(mean) + 1e-12)
    path = Path(args.out)
    if path.suffix != ".json":
        path = path / "voices_v3_turbo.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"presets": {}}
    data.setdefault("presets", {})[args.name] = {
        "description": f"Trung bình {len(embs)} clip dataset", "gender": "",
        "speaker_emb": [round(float(x), 6) for x in mean], "codes": None,
    }
    if args.default or not data.get("default_voice"):
        data["default_voice"] = args.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"voice '{args.name}' (trung bình {len(embs)} clip) written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
