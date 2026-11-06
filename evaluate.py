"""Değerlendirme. Adaptör verilmezse eğitimsiz (baseline) model ölçülür.

Örnekler:
    python evaluate.py --n 100 --out results/baseline
    python evaluate.py --adapter isu_lora_adapter --n 260 --out results/lora
"""
import argparse
import json
import os

import common

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None, help="LoRA adaptör klasörü (yoksa baseline)")
ap.add_argument("--n", type=int, default=100, help="test sahnesi sayısı (en fazla 260)")
ap.add_argument("--out", default="results/run", help="çıktı dosya öneki")
args = ap.parse_args()

_, test_scenes, prompt = common.load_splits()
processor, model = common.load_model(adapter_dir=args.adapter)
table, rows, summary = common.evaluate(model, processor, prompt, test_scenes[: args.n])

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
table.to_csv(args.out + "_ozellik_tablosu.csv", index=False)
with open(args.out + "_tahminler.json", "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=1)
print("Kaydedildi:", args.out + "_*")
