"""Ortak yardımcılar: veri bölme, model yükleme, çıkarım ve değerlendirme."""
import collections
import os
import random
import sys
import time

import pandas as pd
import torch
from PIL import Image

import config

# Yarışmanın değerlendirme kodu starter kit'ten gelir
sys.path.insert(0, os.path.join(config.STARTERKIT_DIR, "track_b"))
import track_b_evaluator as ev  # noqa: E402

IGNORED = set(ev.DEFAULT_IGNORED_FEATURES) | (set(ev.TRACK_B_FEATURES) - config.KEEP)


def load_splits():
    """Sabit tohumla 800 eğitim / 260 test ayrımı ve istemi (prompt) döndürür."""
    scenes = ev.load_dataset(config.DATA_DIR, "synthetic")
    random.Random(config.SEED).shuffle(scenes)
    test = scenes[: config.N_TEST]
    train = scenes[config.N_TEST:]
    vocabulary = ev.build_vocabulary(scenes, IGNORED)
    prompt = ev.build_prompt(vocabulary)
    return train, test, prompt


def load_model(adapter_dir=None, trainable=False):
    """Qwen2.5-VL'i yükler; adapter_dir verilirse LoRA adaptörünü de takar."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME, dtype=torch.float16, device_map="cuda:0")
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    model.eval()
    return processor, model


def ask(model, processor, prompt, image):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    out = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def evaluate(model, processor, prompt, scene_list):
    """Özellik bazlı doğruluk, çoğunluk sınıfı karşılaştırması ve tam eşleşme."""
    hits = collections.defaultdict(lambda: [0, 0])
    gold = collections.defaultdict(list)
    pred_yes = collections.defaultdict(list)
    exact, parse_fail, times, rows = 0, 0, [], []

    for i, (stem, path, label) in enumerate(scene_list, 1):
        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        pred = ev.parse_json_answer(ask(model, processor, prompt, img)) or {}
        times.append(time.perf_counter() - t0)
        parse_fail += (not pred)

        expected = ev.comparable_features(label, IGNORED)
        all_ok = True
        for key, value in expected.items():
            p = str(pred.get(key)).strip().upper()
            g = str(value).strip().upper()
            hit = p == g
            hits[key][0] += hit
            hits[key][1] += 1
            gold[key].append(g)
            pred_yes[key].append(p == "YES")
            all_ok &= hit
        exact += all_ok
        rows.append({"stem": stem, "expected": expected, "prediction": pred})
        if i % 20 == 0:
            print(f"{i}/{len(scene_list)} bitti", flush=True)

    table = pd.DataFrame([
        (k, c / n,
         max(collections.Counter(gold[k]).values()) / n,
         sum(pred_yes[k]) / n, n)
        for k, (c, n) in sorted(hits.items())
    ], columns=["ozellik", "dogruluk", "cogunluk_sinifi", "tahmin_YES_orani", "ornek"])

    print(table.to_string(index=False))
    print(f"\nSahne düzeyinde tam eşleşme: {exact / len(scene_list):.3f}")
    print(f"JSON ayrıştırma hatası: {parse_fail}")
    print(f"Ortalama süre: {sum(times) / len(times):.2f} sn/görüntü")
    summary = {"exact_match": exact / len(scene_list), "parse_fail": parse_fail,
               "sec_per_image": sum(times) / len(times), "n": len(scene_list)}
    return table, rows, summary
