"""LoRA ince ayarı. Adaptör klasöründe kayıt varsa eğitim oradan devam eder.

Örnek:
    python train_lora.py --updates 60
Bir güncelleme = accum x 1 örnek (varsayılan 8 görüntü).
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image

import common
import config


def make_target(label):
    """Modelin üretmesi gereken doğru JSON cevabı."""
    exp = common.ev.comparable_features(label, common.IGNORED)
    return json.dumps({k: str(v).upper() for k, v in sorted(exp.items())})


def build_example(processor, prompt, path, label):
    img = Image.open(path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "image", "image": img}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_enc = processor(text=[text], images=[img], return_tensors="pt")
    full_enc = processor(text=[text + make_target(label) + "<|im_end|>"],
                         images=[img], return_tensors="pt")
    labels = full_enc["input_ids"].clone()
    labels[:, : prompt_enc["input_ids"].shape[1]] = -100  # kayıp sadece cevap kısmında
    out = dict(full_enc)
    out["labels"] = labels
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--adapter-dir", default=config.ADAPTER_DIR)
    ap.add_argument("--seed", type=int, default=None, help="boşsa zamana göre")
    args = ap.parse_args()

    train_scenes, _, prompt = common.load_splits()
    processor, model = common.load_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    os.makedirs(args.adapter_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.adapter_dir, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=True)
        print("Kayıtlı adaptör yüklendi, eğitim kaldığı yerden sürecek.")
    else:
        model = get_peft_model(model, LoraConfig(**config.LORA))
        print("Kayıtlı adaptör yok, sıfırdan başlanıyor.")
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    updates = args.updates
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / updates))
    scaler = torch.amp.GradScaler("cuda")

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    idx = []
    while len(idx) < updates * args.accum:
        perm = list(range(len(train_scenes)))
        rng.shuffle(perm)
        idx += perm
    idx = idx[: updates * args.accum]

    model.train()
    model.config.use_cache = False
    t0, run_loss = time.time(), 0.0
    for i, j in enumerate(idx, 1):
        _, path, label = train_scenes[j]
        batch = {k: v.to(model.device)
                 for k, v in build_example(processor, prompt, path, label).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss / args.accum
        scaler.scale(loss).backward()
        run_loss += loss.item()

        if i % args.accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            torch.cuda.empty_cache()
            u = i // args.accum
            if u % 5 == 0 or u == updates:
                per = (time.time() - t0) / i
                print(f"update {u}/{updates} | loss {run_loss:.4f} | "
                      f"kalan ~{per * (len(idx) - i) / 60:.0f} dk", flush=True)
            if u % args.save_every == 0 or u == updates:
                model.save_pretrained(args.adapter_dir)
                with open(os.path.join(args.adapter_dir, "progress.txt"), "a") as f:
                    f.write(f"{time.ctime()} update {u}/{updates}\n")
            run_loss = 0.0
    print("Bitti. Adaptör kaydedildi:", args.adapter_dir)


if __name__ == "__main__":
    main()
