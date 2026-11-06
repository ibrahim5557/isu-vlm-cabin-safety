"""Veri setini Hugging Face'ten indirir (10 dakikadan uzun sürebilir)."""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 429 hız sınırı hatalarını azaltır

from huggingface_hub import snapshot_download

import config

snapshot_download(
    repo_id="ISU-Test/isu-challenge-dataset",
    repo_type="dataset",
    local_dir=config.DATA_DIR,
    max_workers=2,
    # Bu projede kullanılmayan yardımcı kanallar
    ignore_patterns=["depth/*", "canny/*", "instance_seg/*"],
)
print("Veri indirildi:", config.DATA_DIR)
