"""Proje ayarları. Yollar ortam değişkenleriyle değiştirilebilir."""
import os

# Starter kit'i (yarışmanın deposu) ayrıca klonlayın, bkz. README
STARTERKIT_DIR = os.environ.get("STARTERKIT_DIR", "Starterkit-ICSE-2027")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(STARTERKIT_DIR, "data"))
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "isu_lora_adapter")

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Görüntü boyut sınırı (token sayısını ve GPU belleğini belirler)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 448 * 28 * 28

# Veri bölme: 1060 sentetik sahne -> 260 test, 800 eğitim
SEED = 42
N_TEST = 260

# Yarışmadaki 23 özellikten seçilen 11 özellik
KEEP = {
    "driver_phone", "driver_safety_belt",
    "passenger_codriver", "passenger_rear_seat_left", "passenger_rear_seat_right",
    "codriver_safety_belt", "passenger_rear_left_safety_belt",
    "passenger_rear_right_safety_belt",
    "phone_codriver_seat", "baby", "baby_seat",
}

# LoRA ayarları: sadece dil katmanlarındaki projeksiyonlara adaptör eklenir
LORA = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
)
