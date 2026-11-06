# Görüntü-Dil Modeli ile Araç İçi Güvenlik Durumu Tespiti

Tek bir araç içi RGB görüntüden emniyet kemeri, telefon kullanımı, koltuk doluluğu ve bebek durumunu sınıflandıran bir sistem. Tasarım dersi projesi olarak [ISU-Challenge](https://isu-challenge.github.io/) (Track B) veri seti ve değerlendirme koduyla geliştirilmiştir. Temel model: `Qwen/Qwen2.5-VL-3B-Instruct`, uyarlama yöntemi: LoRA.

## Proje durumu

- Baseline (eğitimsiz) ölçüm: **tamamlandı**.
- LoRA ince ayarı: **kod hazır, eğitim sürüyor.** İnce ayar sonuçları henüz yoktur, aşağıdaki tabloya eklenecektir.
- 60 gerçek fotoğrafta test: **yapılmadı.** Gerçek görüntü etiketlerinin sahne numaralarıyla eşleşmesi önce doğrulanmalıdır.

## Seçilen 11 özellik

Veri setindeki 23 tahmin özelliğinden projenin amacına uygun olanlar seçilmiştir (`config.py` içinde `KEEP`).

| Grup | Özellikler |
|---|---|
| Sürücü | `driver_phone`, `driver_safety_belt` |
| Doluluk | `passenger_codriver`, `passenger_rear_seat_left`, `passenger_rear_seat_right` |
| Kemerler | `codriver_safety_belt`, `passenger_rear_left_safety_belt`, `passenger_rear_right_safety_belt` |
| Nesne | `phone_codriver_seat` |
| Çocuk güvenliği | `baby`, `baby_seat` |

Kapsam dışı: duygu tanıma (veri setinde yalnızca HAPPY/SERIOUS etiketi var, öfke etiketi yok) ve sürücü dikkati / bakış yönü (etiket yok).

## Veri bölme

1060 sentetik görüntü sabit tohumla (`SEED = 42`) karıştırılıp **800 eğitim / 260 test** olarak ayrılır. Test görüntüleri eğitimde kullanılmaz. 60 gerçek fotoğraf ayrıca tutulur.

## Baseline sonuçları

Eğitimsiz Qwen2.5-VL-3B, test kümesinin ilk 100 görüntüsünde ölçülmüştür. Tüm özellikler iki sınıflıdır, rastgele tahmin yaklaşık %50 verir. Örnek sayısı 100 olduğu için oranlarda yaklaşık ±5 puan belirsizlik vardır. Bu ölçüm `MAX_PIXELS = 640*28*28` ile yapılmıştır, eğitimde ise bellek için 448 kullanılır, bu yüzden son karşılaştırmada baseline aynı ayarla yeniden ölçülmelidir.

| Özellik | Baseline | Çoğunluk sınıfı | LoRA |
|---|---|---|---|
| `baby_seat` | 1,00 | 0,50 | - |
| `driver_safety_belt` | 0,79 | 0,53 | - |
| `driver_phone` | 0,78 | 0,58 | - |
| `baby` | 0,72 | 0,62 | - |
| `passenger_rear_seat_right` | 0,63 | 0,50 | - |
| `passenger_rear_seat_left` | 0,59 | 0,56 | - |
| `codriver_safety_belt` | 0,57 | 0,57 | - |
| `phone_codriver_seat` | 0,82 | 0,88 | - |
| `passenger_codriver` | 0,50 | 0,77 | - |
| `passenger_rear_left_safety_belt` | 0,48 | 0,55 | - |
| `passenger_rear_right_safety_belt` | 0,46 | 0,50 | - |

Sahne düzeyinde tam eşleşme: 0,14. Ortalama çıkarım süresi: 6,6 sn/görüntü. JSON ayrıştırma hatası: 0.

## Kurulum

NVIDIA GPU gerekir (Colab T4, 16 GB ile denenmiştir; LoRA eğitiminde tepe bellek yaklaşık 10 GB).

```bash
git clone <bu-deponun-adresi>
cd isu-vlm-cabin-safety

# Yarışmanın starter kit'i (değerlendirme kodu burada). Bu depoya kopyalanmamıştır.
git clone https://github.com/isu-challenge/Starterkit-ICSE-2027.git

pip install -r requirements.txt
python setup_data.py        # veri setini indirir (10 dk'dan uzun sürebilir)
```

Colab'da `torchao` kütüphanesinin eski sürümü `peft` ile çakışabilir. `ImportError: ... torchao` hatası alırsanız `pip uninstall -y torchao` çalıştırın.

## Kullanım

```bash
# 1) Baseline (eğitimsiz) ölçümü
python evaluate.py --n 100 --out results/baseline

# 2) LoRA ince ayarı (60 güncelleme ~ 27 dk, T4). Kayıt varsa kaldığı yerden devam eder.
python train_lora.py --updates 60

# 3) İnce ayarlı modelin ölçümü
python evaluate.py --adapter isu_lora_adapter --n 260 --out results/lora
```

Değerlendirme çıktısında her özellik için doğruluk, çoğunluk sınıfı doğruluğu ve modelin YES deme oranı verilir. Çoğunluk sınıfı karşılaştırması, doğruluğun gerçek bir beceriyi mi yoksa sınıf dengesizliğini mi yansıttığını görmek için gereklidir.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `config.py` | Model, yollar, 11 özellik, veri bölme ve LoRA ayarları |
| `setup_data.py` | Veri setini Hugging Face'ten indirir |
| `common.py` | Veri bölme, model yükleme, çıkarım ve değerlendirme |
| `evaluate.py` | Baseline veya ince ayarlı model değerlendirmesi |
| `train_lora.py` | LoRA eğitimi (sadece cevap kısmında kayıp hesaplanır) |

## Yöntem özeti

- Model her görüntü için 11 özelliğin YES/NO cevabını tek bir JSON olarak üretir. İstem ve cevap ayrıştırma, yarışmanın değerlendirme kodundan alınmıştır.
- Ana model dondurulur, dil katmanlarındaki dikkat ve ileri besleme projeksiyonlarına LoRA adaptörleri eklenir (rank 16, alfa 32, eğitilen parametre yaklaşık %0,8).
- Bellek için float16, gradient checkpointing ve 8 adımlık gradient accumulation kullanılır.

## Sınırlılıklar

- Çıkarım görüntü başına yaklaşık 6 saniye sürer, gerçek zamanlı değildir.
- Eğitim sentetik görüntülerle yapılır, gerçek görüntülere genelleme henüz ölçülmemiştir.
- İki sınıflı etiketlerde doğruluk tek başına yanıltıcı olabilir, bu yüzden çoğunluk sınıfı ile birlikte raporlanır.

## Veri ve lisans

Veri seti ve starter kit ISU-Challenge'a aittir, kendi lisans ve atıf koşullarına tabidir. Kullanmadan önce yarışma sitesindeki atıf bilgisini kaynakçanıza ekleyin. Starter kit kodu bu depoya kopyalanmamıştır. Eğitilmiş adaptör (yaklaşık 120 MB) depoya konmamıştır, eğitim bittikten sonra Hugging Face Hub gibi bir yere yüklenip buraya bağlantı verilebilir.

## Kaynaklar

- ISU-Challenge: <https://isu-challenge.github.io/>
- Veri seti: <https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset>
- Qwen2.5-VL Technical Report (Bai ve ark., 2025)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu ve ark., 2021)
