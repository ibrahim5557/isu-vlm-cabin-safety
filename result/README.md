# Görüntü-Dil Modeli ile Araç İçi Güvenlik Durumu Tespiti

Tek bir araç içi RGB görüntüden emniyet kemeri, telefon kullanımı, koltuk doluluğu ve bebek durumunu sınıflandıran bir sistem. Tasarım dersi projesi olarak [ISU-Challenge](https://isu-challenge.github.io/) (Track B) veri seti ve değerlendirme koduyla geliştirilmiştir. Temel model: `Qwen/Qwen2.5-VL-3B-Instruct`, uyarlama yöntemi: LoRA.

## Proje durumu

- Baseline (eğitimsiz) ölçüm: **tamamlandı** (260 test görüntüsü).
- LoRA ince ayarı: **60, 140 ve 300 güncellemelik üç sürüm eğitildi ve ölçüldü.**
- Gerçek fotoğraflarda test: **yapılmadı.** Gerçek fotoğraf dosyalarının etiketlerle eşleşmesi önce doğrulanmalıdır.
- Arka koltuk kemerleri (özellikle arka sağ) hâlâ zayıftır, bkz. Sonuçlar.

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

İndirilen veride 1030 sentetik görüntü ve etiket çifti ile 30 gerçek fotoğraf bulunmuştur (yarışmanın ilk duyurusundaki sayılardan farklı olabilir). Sentetik görüntüler sabit tohumla (`SEED = 42`) karıştırılıp **770 eğitim / 260 test** olarak ayrılır. Test görüntüleri eğitimde kullanılmaz.

## Sonuçlar

260 test görüntüsünde özellik bazlı doğruluk. Tüm özellikler iki sınıflıdır, rastgele tahmin yaklaşık %50 verir. Koltuk boşsa o koltuğun kemeri etiketlenmediği için kemer ve bebek özelliklerinde örnek sayısı 260'tan azdır (bebek 118, yolcu kemeri 58, arka sol kemer 129, arka sağ kemer 121). Güncelleme sayıları toplam hedeftir, bir güncelleme 8 görüntüdür.

| Özellik | Baseline | 60 güncelleme | 140 güncelleme | 300 güncelleme |
|---|---|---|---|---|
| `baby` | 0,70 | 0,97 | 1,00 | 1,00 |
| `baby_seat` | 1,00 | 1,00 | 1,00 | 1,00 |
| `driver_phone` | 0,80 | 1,00 | 1,00 | 1,00 |
| `phone_codriver_seat` | 0,72 | 0,96 | 1,00 | 1,00 |
| `passenger_codriver` | 0,61 | 0,91 | 1,00 | 1,00 |
| `passenger_rear_seat_left` | 0,66 | 0,72 | 1,00 | 1,00 |
| `passenger_rear_seat_right` | 0,59 | 0,63 | 0,99 | 1,00 |
| `codriver_safety_belt` | 0,59 | 0,52 | 0,98 | 1,00 |
| `driver_safety_belt` | 0,68 | 0,70 | 0,83 | 0,90 |
| `passenger_rear_left_safety_belt` | 0,50 | 0,50 | 0,63 | 0,76 |
| `passenger_rear_right_safety_belt` | 0,40 | 0,13 | 0,44 | 0,51 |
| **Sahne düzeyinde tam eşleşme** | 0,069 | 0,227 | 0,488 | 0,600 |

Ayrıntılı tablolar ve tahminler `results/` klasöründeki dosyalardadır. JSON ayrıştırma hatası tüm ölçümlerde 0, çıkarım süresi ince ayarlı modelde yaklaşık 8 sn/görüntüdür.

Başlıca bulgular:

- İnce ayar, tam eşleşme oranını 0,069'dan 0,600'e çıkarmıştır.
- 60 güncellemelik modelde kemer sorularının çoğu cevapsız kalmıştır, çünkü model arka koltuktaki kişiyi boş sanmıştır (arka sağda 121 dolu koltuğun 88'inde). 140 güncellemede bu sorun neredeyse ortadan kalkmıştır.
- Arka sağ kemerde doğruluk (0,51) çoğunluk sınıfı doğruluğundan (0,53) istatistiksel olarak ayırt edilemez.

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

Colab'da `torchao` kütüphanesinin eski sürümü `peft` ile çakışabilir. `ImportError: ... torchao` hatası alırsanız `pip uninstall -y torchao` çalıştırın. Hugging Face sunucusu indirme sırasında 503 verirse `setup_data.py` komutunu tekrar çalıştırmak kaldığı yerden devam eder.

## Kullanım

```bash
# 1) Baseline (eğitimsiz) ölçümü
python evaluate.py --n 260 --out results/baseline

# 2) LoRA ince ayarı. Sayı TOPLAM güncelleme hedefidir, adaptör klasöründe kayıt varsa oradan devam eder.
python train_lora.py --updates 300

# 3) İnce ayarlı modelin ölçümü
python evaluate.py --adapter isu_lora_adapter --n 260 --out results/lora_300
```

E�itim adaptörün üzerine yazar. Yeni bir eğitimden önce adaptör klasörünü kopyalayıp yedekleyin.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `config.py` | Model, yollar, 11 özellik, veri bölme ve LoRA ayarları |
| `setup_data.py` | Veri setini Hugging Face'ten indirir |
| `common.py` | Veri bölme, model yükleme, çıkarım ve değerlendirme |
| `evaluate.py` | Baseline veya ince ayarlı model değerlendirmesi |
| `train_lora.py` | LoRA eğitimi (sadece cevap kısmında kayıp hesaplanır) |
| `results/` | Ölçüm çıktıları (csv ve json) |

## Yöntem özeti

- Model her görüntü için 11 özelliğin YES/NO cevabını tek bir JSON olarak üretir. İstem ve cevap ayrıştırma, yarışmanın değerlendirme kodundan alınmıştır.
- Ana model dondurulur, dil katmanlarındaki dikkat ve ileri besleme projeksiyonlarına LoRA adaptörleri eklenir (rank 16, alfa 32, eğitilen parametre yaklaşık %0,8).
- Bellek için float16, gradient checkpointing, 8 adımlık gradient accumulation ve en fazla 448×28×28 piksel görüntü boyutu kullanılır.

## Sınırlılıklar

- Çıkarım görüntü başına yaklaşık 8 saniye sürer, gerçek zamanlı değildir.
- Eğitim sentetik görüntülerle yapılmıştır, gerçek görüntülere genelleme henüz ölçülmemiştir.
- Arka kemer özelliklerinde yaklaşık 120 örnek vardır, bu büyüklükte %95 güven aralığı yaklaşık ±9 puandır.
- Eğitim süresine (60, 140, 300 güncelleme) test sonuçlarına bakılarak karar verilmiştir, ayrı bir doğrulama kümesi kullanılmamıştır. Test sonuçları bu nedenle hafifçe iyimser olabilir.
- Her sürüm bir kez eğitilmiştir, farklı tohumlarla tekrarlanmamıştır.

## Veri ve lisans

Veri seti ve starter kit ISU-Challenge'a aittir, kendi lisans ve atıf koşullarına tabidir. Kullanmadan önce yarışma sitesindeki atıf bilgisini kaynakçanıza ekleyin. Starter kit kodu bu depoya kopyalanmamıştır. Eğitilmiş adaptör (yaklaşık 120 MB) depoya konmamıştır.

## Kaynaklar

- ISU-Challenge: <https://isu-challenge.github.io/>
- Veri seti: <https://huggingface.co/datasets/ISU-Test/isu-challenge-dataset>
- Qwen2.5-VL Technical Report (Bai ve ark., 2025)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu ve ark., 2021)
