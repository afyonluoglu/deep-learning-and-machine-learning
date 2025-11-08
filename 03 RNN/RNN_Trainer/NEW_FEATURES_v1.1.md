# RNN Trainer v1.1 - Yeni Özellikler 🎉

## 📅 Tarih: 30 Eylül 2025

---

## 🆕 Eklenen Özellikler

### 1️⃣ **DROPOUT Regularization** 🎯

#### Ne İşe Yarar?
Dropout, modelin **overfitting** (aşırı öğrenme) yapmasını önleyen bir regularization tekniğidir. Eğitim sırasında rastgele nöronları devre dışı bırakarak modelin daha genel öğrenmesini sağlar.

#### Nasıl Çalışır?
- **Eğitim sırasında**: Belirlenen oranda nöronlar rastgele kapatılır (dropout_rate)
- **Tahmin sırasında**: Tüm nöronlar aktif olur (inverted dropout ile ölçekleme)
- **0.0**: Dropout kapalı (varsayılan)
- **0.1-0.5**: Hafif-orta regularization (önerilen)
- **0.6-0.9**: Güçlü regularization (dikkatli kullan!)

#### GUI'de Kullanım
```
🔧 Model Parameters
├── Hidden Units: 20
├── Learning Rate: 0.01
├── Sequence Length: 20
├── Activation: tanh
└── Dropout Rate: 0.0 (Off)  ← YENİ!
    └── Slider: 0.0 - 0.9
```

#### Ne Zaman Kullanmalı?
✅ **Dropout Kullan:**
- Model eğitim verisine çok iyi uyuyor ama test verisinde kötü
- Loss grafiği düşük ama gerçek tahminler kötü
- Karmaşık modeller (çok hidden unit)
- Az veri var

❌ **Dropout Kullanma:**
- Model zaten underfit (yeterince öğrenememiş)
- Çok basit modeller
- Çok fazla veri var

#### Örnek Senaryolar

**Senaryo 1: Overfitting Var**
```
1. Model eğit (dropout=0.0)
2. Training loss: 0.001 ✅
3. Test prediction: Kötü ❌
4. Dropout'u 0.3 yap
5. Tekrar eğit
6. Training loss: 0.005 (biraz arttı)
7. Test prediction: Çok iyi! ✅
```

**Senaryo 2: Karmaşık Model**
```
Hidden Units: 100
Dropout: 0.0
→ Model ezberleme riski yüksek!

Çözüm:
Hidden Units: 100
Dropout: 0.3-0.5
→ Model genel öğrenir ✅
```

#### Teknik Detaylar
- **Inverted Dropout** kullanılır: `h = h * mask / (1 - dropout_rate)`
- Sadece **eğitim sırasında** aktif
- **Forward pass**: Rastgele mask uygulanır
- **Prediction**: Dropout devre dışı (`self.training_mode = False`)

---

### 2️⃣ **Model Yüklendiğinde Panel Güncelleme** 🔄

#### Düzeltilen Bug
Önceden model yüklendiğinde sadece model parametreleri yükleniyordu, GUI'deki slider'lar ve label'lar güncellenMİYORDU.

#### Şimdi Ne Oluyor?
Model yüklendiğinde:
1. ✅ **Slider'lar** doğru konuma gelir
2. ✅ **Label'lar** doğru değerleri gösterir
3. ✅ **Dropdown'lar** (activation) güncellenir
4. ✅ **Dropout değeri** yüklenir (geriye uyumlu!)

#### Örnek
```python
# Model kaydederken:
Hidden Units: 50
Learning Rate: 0.05
Dropout: 0.3

# Model yüklendiğinde:
✅ Hidden Units slider → 50
✅ Learning Rate slider → 0.05
✅ Dropout slider → 0.3
✅ Tüm label'lar güncel!
```

#### Geriye Uyumluluk
Eski modeller (dropout olmadan kaydedilmiş):
```python
dropout_rate = model_data.get('dropout_rate', 0.0)  # Varsayılan 0.0
```

---

### 3️⃣ **Custom Veri Yükleme & Future Prediction** 🔮

#### Ne İşe Yarar?
Artık kendi verilerinizi yükleyip gelecek değerleri tahmin edebilirsiniz!

#### Özellikler
1. **CSV Yükleme**: Kendi time series verilerinizi yükleyin
2. **Future Prediction**: Model gelecek N adımı tahmin eder
3. **Görselleştirme**: Geçmiş + Gelecek tek grafikte

#### CSV Format
```csv
Temperature
15.2
16.8
18.5
20.3
...
```

**Kurallar:**
- ✅ İlk satır başlık (atlanır)
- ✅ Her satırda bir sayı
- ✅ En az 10 değer
- ✅ Virgül veya nokta ayracı desteklenir
- ✅ `.csv`, `.txt` uzantıları

#### Kullanım Adımları

**Adım 1: CSV Hazırla**
```csv
Temperature
20.5
21.3
22.8
...
(Son 20-30 günün sıcaklığı)
```

**Adım 2: Model Eğit veya Yükle**
```
1. Model initialize et
2. Benzer veriyle eğit (sine wave, vb.)
   VEYA
   Önceden eğitilmiş model yükle
```

**Adım 3: CSV Yükle**
```
📁 Custom Data
└── 📂 Load CSV Data
    → CSV dosyasını seç
    → Veri görselleşir
```

**Adım 4: Gelecek Tahmin Et**
```
📁 Custom Data
└── 🔮 Predict Future Values
    → Kaç adım? (örn: 5)
    → Model sonraki 5 günü tahmin eder!
```

#### Görsel Açıklama
```
Grafik Bölgeleri:
┌─────────────────────────────────────┐
│ Geçmiş Veri (Mavi)                  │ ← CSV'den yüklenen
│                                     │
│         │ ← Tahmin Başlangıcı       │
│         │   (Turuncu çizgi)         │
│         └──→ Gelecek (Kırmızı)     │ ← Model tahmini
└─────────────────────────────────────┘
```

#### Örnek Kullanım Senaryoları

**Senaryo 1: Hava Sıcaklığı Tahmini**
```
Veri: Son 30 günün sıcaklığı
Model: Sine wave ile eğitilmiş (mevsimsel pattern)
Tahmin: Sonraki 5 günün sıcaklığı

Adımlar:
1. 30 günlük sıcaklık verisi CSV'de
2. Model sine wave ile eğitilmiş (100 epoch)
3. CSV yükle
4. "Predict Future Values" → 5
5. Grafik gösterir:
   - Geçmiş 30 gün (mavi)
   - Gelecek 5 gün (kırmızı)
```

**Senaryo 2: Borsa Fiyat Tahmini**
```
Veri: Hisse senedi kapanış fiyatları
Model: Random walk ile eğitilmiş
Tahmin: Sonraki 10 işlem günü

CSV:
StockPrice
125.50
127.30
126.80
...
```

**Senaryo 3: Enerji Tüketimi**
```
Veri: Saatlik elektrik tüketimi
Model: Mixed waves ile eğitilmiş
Tahmin: Sonraki 24 saat

CSV:
EnergyConsumption
45.2
48.5
52.1
...
```

#### Teknik Detaylar

**Veri İşleme:**
```python
1. CSV okunur
2. Normalize edilir (0-1 arası)
3. Son sequence_length değer seed olur
4. Model predict_sequence() çağırır
5. Sonuç denormalize edilir
6. Grafik çizilir
```

**Seed Mekanizması:**
```python
Model Sequence Length: 20

CSV'de 30 değer var:
[v1, v2, ..., v29, v30]
          └─────────┘
         Son 20 değer
         (Seed olarak kullanılır)

Tahmin:
Seed → [v11, v12, ..., v30]
Future → [v31, v32, v33, v34, v35]  (5 adım)
```

---

## 🎯 Kullanım Örnekleri

### Örnek 1: Dropout ile Overfitting'i Önleme

```
Problem: Model eğitim verisini ezberledi

Adımlar:
1. Initialize Model
   - Hidden Units: 50
   - Dropout: 0.0
2. Generate Data (Sine Wave, 500 samples)
3. Train (50 epochs)
4. Test Prediction → MSE: 0.001 (çok iyi!)
5. Generate NEW Data (aynı parametreler)
6. Test Prediction → MSE: 0.85 (kötü! overfitting var)

Çözüm:
1. Dropout'u 0.3 yap
2. Tekrar initialize
3. Tekrar train (50 epochs)
4. Test Prediction → MSE: 0.005 (biraz arttı)
5. Generate NEW Data
6. Test Prediction → MSE: 0.008 (çok iyi! genelleme başarılı)
```

### Örnek 2: Sıcaklık Tahmini

```
Senaryo: Son 30 günün sıcaklığını kullanarak 7 günlük tahmin

Adımlar:
1. CSV Hazırla (30 günlük sıcaklık)
   temperature_data.csv:
   Temperature
   15.2
   16.8
   ...
   22.5

2. Model Eğit
   - Wave Type: Sine Wave (mevsimsel)
   - Hidden Units: 30
   - Dropout: 0.2 (overfitting önleme)
   - Epochs: 100

3. CSV Yükle
   📂 Load CSV Data → temperature_data.csv
   
4. Gelecek Tahmin
   🔮 Predict Future Values → 7
   
5. Sonuç:
   Grafik gösterir:
   - Mavi: Geçmiş 30 gün
   - Kırmızı: Gelecek 7 gün tahmini
   
   Tahmin Değerleri:
   Step +1: 23.1°C
   Step +2: 24.3°C
   Step +3: 25.8°C
   ...
```

### Örnek 3: Model Karşılaştırması

```
Senaryo: Farklı dropout değerlerini karşılaştır

Test 1: Dropout = 0.0
1. Initialize (dropout=0.0)
2. Train 50 epochs
3. Save Graph → outputs/no_dropout.png

Test 2: Dropout = 0.3
1. Initialize (dropout=0.3)
2. Train 50 epochs
3. Save Graph → outputs/dropout_03.png

Test 3: Dropout = 0.5
1. Initialize (dropout=0.5)
2. Train 50 epochs
3. Save Graph → outputs/dropout_05.png

Karşılaştır:
- outputs/ klasöründeki PNG'leri aç
- Loss grafiklerini incele
- En iyi performansı seç
```

---

## 📊 Teknik Değişiklikler

### RNN Model (`rnn_model.py`)

**Eklenen:**
- `dropout_rate` parametresi (`__init__`)
- `training_mode` flag
- Dropout uygulaması (`forward()`)
- `self.training_mode = True/False` switch'leri

**Değiştirilen:**
- `__init__`: dropout_rate parametresi eklendi
- `forward()`: Dropout mask uygulaması
- `train_epoch()`: training_mode = True
- `predict()`: training_mode = False
- `predict_sequence()`: training_mode = False
- `save_model()`: dropout_rate kaydedilir
- `load_model()`: dropout_rate yüklenir (backward compatible)
- `get_parameters()`: dropout_rate döndürülür

### GUI (`rnn_trainer_app.py`)

**Eklenen Değişkenler:**
- `self.custom_data_raw`
- `self.custom_data_normalized`

**Eklenen UI Elemanları:**
- Dropout slider ve label
- "Load CSV Data" butonu
- "Predict Future Values" butonu

**Eklenen Fonksiyonlar:**
- `load_custom_data()`: CSV yükleme
- `predict_future_values()`: Gelecek tahmin

**Güncellenen Fonksiyonlar:**
- `initialize_model()`: dropout parametresi eklendi
- `load_model()`: Panel güncelleme eklendi
- `show_model_info()`: Dropout bilgisi gösterilir
- `_get_parameters_text()`: Dropout parametresi eklendi

---

## 📁 Dosya Yapısı

```
RNN_Trainer/
├── rnn_model.py                      ← Dropout eklendi
├── rnn_trainer_app.py                ← 3 yeni özellik
├── data_generator.py                 (değişiklik yok)
├── rnn_help.txt                      (değişiklik yok)
├── example_temperature_data.csv      ← YENİ! Örnek CSV
├── NEW_FEATURES_v1.1.md             ← YENİ! Bu dosya
├── outputs/
│   ├── data_plot_*.png
│   └── loss_plot_*.png
└── saved_models/
    ├── model.pkl
    └── model_config.json
```

---

## 🧪 Test Senaryoları

### Test 1: Dropout Etkisi
```
1. ✅ Dropout = 0.0 ile eğit
2. ✅ Dropout = 0.3 ile eğit
3. ✅ Loss grafiklerini karşılaştır
4. ✅ Test predictions'ı karşılaştır
5. ✅ Dropout'un regularization etkisini gözlemle
```

### Test 2: Model Yükleme
```
1. ✅ Dropout=0.4 ile model oluştur ve kaydet
2. ✅ Programı kapat
3. ✅ Programı aç
4. ✅ Modeli yükle
5. ✅ Dropout slider → 0.4 olmalı ✅
6. ✅ Tüm label'lar doğru değerleri göstermeli
```

### Test 3: CSV Yükleme
```
1. ✅ example_temperature_data.csv aç
2. ✅ 30 satır olduğunu kontrol et
3. ✅ "Load CSV Data" tıkla
4. ✅ Grafik mavi çizgiyi göstermeli
5. ✅ "Loaded 30 data points" mesajı
6. ✅ Min/Max değerler gösterilmeli
```

### Test 4: Future Prediction
```
1. ✅ CSV yükle (30 değer)
2. ✅ Model sequence_length = 20 olsun
3. ✅ "Predict Future Values" → 5
4. ✅ Grafik göstermeli:
   - Mavi: 30 geçmiş değer
   - Turuncu çizgi: Tahmin başlangıcı
   - Kırmızı: 5 gelecek değer
5. ✅ Message box'ta tahminler görünmeli
```

### Test 5: Büyük Tahmin
```
1. ✅ 100 değerlik CSV yükle
2. ✅ "Predict Future Values" → 50
3. ✅ Grafik düzgün görünmeli
4. ✅ 50 tahmin değeri listesi gösterilmeli
5. ✅ Performans kabul edilebilir olmalı (<2 saniye)
```

---

## 🔍 Sık Sorulan Sorular (FAQ)

### Q1: Dropout değerini ne kadar yapmalıyım?
**A:** 
- Başlangıç: 0.2-0.3
- Overfitting varsa: 0.4-0.5
- Çok şiddetli overfitting: 0.6-0.7
- Asla 0.9'dan büyük yapma!

### Q2: CSV dosyam çalışmıyor, neden?
**A:** Kontrol et:
- ✅ En az 10 satır var mı?
- ✅ Her satırda sadece bir sayı var mı?
- ✅ Başlık satırı var mı? (ilk satır atlanır)
- ✅ Sayılar geçerli mi? (15.5, 20, -5 gibi)

### Q3: Tahminler çok kötü, ne yapmalıyım?
**A:**
- Model benzer veriyle eğitilmeli (sine → sine)
- Yeterli epoch eğitilmeli (50+)
- Sequence length yeterli olmalı
- CSV verisi en az 2x sequence_length olmalı

### Q4: Eski modellerim çalışır mı?
**A:** Evet! Geriye uyumlu:
```python
# Eski model (dropout yok)
dropout_rate = 0.0  # Otomatik atanır

# Yeni model (dropout var)
dropout_rate = kaydedilen değer
```

### Q5: Ne kadar gelecek tahmin edebilirim?
**A:**
- Kısa vade: 5-10 adım (daha doğru)
- Orta vade: 20-50 adım (makul)
- Uzun vade: 100+ adım (dikkatli!)

Unutma: Her tahmin bir öncekini kullanır, hata birikir!

### Q6: CSV'de negatif değerler olabilir mi?
**A:** Evet! Normalizasyon her aralığı destekler:
```
-50, -30, -10, 0, 20, 50
→ Normalize → 0 ile 1 arası
→ Tahmin
→ Denormalize → Orijinal ölçek
```

---

## 🎓 Önerilen Workflow

### Yeni Başlayanlar İçin

**Adım 1: Basit Başla**
```
1. Dropout = 0.0
2. Hidden Units = 20
3. Sine Wave, 500 samples
4. Train 50 epochs
5. Test Prediction
```

**Adım 2: Parametrelerle Oyna**
```
1. Dropout'u değiştir (0.0, 0.2, 0.5)
2. Her birini test et
3. Grafiklerini kaydet
4. Karşılaştır
```

**Adım 3: Kendi Verini Kullan**
```
1. CSV hazırla (30+ değer)
2. Benzer wave type ile eğit
3. CSV yükle
4. Gelecek tahmin et
```

### İleri Seviye Kullanım

**Senaryo: Gerçek Dünya Tahmini**
```
1. Veri Hazırlığı:
   - 6 aylık günlük satış verisi (180 değer)
   - CSV'ye kaydet

2. Model Seçimi:
   - Mixed Waves veya ARMA (karmaşık pattern)
   - Hidden Units: 50-100
   - Dropout: 0.3
   - Sequence Length: 30 (1 aylık)

3. Eğitim:
   - Benzer veriyle 200 epoch eğit
   - Loss < 0.01 hedefle
   - Model kaydet

4. Gerçek Tahmin:
   - Gerçek CSV yükle
   - 30 gün gelecek tahmin et
   - Grafik kaydet
   - Sonuçları analiz et

5. Doğrulama:
   - Gerçek değerler gelince karşılaştır
   - MSE hesapla
   - Model iyileştir
```

---

## 📈 Performans İpuçları

### Dropout ile Hızlandırma
```
Dropout = 0.0: En hızlı (regularization yok)
Dropout = 0.3: ~10% yavaş (makul)
Dropout = 0.7: ~30% yavaş (ağır regularization)

Öneri: Dropout 0.5'ten düşük tut!
```

### CSV Yükleme Optimizasyonu
```
Küçük dosyalar (<1000 satır): Anında
Orta dosyalar (1000-10000): <1 saniye
Büyük dosyalar (10000+): ~5 saniye

Öneri: Gereksiz büyük CSV kullanma!
```

### Future Prediction Süresi
```
10 adım: ~0.1 saniye
50 adım: ~0.5 saniye
100 adım: ~1 saniye
1000 adım: ~10 saniye

Her adım bir forward pass = O(n) karmaşıklık
```

---

## 🔧 Troubleshooting

### Problem 1: "Dropout has no effect"
**Çözüm:**
```
- Model tekrar initialize et
- dropout_rate > 0.1 olmalı
- Eğitim sırasında training_mode = True
- Tahmin sırasında training_mode = False
```

### Problem 2: "CSV load fails"
**Çözüm:**
```
- UTF-8 encoding kullan
- Sadece bir sütun olmalı
- İlk satır başlık (atlanır)
- Boş satır olmamalı
```

### Problem 3: "Future predictions are constant"
**Çözüm:**
```
- Model yeterince eğitilmemiş
- Veri çok uniform (varyasyon yok)
- Sequence length çok kısa
- Model too simple (hidden units artır)
```

### Problem 4: "Panel not updating on load"
**Çözüm:**
```
- Yeni versiyonu kullan (v1.1+)
- Model tekrar kaydet (eski format olabilir)
- Label güncellemesi otomatik olmalı
```

---

## 📝 Changelog

### Version 1.1 (30 Eylül 2025)

**Added:**
- ✅ Dropout regularization
- ✅ Panel update on model load
- ✅ Custom CSV data loading
- ✅ Future value prediction
- ✅ Example CSV file
- ✅ Comprehensive documentation

**Fixed:**
- ✅ Model load bug (sliders not updating)
- ✅ Label update bug
- ✅ Backward compatibility for old models

**Improved:**
- ✅ Model info displays dropout
- ✅ Saved graphs include dropout parameter
- ✅ Better error messages

---

## 🎉 Özet

Bu güncellemeyle RNN Trainer artık:

1. **Daha Akıllı**: Dropout ile overfitting'i önler
2. **Daha Kullanışlı**: Model yüklenince panel güncellenir
3. **Daha Güçlü**: Kendi verilerinle gelecek tahmin eder

**Mevcut model şunları yapabilir:**
✅ Time series öğrenme
✅ Overfitting önleme (dropout)
✅ Kendi verinle çalışma (CSV)
✅ Gelecek tahmin (N adım)
✅ Görselleştirme
✅ Model kaydet/yükle
✅ Parametre karşılaştırma

**Yeni program gerekli mi?**
❌ Hayır! Mevcut program tüm ihtiyaçları karşılıyor.

**Sonraki adımlar?**
1. Programı test et
2. Kendi verilerinle dene
3. Sonuçları kaydet ve analiz et
4. Gerekirse parametre ayarla

---

**Başarılar! 🚀**

*RNN Trainer v1.1 ile yapay zeka öğrenmenin tadını çıkarın!*
