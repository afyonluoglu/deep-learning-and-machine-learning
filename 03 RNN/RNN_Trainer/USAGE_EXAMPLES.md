# RNN Trainer - Detaylı Kullanım Örnekleri

Bu dosya, RNN Trainer uygulamasının çeşitli kullanım senaryolarını adım adım örneklerle açıklamaktadır.

## 📋 İçindekiler

1. [Başlangıç Seviyesi Örnekler](#başlangıç-seviyesi-örnekler)
2. [Orta Seviye Örnekler](#orta-seviye-örnekler)
3. [İleri Seviye Örnekler](#ileri-seviye-örnekler)
4. [Parametre Optimizasyon Örnekleri](#parametre-optimizasyon-örnekleri)
5. [Hata Ayıklama Senaryoları](#hata-ayıklama-senaryoları)

---

## Başlangıç Seviyesi Örnekler

### Örnek 1: İlk RNN Modeliniz - Basit Sinüs Dalgası

**Hedef**: RNN'in basit bir periyodik örüntüyü nasıl öğrendiğini anlamak.

**Adım Adım**:

1. **Uygulamayı Başlatın**
   ```bash
   python rnn_trainer_app.py
   ```

2. **Model Parametrelerini Ayarlayın** (Sol panel - Model Parameters bölümü)
   - Hidden Units: `20` (slider'ı 20'ye getirin)
   - Learning Rate: `0.01` (slider'ı 0.01'e getirin)
   - Sequence Length: `20` (slider'ı 20'ye getirin)
   - Activation Function: `tanh` (dropdown'dan seçin)

3. **"Initialize Model" Butonuna Tıklayın**
   - Popup mesaj: "Model initialized successfully!" göreceksiniz
   - Status bar: Model bilgilerini gösterecek

4. **Veri Üretin** (Data Generation bölümü)
   - Wave Type: `Sine Wave` seçin
   - Samples: `500`
   - Frequency: `1.0`
   - Noise Level: `0.05`

5. **"Generate Data" Butonuna Tıklayın**
   - Sağ üst grafikte mavi sinüs dalgası görünecek
   - Status bar: "Generated Sine Wave: 500 samples..." gösterecek

6. **Eğitim Başlatın** (Training bölümü)
   - Epochs: `100`
   - **"Start Training"** butonuna tıklayın

7. **Eğitimi İzleyin**
   - Alt grafikte loss değeri düşecek (logaritmik ölçekte)
   - Status bar her 5 epoch'ta güncellenecek
   - Beklenen süre: ~30 saniye

8. **Test Edin**
   - Eğitim bitince **"Test Prediction"** tıklayın
   - Üst grafikte:
     - Mavi çizgi: Gerçek veri
     - Kırmızı kesikli çizgi: Model tahmini
   - Popup'ta MSE değerini görün (beklenen: < 0.02)

**Beklenen Sonuç**:
- Loss: ~0.001 seviyesine düşmeli
- Tahminler gerçek veriyi çok yakın takip etmeli
- MSE < 0.02 olmalı

**Öğrenilenler**:
- ✅ RNN periyodik örüntüleri öğrenebilir
- ✅ Loss değeri düzenli azalmalı
- ✅ Daha fazla epoch daha iyi sonuç verir

---

### Örnek 2: Farklı Dalga Tiplerini Keşfetme

**Hedef**: Farklı dalga tiplerinin RNN öğrenimi üzerindeki etkisini görmek.

**Test 1: Kare Dalga**
```
Model: Aynı (Hidden=20, LR=0.01, SeqLen=20)
Data:
  - Wave Type: Square Wave
  - Samples: 500
  - Frequency: 0.5  (daha yavaş)
  - Noise: 0.02     (daha az gürültü, keskin geçişler için)
Training: 150 epochs
```

**Gözlem**:
- Keskin geçişlerde modelin zorlandığını göreceksiniz
- Loss daha yavaş azalır
- Tahminler yuvarlatılmış olacak (RNN'in doğası)

**Test 2: Üçgen Dalga**
```
Model: Aynı
Data:
  - Wave Type: Triangular Wave
  - Samples: 500
  - Frequency: 0.8
  - Noise: 0.05
Training: 100 epochs
```

**Gözlem**:
- Doğrusal rampalar iyi öğrenilir
- Tepe noktalarında küçük yuvarlatma
- Sinüsten daha kolay öğrenilir

**Test 3: Karışık Dalgalar**
```
Model: Hidden=40 (daha fazla kapasite gerekli)
Data:
  - Wave Type: Mixed Waves
  - Samples: 1000
  - Frequency: 1.5
  - Noise: 0.1
Training: 200 epochs
```

**Gözlem**:
- Daha karmaşık, loss daha yavaş azalır
- Birden fazla frekansı ayrıştırmaya çalışır
- MSE biraz daha yüksek (~0.05-0.1)

**Karşılaştırma Tablosu**:
| Dalga Tipi | Epochs | Final Loss | MSE | Zorluk |
|------------|--------|------------|-----|--------|
| Sine       | 100    | ~0.001     | <0.02 | Kolay |
| Square     | 150    | ~0.005     | <0.05 | Orta |
| Triangle   | 100    | ~0.002     | <0.03 | Kolay |
| Mixed      | 200    | ~0.008     | <0.10 | Zor |

---

### Örnek 3: Gürültü Etkisini Anlama

**Hedef**: Gürültünün model performansına etkisini görmek.

**Deney Serisi**: Aynı model, aynı veri, sadece gürültü değişiyor

**Model Parametreleri** (sabit):
```
Hidden Units: 30
Learning Rate: 0.01
Sequence Length: 20
Activation: tanh
```

**Veri** (sabit):
```
Wave Type: Sine Wave
Samples: 500
Frequency: 1.0
```

**Deney 1: Gürültüsüz** (Noise = 0.0)
```
Training: 100 epochs
Beklenen MSE: < 0.005
Gözlem: Mükemmel fit, model tamamen ezberledi
```

**Deney 2: Hafif Gürültü** (Noise = 0.05)
```
Training: 100 epochs
Beklenen MSE: ~0.02
Gözlem: Çok iyi, ana örüntü korundu
```

**Deney 3: Orta Gürültü** (Noise = 0.15)
```
Training: 150 epochs
Beklenen MSE: ~0.08
Gözlem: İyi, model gürültüyü filtreledi
```

**Deney 4: Yüksek Gürültü** (Noise = 0.3)
```
Training: 200 epochs
Beklenen MSE: ~0.15
Gözlem: Zorlanıyor ama ana trendi yakalıyor
```

**Önemli Notlar**:
- 🔍 Gürültü generalizasyonu artırır
- ⚠️ Çok fazla gürültü öğrenmeyi zorlaştırır
- 💡 Optimal: 0.05-0.1 arası

---

## Orta Seviye Örnekler

### Örnek 4: Learning Rate Optimizasyonu

**Hedef**: Optimal learning rate'i bulmak.

**Sabit Parametreler**:
```
Hidden Units: 25
Sequence Length: 20
Activation: tanh
Data: Sine Wave (500 samples, freq=1.0, noise=0.05)
Epochs: 100
```

**Test Serisi**:

**Test A: Çok Düşük LR** (0.001)
```
Learning Rate: 0.001
Sonuç:
  - Loss çok yavaş azalır
  - 100 epoch yetmez
  - Final Loss: ~0.05
  - Durum: Underfitting
Çözüm: LR artır
```

**Test B: Düşük LR** (0.005)
```
Learning Rate: 0.005
Sonuç:
  - Loss düzenli azalır
  - Stabil eğitim
  - Final Loss: ~0.01
  - Durum: İyi ama yavaş
```

**Test C: Optimal LR** (0.01)
```
Learning Rate: 0.01
Sonuç:
  - Loss hızla azalır
  - Stabil ve hızlı
  - Final Loss: ~0.002
  - Durum: OPTIMAL ✓
```

**Test D: Yüksek LR** (0.05)
```
Learning Rate: 0.05
Sonuç:
  - Loss salınım yapar
  - Kararsız eğitim
  - Final Loss: ~0.02 (salınımlı)
  - Durum: Çok yüksek
Çözüm: LR azalt
```

**Test E: Çok Yüksek LR** (0.1)
```
Learning Rate: 0.1
Sonuç:
  - Loss diverge olabilir (artar)
  - Çok kararsız
  - Final Loss: Artıyor!
  - Durum: Divergence
Çözüm: Çok azalt
```

**Grafik Analizi**:
```
LR = 0.001: ________   (düz, yavaş)
LR = 0.005: \____      (düzenli düşüş)
LR = 0.01:  \___       (hızlı düşüş) ← OPTIMAL
LR = 0.05:  \/\/\/\    (salınımlı)
LR = 0.1:   /\/\       (diverge)
```

---

### Örnek 5: Hidden Units Kapasitesi

**Hedef**: Model kapasitesinin etkisini anlamak.

**Senaryo**: Karmaşık Mixed Waves öğrenimi

**Veri**:
```
Wave Type: Mixed Waves
Samples: 1000
Frequency: 2.0
Noise: 0.1
```

**Sabit Parametreler**:
```
Learning Rate: 0.01
Sequence Length: 25
Activation: tanh
Epochs: 150
```

**Test A: Küçük Model** (Hidden = 10)
```
Hidden Units: 10
Total Parameters: ~131

Sonuç:
  - Yetersiz kapasite
  - Ana frekansı yakalıyor
  - Detayları kaçırıyor
  - MSE: ~0.15
  - Durum: UNDERFITTING

Gözlem: Model çok basit, karmaşık örüntü için yetersiz
```

**Test B: Orta Model** (Hidden = 30)
```
Hidden Units: 30
Total Parameters: ~991

Sonuç:
  - İyi denge
  - Ana örüntü ve bazı detaylar
  - MSE: ~0.08
  - Durum: İYİ

Gözlem: Dengeli kapasite
```

**Test C: Büyük Model** (Hidden = 60)
```
Hidden Units: 60
Total Parameters: ~3,721

Sonuç:
  - Mükemmel fit
  - Tüm detayları yakalıyor
  - MSE: ~0.04
  - Durum: ÇOK İYİ
  - Uyarı: Eğitim 2x daha yavaş

Gözlem: Yüksek kapasite, en iyi sonuç
```

**Test D: Çok Büyük Model** (Hidden = 100)
```
Hidden Units: 100
Total Parameters: ~10,201

Sonuç:
  - Mükemmel fit (60 ile aynı)
  - MSE: ~0.04 (iyileşme yok!)
  - Durum: OVERPARAMETERIZED
  - Uyarı: Eğitim 4x daha yavaş

Gözlem: Gereksiz kapasite, verimlilik kaybı
```

**Optimal Seçim**: Hidden = 60 (en iyi MSE/hız dengesi)

---

### Örnek 6: Sequence Length Optimizasyonu

**Hedef**: Doğru sequence length'i bulmak.

**Problem**: Damped Oscillation (sönümlü salınım) öğrenimi

**Veri**:
```
Wave Type: Damped Oscillation
Samples: 800
Frequency: 1.0
Damping: 0.1
Noise: 0.05
```

**Model**:
```
Hidden Units: 30
Learning Rate: 0.01
Activation: tanh
Epochs: 150
```

**Analiz**: Damped oscillation'ın periyodu ~20 adım

**Test A: Çok Kısa** (SeqLen = 5)
```
Sequence Length: 5

Sonuç:
  - Yerel örüntüleri yakalar
  - Uzun vadeli trendi kaçırır
  - MSE: ~0.12
  
Gözlem: Pencere çok dar, genel dinamiği görmüyor
```

**Test B: Kısa** (SeqLen = 15)
```
Sequence Length: 15

Sonuç:
  - Kısa vadeli tahminler iyi
  - Sönümleme trendini kısmen yakalar
  - MSE: ~0.08
  
Gözlem: Biraz daha iyi ama hala yetersiz
```

**Test C: Optimal** (SeqLen = 25)
```
Sequence Length: 25

Sonuç:
  - Tam bir periyodu görebiliyor
  - Hem salınımı hem sönümlemeyi öğreniyor
  - MSE: ~0.04
  - Durum: OPTIMAL ✓
  
Gözlem: Periyottan biraz uzun = ideal
```

**Test D: Uzun** (SeqLen = 40)
```
Sequence Length: 40

Sonuç:
  - Çok iyi sonuçlar
  - MSE: ~0.04 (25 ile aynı)
  - Uyarı: Eğitim daha yavaş
  
Gözlem: Ek kazanç yok, gereksiz uzun
```

**Kural**: Sequence Length ≈ 1.2 × Periyot

---

## İleri Seviye Örnekler

### Örnek 7: Model Persistency - Kaydetme ve Yükleme

**Hedef**: Eğitilmiş modeli kaydetme ve yeniden kullanma.

**Senaryo 1: Model Eğitimi ve Kaydetme**

1. **İlk Eğitim**:
   ```
   Model:
     - Hidden Units: 40
     - Learning Rate: 0.01
     - Sequence Length: 25
     - Activation: tanh
   
   Data:
     - Wave Type: ARMA
     - Samples: 1000
     - Noise: 0.1
   
   Training:
     - Epochs: 300
     - Final Loss: ~0.005
     - MSE: ~0.06
   ```

2. **Modeli Kaydedin**:
   - "Save Model" butonuna tıklayın
   - Dosya adı: `arma_model_v1.pkl`
   - Konum: `RNN_Trainer/models/` (klasör oluşturun)
   - Otomatik oluşur: `arma_model_v1_config.json`

3. **Uygulamayı Kapatın ve Yeniden Açın**

**Senaryo 2: Model Yükleme ve Test**

1. **Modeli Yükleyin**:
   - "Load Model" butonuna tıklayın
   - `arma_model_v1.pkl` seçin
   - Parametreler otomatik yüklenir

2. **Yeni Veri Üretin** (aynı tip):
   ```
   Wave Type: ARMA (aynı parametreler)
   Samples: 500
   ```

3. **Test Edin**:
   - "Test Prediction" tıklayın
   - Model direkt çalışır (eğitimsiz!)
   - MSE kontrol edin

**Senaryo 3: Transfer Learning - Devam Eğitimi**

1. **Yüklenen Model ile Devam**:
   - Model zaten yüklü
   - Farklı veri üretin (örn: biraz farklı parametreler)
   - Epochs: 50 (fine-tuning)
   - "Start Training" tıklayın

2. **Gelişmiş Model Olarak Kaydedin**:
   - "Save Model"
   - Dosya adı: `arma_model_v2.pkl`

**Kullanım Senaryoları**:
- ✅ Uzun eğitimleri bölmek
- ✅ Farklı veri setlerinde test
- ✅ Model versiyonlama
- ✅ En iyi modeli koruma
- ✅ Ekip içinde paylaşım

---

### Örnek 8: Çoklu Model Karşılaştırması

**Hedef**: Birden fazla konfigürasyon denemek ve en iyisini seçmek.

**Problem**: Polynomial trend prediction

**Veri** (sabit):
```
Wave Type: Polynomial
Samples: 800
Coefficients: [0, 0.5, 0.1]
Noise: 0.08
```

**Model Varyasyonları**:

**Model A: "Fast"**
```
Hidden Units: 15
Learning Rate: 0.02
Sequence Length: 15
Epochs: 100

Eğit → Test → Kaydet: "poly_model_fast.pkl"

Sonuç:
  - Eğitim süresi: ~20 saniye
  - MSE: 0.12
  - Notlar: Hızlı ama orta doğruluk
```

**Model B: "Balanced"**
```
Hidden Units: 30
Learning Rate: 0.01
Sequence Length: 25
Epochs: 150

Eğit → Test → Kaydet: "poly_model_balanced.pkl"

Sonuç:
  - Eğitim süresi: ~45 saniye
  - MSE: 0.06
  - Notlar: İyi denge
```

**Model C: "Accurate"**
```
Hidden Units: 50
Learning Rate: 0.008
Sequence Length: 35
Epochs: 250

Eğit → Test → Kaydet: "poly_model_accurate.pkl"

Sonuç:
  - Eğitim süresi: ~90 saniye
  - MSE: 0.03
  - Notlar: En iyi doğruluk
```

**Model D: "Experimental"** (relu activation)
```
Hidden Units: 40
Learning Rate: 0.01
Sequence Length: 25
Activation: relu  (farklı!)
Epochs: 150

Eğit → Test → Kaydet: "poly_model_relu.pkl"

Sonuç:
  - Eğitim süresi: ~40 saniye
  - MSE: 0.08
  - Notlar: ReLU bu problem için tanh'dan kötü
```

**Karşılaştırma Tablosu**:
| Model | Hidden | LR | SeqLen | Act | Epochs | Time | MSE | Score |
|-------|--------|-------|--------|-----|--------|------|-----|-------|
| Fast | 15 | 0.020 | 15 | tanh | 100 | 20s | 0.12 | ⭐⭐ |
| Balanced | 30 | 0.010 | 25 | tanh | 150 | 45s | 0.06 | ⭐⭐⭐⭐ |
| Accurate | 50 | 0.008 | 35 | tanh | 250 | 90s | 0.03 | ⭐⭐⭐⭐⭐ |
| Experimental | 40 | 0.010 | 25 | relu | 150 | 40s | 0.08 | ⭐⭐⭐ |

**Sonuç**: "Balanced" en iyi MSE/süre dengesi!

---

### Örnek 9: Aktivasyon Fonksiyonu Analizi

**Hedef**: tanh vs relu karşılaştırması

**Test Veri Setleri**:

**Dataset 1: Sine Wave** ([-1, 1] aralığında)
```
Samples: 500, Frequency: 1.0, Noise: 0.05

Model Tanh:
  Hidden: 25, LR: 0.01, SeqLen: 20, Epochs: 100
  MSE: 0.018
  Gözlem: Mükemmel, tanh [-1,1] için ideal

Model ReLU:
  Hidden: 25, LR: 0.01, SeqLen: 20, Epochs: 100
  MSE: 0.035
  Gözlem: İyi ama tanh kadar değil
  
Kazanan: TANH ✓
```

**Dataset 2: Exponential** (pozitif, büyüyen)
```
Samples: 600, Growth: 0.02, Noise: 0.05

Model Tanh:
  Hidden: 30, LR: 0.01, SeqLen: 25, Epochs: 150
  MSE: 0.08
  Gözlem: Zorlanıyor, saturation problemi

Model ReLU:
  Hidden: 30, LR: 0.01, SeqLen: 25, Epochs: 150
  MSE: 0.05
  Gözlem: Daha iyi, pozitif değerler için uygun
  
Kazanan: RELU ✓
```

**Dataset 3: Square Wave** (keskin geçişler)
```
Samples: 500, Frequency: 0.5, Noise: 0.02

Model Tanh:
  Hidden: 35, LR: 0.01, SeqLen: 20, Epochs: 150
  MSE: 0.045
  Gözlem: Kenarlar yuvarlatılmış

Model ReLU:
  Hidden: 35, LR: 0.01, SeqLen: 20, Epochs: 150
  MSE: 0.052
  Gözlem: Daha yuvarlatılmış, dying ReLU problemi
  
Kazanan: TANH ✓
```

**Genel Kural**:
- **Tanh**: Bounded data ([-1,1]), smooth patterns → Önerilen
- **ReLU**: Unbounded data, sparse activations → Bazı durumlarda

---

## Parametre Optimizasyon Örnekleri

### Örnek 10: Grid Search ile En İyi Parametreleri Bulma

**Hedef**: Sistematik parametre araması

**Problem**: Mixed Waves optimal konfigürasyonu

**Veri** (sabit):
```
Wave Type: Mixed Waves
Samples: 800
Frequency: 1.5
Noise: 0.1
```

**Grid Search Parametreleri**:
```
Hidden Units: [20, 30, 40]
Learning Rate: [0.005, 0.01, 0.02]
Sequence Length: [20, 30]
```

**Toplam Kombinasyon**: 3 × 3 × 2 = 18 test

**Prosedür**:

1. **Her kombinasyon için**:
   - Modeli initialize et
   - Veriyi üret (aynı seed için aynı olacak)
   - 100 epoch eğit
   - Test et
   - MSE kaydet
   - Modeli kaydet (örn: `grid_h20_lr005_s20.pkl`)

2. **Sonuçları Kaydet**:

| # | Hidden | LR | SeqLen | MSE | Time |
|---|--------|-------|--------|------|------|
| 1 | 20 | 0.005 | 20 | 0.095 | 25s |
| 2 | 20 | 0.005 | 30 | 0.088 | 28s |
| 3 | 20 | 0.010 | 20 | 0.082 | 24s |
| 4 | 20 | 0.010 | 30 | 0.079 | 27s |
| 5 | 20 | 0.020 | 20 | 0.091 | 23s |
| 6 | 20 | 0.020 | 30 | 0.086 | 26s |
| 7 | 30 | 0.005 | 20 | 0.071 | 35s |
| 8 | 30 | 0.005 | 30 | 0.065 | 38s |
| 9 | 30 | 0.010 | 20 | **0.058** | 34s | ← BEST!
| 10 | 30 | 0.010 | 30 | 0.062 | 37s |
| 11 | 30 | 0.020 | 20 | 0.074 | 33s |
| 12 | 30 | 0.020 | 30 | 0.068 | 36s |
| 13 | 40 | 0.005 | 20 | 0.069 | 48s |
| 14 | 40 | 0.005 | 30 | 0.063 | 51s |
| 15 | 40 | 0.010 | 20 | 0.061 | 47s |
| 16 | 40 | 0.010 | 30 | 0.059 | 50s |
| 17 | 40 | 0.020 | 20 | 0.076 | 46s |
| 18 | 40 | 0.020 | 30 | 0.071 | 49s |

**En İyi Konfigürasyon**:
```
✓ Hidden Units: 30
✓ Learning Rate: 0.01
✓ Sequence Length: 20
✓ MSE: 0.058
✓ Time: 34s (orta)
```

**İkinci En İyi** (daha yavaş ama biraz daha iyi):
```
• Hidden Units: 40
• Learning Rate: 0.01
• Sequence Length: 30
• MSE: 0.059
• Time: 50s
```

**İçgörüler**:
- Hidden 30-40 arası optimal
- LR 0.01 en dengeli
- SeqLen 20 yeterli (30 çok az iyileştirme)
- LR 0.02 çok yüksek (her hidden size'da kötü)

---

## Hata Ayıklama Senaryoları

### Senaryo 1: Loss Artıyor (Divergence)

**Problem Durumu**:
```
Model: Hidden=25, LR=0.1, SeqLen=20
Data: Sine Wave
Epochs: 50

Gözlem:
  - Loss başlangıçta 0.5
  - Epoch 10'da 1.2
  - Epoch 20'de 3.5
  - Epoch 30'da 8.9
  - Loss patladı!
```

**Teşhis**:
- Learning rate çok yüksek
- Ağırlıklar optimize noktasını aşıyor

**Çözüm Adımları**:

1. **Stop Training** butonuna bas
2. Learning Rate'i 0.01'e düşür
3. "Initialize Model" ile modeli sıfırla
4. "Start Training" tekrar başlat

**Sonuç**:
```
Epochs: 50
Final Loss: 0.003
Durum: Çözüldü ✓
```

**Önleyici Tedbirler**:
- LR > 0.05 kullanma
- Gradient clipping aktif (otomatik)
- İlk birkaç epoch'u izle

---

### Senaryo 2: Loss Platolaştı

**Problem Durumu**:
```
Model: Hidden=10, LR=0.01, SeqLen=20
Data: Mixed Waves
Epochs: 200

Gözlem:
  - Epoch 0-50: Loss 0.5 → 0.2
  - Epoch 50-100: Loss 0.2 → 0.15
  - Epoch 100-200: Loss 0.15 → 0.15 (değişmiyor!)
```

**Teşhis**:
- Model kapasitesi yetersiz
- Underfitting

**Çözüm Adımları**:

1. **Stop Training**
2. Hidden Units'i 30'a çıkar
3. "Initialize Model"
4. Veriyi tekrar üret (aynı olacak)
5. "Start Training"

**Sonuç**:
```
Epochs: 200
Final Loss: 0.05
Durum: Çözüldü ✓ (daha fazla kapasite yardım etti)
```

**Alternatif Çözümler**:
- Learning Rate artır (0.01 → 0.02)
- Sequence Length artır
- Her ikisi

---

### Senaryo 3: Overfitting Tespit

**Problem Durumu**:
```
Model: Hidden=80, LR=0.01, SeqLen=20
Data: Sine Wave (clean, noise=0.0)
Epochs: 300

Training Sonrası:
  - Training MSE: 0.001 (mükemmel!)
  
Yeni Test Verisi Üret (aynı tip, farklı gürültü):
  - Test MSE: 0.15 (çok kötü!)
```

**Teşhis**:
- Model eğitim verisini ezberled i
- Generalize edemiyor

**Çözüm Adımları**:

**Yöntem 1: Regularization (Gürültü)**
```
1. Noise Level'i 0.1'e çıkar
2. Modeli yeniden eğit
3. Test et

Sonuç:
  - Training MSE: 0.025
  - Test MSE: 0.03
  - Durum: İyi generalization ✓
```

**Yöntem 2: Model Kapasitesi Azalt**
```
1. Hidden Units: 80 → 30
2. Modeli yeniden eğit
3. Test et

Sonuç:
  - Training MSE: 0.015
  - Test MSE: 0.02
  - Durum: Daha dengeli ✓
```

**Yöntem 3: Erken Durdurma**
```
1. 300 epochs yerine 100 epochs
2. Loss plato olduğunda dur
```

---

### Senaryo 4: Yavaş Eğitim

**Problem Durumu**:
```
Model: Hidden=100, LR=0.005, SeqLen=50
Data: 2000 samples
Epochs: 500

Gözlem:
  - Her epoch ~5 saniye sürüyor
  - Toplam: 500 × 5s = ~40 dakika!
  - Çok yavaş!
```

**Teşhis**:
- Çok büyük model
- Çok fazla veri
- Uzun sequence

**Hızlandırma Adımları**:

**Optimizasyon 1: Parametreleri Azalt**
```
Hidden: 100 → 50
SeqLen: 50 → 30
Samples: 2000 → 1000

Sonuç: Her epoch ~1.5 saniye (3x hızlanma)
MSE: Sadece %10 kötüleşti
```

**Optimizasyon 2: Erken Durdurma**
```
Epochs: 500 → 150
(Loss 100. epoch'tan sonra çok az iyileşiyordu)

Sonuç: Toplam süre 7.5 dakika
MSE: Hemen hemen aynı
```

**Optimizasyon 3: Learning Rate Artır**
```
LR: 0.005 → 0.015
Epochs: 150 → 100

Sonuç: Daha az epoch'ta aynı sonuç
Toplam süre: 5 dakika
```

**Final Konfigürasyon**:
```
Hidden: 50
LR: 0.015
SeqLen: 30
Samples: 1000
Epochs: 100

Toplam süre: 5 dakika (40'dan 8x hızlanma!)
MSE: Orijinal ile %95 aynı
```

---

### Senaryo 5: Tahminler Tamamen Yanlış

**Problem Durumu**:
```
Model eğitildi, loss azaldı, ama tahminler saçma

Training:
  - Final Loss: 0.01 (iyi görünüyor)
  - 100 epochs

Test Prediction:
  - Gerçek veri: Sinüs dalgası [-1, 1]
  - Tahminler: Sabit çizgi (0.0)
  - Model hiçbir şey öğrenmemiş gibi!
```

**Olası Nedenler ve Çözümler**:

**Neden 1: Veri Normalizasyonu Problemi**
```
Kontrol: Data min/max değerleri
Çözüm: Veriyi yeniden üret (otomatik normalize olur)
```

**Neden 2: Model Initialize Edilmemiş**
```
Kontrol: "Initialize Model" butonu basıldı mı?
Çözüm: Model parametrelerini ayarla ve initialize et
```

**Neden 3: Yanlış Sequence Length**
```
Kontrol: SeqLen çok mu kısa? (örn: SeqLen=2)
Çözüm: SeqLen'i en az 10'a çıkar
```

**Neden 4: Vanishing Gradients**
```
Kontrol: Çok uzun sequence (>50) kullanıldı mı?
Çözüm: SeqLen'i azalt, LR artır
```

**Debug Prosedürü**:
```
1. Model Info butonuna bas → Parametreleri kontrol et
2. Basit veri ile test et (Sine, noise=0.0)
3. Küçük model ile test et (Hidden=10, epochs=50)
4. Çalışırsa, parametreleri yavaşça artır
```

---

## 🎓 Sonuç ve İpuçları

### Genel Başlangıç Tavsiyesi:
```
✓ Hidden Units: 20-30
✓ Learning Rate: 0.01
✓ Sequence Length: 20
✓ Activation: tanh
✓ Epochs: 100
✓ Data: Sine Wave, 500 samples, noise=0.05
```

### İlerleme Yolu:
1. Basit → Karmaşık veri
2. Küçük → Büyük model
3. Az → Çok epoch
4. Temiz → Gürültülü veri

### Deneme Sırası:
1. Veri tipini değiştir
2. Gürültü seviyesini ayarla
3. Learning rate optimize et
4. Hidden units ayarla
5. Sequence length bul
6. Aktivasyon dene

Başarılar! 🚀
