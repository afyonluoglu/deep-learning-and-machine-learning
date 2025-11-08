# RNN Trainer - Quick Reference Card

## 🚀 Hızlı Başlangıç (5 Adım)

```
1. Initialize Model    → Hidden: 20, LR: 0.01, SeqLen: 20
2. Generate Data       → Sine Wave, 500 samples
3. Set Epochs          → 100
4. Start Training      → Wait for completion
5. Test Prediction     → Compare results
```

## 📊 Parametreler Hızlı Referans

### Hidden Units
| Değer | Kullanım | Hız | Doğruluk |
|-------|----------|-----|----------|
| 5-15  | Basit    | ⚡⚡⚡ | ⭐⭐     |
| 20-40 | Normal   | ⚡⚡  | ⭐⭐⭐⭐ |
| 50+   | Karmaşık | ⚡   | ⭐⭐⭐⭐⭐|

### Learning Rate
| Değer      | Durum | Sonuç |
|------------|-------|-------|
| < 0.001    | Çok Yavaş | 😴 |
| 0.001-0.005| Yavaş | 🐢 |
| 0.01       | İdeal | ✅ |
| 0.01-0.03  | Hızlı | 🚀 |
| > 0.05     | Risk  | ⚠️  |

### Sequence Length
```
Veri Periyodu × 1.2 = Optimal Sequence Length

Örnek:
  Sine (periyot ~20) → SeqLen = 25
  Square (periyot ~10) → SeqLen = 12
```

## 🎯 Veri Tipleri Önerilen Parametreler

### Basit Dalga (Sine/Cosine)
```
Hidden: 20
LR: 0.01
SeqLen: 20
Epochs: 100
Beklenen MSE: < 0.02
```

### Karmaşık Dalga (Mixed/ARMA)
```
Hidden: 40
LR: 0.008
SeqLen: 30
Epochs: 200
Beklenen MSE: < 0.08
```

### Trend (Exponential/Polynomial)
```
Hidden: 30
LR: 0.01
SeqLen: 25
Epochs: 150
Beklenen MSE: < 0.10
```

## 🔧 Sorun Giderme - Hızlı Kılavuz

### Loss Artıyor ⬆️
```
→ LR'yi yarıya indir
→ Model'i yeniden initialize et
```

### Loss Çok Yavaş Azalıyor 🐌
```
→ LR'yi 2x artır
→ Hidden units artır
→ Daha fazla epoch
```

### Loss Takıldı (Plateau) ═══
```
→ Hidden units artır
→ SeqLen ayarla
→ LR hafif artır
```

### Kötü Tahminler 📉
```
→ Daha uzun eğit
→ Model kapasitesini artır
→ Veriyi kontrol et
```

## 💡 İpuçları

### Hızlı Test
```
Samples: 500
Epochs: 50
Hidden: 15
→ Sonuç: ~20 saniye
```

### Dengeli Kullanım
```
Samples: 800
Epochs: 100
Hidden: 30
→ Sonuç: ~45 saniye
```

### Yüksek Doğruluk
```
Samples: 1000
Epochs: 200
Hidden: 50
→ Sonuç: ~2 dakika
```

## 📈 MSE Yorumlama

```
MSE < 0.01     →  Mükemmel! 🏆
MSE 0.01-0.05  →  Çok İyi   ✅
MSE 0.05-0.10  →  İyi       👍
MSE 0.10-0.20  →  Orta      😐
MSE > 0.20     →  Zayıf     ❌
```

## ⌨️ Klavye Kısayolları

```
Model Kaydet:  Ctrl+S (Save Model butonu)
Model Yükle:   Ctrl+O (Load Model butonu)
Yardım:        F1 (Help butonu)
```

## 🎓 En İyi Pratikler

### 1. Her Zaman
- ✅ Basit veriden başla
- ✅ Loss grafiğini izle
- ✅ İyi modelleri kaydet

### 2. Asla
- ❌ LR > 0.1 kullanma
- ❌ İlk denemede karmaşık veri
- ❌ Eğitimi izlemeden bırakma

### 3. Deneme Sırası
```
1. Veri tipi
2. Gürültü seviyesi
3. Learning rate
4. Hidden units
5. Sequence length
6. Aktivasyon fonksiyonu
```

## 🔬 Experiment Şablonu

```python
# Deney İsmi: ________________
# Tarih: ____________________

Model:
  Hidden Units: ___
  Learning Rate: ___
  Sequence Length: ___
  Activation: ___

Data:
  Type: ___
  Samples: ___
  Frequency: ___
  Noise: ___

Training:
  Epochs: ___
  
Results:
  Final Loss: ___
  MSE: ___
  Time: ___
  
Notes:
  ________________________
  ________________________
```

## 📱 Durum İkonları

```
🟢 Hazır     → Model ve veri yüklü
🟡 Eğitim    → Training devam ediyor
🔵 Test      → Prediction yapılıyor
🔴 Hata      → Bir sorun var
⚪ Bekliyor  → Kullanıcı girişi gerekli
```

## 🎯 Hedef MSE Değerleri

### Veri Tipine Göre
```
Sine Wave:          < 0.02
Cosine Wave:        < 0.02
Square Wave:        < 0.05
Triangle Wave:      < 0.03
Sawtooth Wave:      < 0.04
Mixed Waves:        < 0.10
Exponential:        < 0.12
Polynomial:         < 0.15
Random Walk:        < 0.20
ARMA:               < 0.08
Damped Oscillation: < 0.06
```

## 🔄 Tipik İş Akışı

```
Başla
  ↓
Model Oluştur (Hidden, LR, SeqLen)
  ↓
Veri Üret (Tip, Samples, Noise)
  ↓
Eğit (Epochs ayarla, Start)
  ↓
İzle (Loss düşüyor mu?)
  ├─ Hayır → Parametreleri ayarla, tekrar eğit
  └─ Evet → Devam
       ↓
Test Et (Prediction)
  ↓
MSE Kontrol
  ├─ İyi → Modeli kaydet ✅
  └─ Kötü → Daha fazla epoch veya parametre ayarla
       ↓
Farklı veri ile test et
  ↓
En iyi modeli kullan
```

## 💾 Dosya Yönetimi

### Model Dosyaları
```
my_model.pkl          → Model ağırlıkları
my_model_config.json  → Normalizasyon bilgisi
```

### İsimlendirme Önerisi
```
[veri_tipi]_[hidden]h_[lr]lr_v[versiyon].pkl

Örnekler:
  sine_20h_001lr_v1.pkl
  mixed_40h_008lr_v2.pkl
  arma_30h_010lr_final.pkl
```

## 📊 Grafik Yorumlama

### Loss Grafiği (Alt Panel)
```
İdeal:    \___        (düzenli düşüş, sonra düz)
Yavaş:    \____       (çok yavaş azalma)
Hızlı:    \___        (hızlı düşüş)
Problem:  \/\/\/\     (salınımlı)
Hata:     /           (artıyor!)
```

### Prediction Grafiği (Üst Panel)
```
Mükemmel: Mavi ve kırmızı çizgiler üst üste
İyi:      Küçük sapmalar
Orta:     Genel trend doğru, detaylar farklı
Kötü:     Tamamen farklı
```

## 🎨 Renk Kodları

```
Mavi (Blue):      Gerçek veri
Kırmızı (Red):    Model tahmini
Yeşil (Green):    Initialize butonu
Turuncu (Orange): Training butonu
Mor (Purple):     Data generation
Mavi (Blue):      Save butonu
```

## ⏱️ Süre Tahminleri

```
Parametreler: Hidden=30, SeqLen=20, Samples=500

Epochs:
  50   → ~20 saniye
  100  → ~40 saniye
  200  → ~80 saniye
  500  → ~3 dakika
```

## 🏆 Başarı Kriterleri

```
Başarılı Eğitim:
  ✓ Loss düzenli azalıyor
  ✓ MSE hedef değerin altında
  ✓ Prediction grafiği uyumlu
  ✓ Test verisinde de iyi sonuç

Başarısız Eğitim:
  ✗ Loss artıyor veya sabit
  ✗ MSE çok yüksek
  ✗ Prediction saçma
  ✗ Eğitim çok yavaş
```

---

**Son Tavsiye**: Sabırlı olun, deneyerek öğrenin! 🚀
