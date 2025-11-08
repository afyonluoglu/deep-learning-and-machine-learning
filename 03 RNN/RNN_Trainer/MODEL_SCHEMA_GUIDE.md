# Model Schema Özelliği - Kullanım Kılavuzu

## 🎨 Genel Bakış

**Model Schema** özelliği, RNN modelinizin mimari yapısını görsel olarak göstermenizi ve kaydetmenizi sağlar.

## 📊 Özellikler

### 1. Görsel Şema
- ✅ Input (Giriş) katmanı
- ✅ Tüm Hidden (Gizli) katmanlar
- ✅ Output (Çıkış) katmanı
- ✅ Katmanlar arası bağlantılar (oklar)
- ✅ Recurrent bağlantılar (zaman içinde geri besleme)
- ✅ Her katmandaki nöron sayıları
- ✅ Model parametreleri ve hyperparameter'lar

### 2. Renk Kodlaması
- 🔵 **Mavi**: Input Layer (Giriş)
- 🟢 **Yeşil**: Hidden Layers (Gizli Katmanlar)
- 🔴 **Kırmızı**: Output Layer (Çıkış)
- ⚪ **Gri**: Bağlantı okları

### 3. Gösterilen Bilgiler

#### Üst Kısım:
- Model tipi (Single-layer / Multi-layer Deep RNN)
- Tam mimari: Input → Hidden Layers → Output

#### Her Katman:
- Katman adı (INPUT, HIDDEN 1, HIDDEN 2, ..., OUTPUT)
- Nöron sayısı (büyük font ile ortada)
- Katman boyutu (altında küçük yazıyla)

#### Alt Kısım:
- **Toplam Parametre Sayısı**: Modeldeki tüm ağırlıklar
- **Activation**: Aktivasyon fonksiyonu (tanh/relu)
- **Dropout**: Regularization oranı
- **Optimizer**: Kullanılan optimizer (SGD/Adam/etc)
- **Sequence Length**: Dizi uzunluğu
- **Learning Rate**: Öğrenme hızı

#### Özel İşaretler:
- **→**: Forward pass (ileri besleme)
- **↻**: Recurrent connection (zaman içinde geri besleme)

## 🚀 Nasıl Kullanılır?

### Adım 1: Model Oluşturma
```
1. Model Parameters bölümünden parametreleri ayarlayın
2. "Initialize Model" butonuna tıklayın
```

### Adım 2: Şemayı Görüntüleme
```
1. Model Management bölümünde "📊 Model Schema" butonuna tıklayın
2. Yeni bir pencere açılır ve model şeması gösterilir
```

### Adım 3: Not Ekleme (Opsiyonel)
```
1. Şema penceresinin altındaki metin kutusuna notlarınızı yazın
2. Örnek: "Bu model için en iyi learning rate 0.01 bulundu"
3. Notlar şema ile birlikte kaydedilecektir
```

### Adım 4: Şemayı Kaydetme
```
1. "💾 Save Schema as PNG" butonuna tıklayın
2. Şema otomatik olarak outputs/ klasörüne kaydedilir
3. Dosya adı: model_schema_YYYYMMDD_HHMMSS.png
4. Notlarınız varsa, şemanın altında görünür
```

## 📸 Örnek Şemalar

### Tek Katmanlı Model
```
INPUT (1) → HIDDEN 1 (20) → OUTPUT (1)
```
- Basit, klasik RNN
- ~461 parametre

### İki Katmanlı Model
```
INPUT (1) → HIDDEN 1 (30) → HIDDEN 2 (20) → OUTPUT (1)
```
- Deep RNN
- ~2,001 parametre
- Daha karmaşık desenler öğrenir

### Üç Katmanlı Model
```
INPUT (1) → HIDDEN 1 (50) → HIDDEN 2 (30) → HIDDEN 3 (20) → OUTPUT (1)
```
- Derin RNN (Deep RNN)
- ~4,571 parametre
- Hiyerarşik özellik öğrenme

## 💡 İpuçları

### Şema Kaydetme:
1. **Notlar Ekleyin**: Model hakkında önemli bilgileri not edin
   - En iyi hyperparameter'lar
   - Elde edilen loss değeri
   - Kullanım amacı
   - Deneme tarihi

2. **Karşılaştırma**: Farklı mimarileri karşılaştırmak için
   - Her modelin şemasını kaydedin
   - Notlarda performans metrikleri yazın
   - Görsel olarak karşılaştırın

3. **Dokümantasyon**: Raporlar ve sunumlar için
   - Yüksek çözünürlükte kaydedilir (150 DPI)
   - Beyaz arka plan ile temiz görünüm
   - Doğrudan kullanıma hazır

### Not Örnekleri:

**Örnek 1 - Performans Notu:**
```
Model trained for 50 epochs. Final loss: 0.0234
Best configuration found: lr=0.01, dropout=0.2
Used for sine wave prediction with 95% accuracy
```

**Örnek 2 - Karşılaştırma Notu:**
```
Comparison Test #3
- Better than 2-layer model (loss: 0.0345)
- Training time: 2.5 min
- Recommended for production use
```

**Örnek 3 - Deney Notu:**
```
Experiment: Testing dropout effect
Baseline: 0.0 dropout → loss 0.0567 (overfitting)
This model: 0.3 dropout → loss 0.0289 (better generalization)
```

## 🎯 Kullanım Senaryoları

### 1. Model Geliştirme
- Farklı mimarileri görsel olarak karşılaştırın
- Katman sayısının etkisini görün
- Parametre sayısını takip edin

### 2. Öğrenme ve Eğitim
- RNN mimarisini öğrencilere gösterin
- Recurrent bağlantıları açıklayın
- Katman yapısını anlayın

### 3. Raporlama
- Proje raporlarında kullanın
- Sunumlara ekleyin
- Dokümantasyon oluşturun

### 4. Debug ve Analiz
- Model yapısını doğrulayın
- Parametre sayısını kontrol edin
- Mimari hataları tespit edin

## 📁 Kayıt Formatı

### Dosya Adı:
```
model_schema_20250101_143025.png
           └─ Tarih    └─ Saat
```

### Kayıt Yeri:
```
RNN_Trainer/
  └── outputs/
      ├── model_schema_20250101_143025.png
      ├── model_schema_20250101_150312.png
      └── ...
```

### Dosya Özellikleri:
- **Format**: PNG
- **Çözünürlük**: 150 DPI
- **Boyut**: ~8x6 inç (genişlik x yükseklik)
- **Arka Plan**: Beyaz (baskı için uygun)
- **Notlar ile**: ~8x7 inç (notlar eklendiğinde)

## 🎨 Şema Elemanları

### Kutular (Layers):
```
┌─────────────┐
│   INPUT     │  ← Katman adı
│      1      │  ← Nöron sayısı (büyük)
└─────────────┘
   Size: 1       ← Katman boyutu (küçük)
```

### Oklar:
```
→  Forward Pass (katmanlar arası)
↻  Recurrent Connection (zaman boyunca)
```

### Bilgi Kutuları:
```
┌────────────────────────────────┐
│ Input: 1 → 30 → 20 → Output: 1│  ← Mimari özeti
└────────────────────────────────┘

┌────────────────────────────────┐
│ Total Parameters: 2,001        │  ← Parametre bilgileri
│ Activation: tanh | Dropout: 0.2│
└────────────────────────────────┘
```

## 🔍 Teknik Detaylar

### Recurrent Connection (↻):
- Her hidden layer'ın üstünde bir döngü gösterilir
- Bu, RNN'in zaman içinde geri besleme özelliğidir
- t anında hesaplanan hidden state, t+1 anında tekrar kullanılır

### Parametre Hesaplama:
Her katman için:
- **Wxh**: input_size × hidden_size (veya prev_hidden × hidden_size)
- **Whh**: hidden_size × hidden_size (recurrent weights)
- **bh**: hidden_size (bias)

Output layer için:
- **Why**: last_hidden_size × output_size
- **by**: output_size (bias)

## 🚀 Gelişmiş Kullanım

### 1. Seri Şema Kaydetme
```python
# Farklı konfigürasyonları test ederken
for num_layers in [1, 2, 3]:
    # Model oluştur
    # Eğit
    # Schema kaydet (notlara loss yaz)
    # Karşılaştır
```

### 2. Notlar ile Otomatik Raporlama
```python
# Not formatı:
note = f"""
Configuration: {num_layers} layers
Training: {num_epochs} epochs
Final Loss: {final_loss:.4f}
Test Accuracy: {test_acc:.2%}
Training Time: {train_time:.1f}s
"""
```

## ❓ Sık Sorulan Sorular

**S: Şema otomatik olarak güncellenmiyor?**
C: Model Schema düğmesine her tıkladığınızda güncel model gösterilir.

**S: Notlar olmadan kaydedebilir miyim?**
C: Evet! Not kutusunu boş bırakın veya varsayılan metni silmeyin.

**S: Kaydedilen şemaları nasıl bulabilirim?**
C: `outputs/` klasöründe `model_schema_*.png` dosyalarını arayın.

**S: Şemayı farklı formatta kaydedebilir miyim?**
C: Şu anda sadece PNG formatı destekleniyor (en yaygın ve yüksek kaliteli format).

**S: Çok katmanlı modelde tüm katmanlar görünmüyor?**
C: Pencere boyutunu büyütün veya kaydedilen PNG dosyasını açın (daha geniş).

## 🎉 Sonuç

Model Schema özelliği ile:
- ✅ Modelinizi görsel olarak anlayın
- ✅ Farklı mimarileri karşılaştırın
- ✅ Raporlarınızı zenginleştirin
- ✅ Öğrenme sürecinizi kolaylaştırın
- ✅ Profesyonel dokümantasyon oluşturun

**İyi görselleştirmeler! 🎨**
