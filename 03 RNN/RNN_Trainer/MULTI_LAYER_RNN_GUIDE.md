# Çok Katmanlı RNN (Multi-Layer/Stacked RNN) Kullanım Kılavuzu

## 🎯 Genel Bakış

Bu programda artık **tek katmanlı** ve **çok katmanlı (stacked/deep)** RNN modelleri oluşturabilirsiniz!

## 📊 Çok Katmanlı RNN Nedir?

Çok katmanlı RNN (Deep RNN veya Stacked RNN), birden fazla gizli katmanın üst üste yığıldığı bir yapıdır:

```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer N → Output
```

Her katman, bir önceki katmanın çıktısını girdi olarak alır ve daha soyut özellikler öğrenir.

## 🎓 Neden Çok Katmanlı RNN Kullanmalı?

### ✅ Avantajları:

1. **Daha Karmaşık Desenler**: Her katman farklı soyutlama seviyelerinde özellikler öğrenir
2. **Hiyerarşik Temsiller**: 
   - Alt katmanlar: Basit, lokal desenler
   - Üst katmanlar: Karmaşık, global desenler
3. **Daha İyi Performans**: Karmaşık zaman serilerinde tek katmana göre daha iyi sonuçlar
4. **Zengin Özellik Çıkarımı**: Her katman bilgiyi dönüştürür ve zenginleştirir

### ⚠️ Dezavantajları:

1. **Daha Fazla Hesaplama**: Eğitim süresi artar
2. **Daha Fazla Veri Gereksinimi**: Daha fazla parametre = daha fazla eğitim verisi
3. **Overfitting Riski**: Küçük veri setlerinde aşırı öğrenme riski
4. **Gradient Problemi**: Vanishing/exploding gradient sorunu daha şiddetli olabilir

## 🛠️ Programda Nasıl Kullanılır?

### 1. **Hidden Layers (Gizli Katman Sayısı)**

**Model Parameters** bölümünde:
- **Slider**: 1-5 arası katman sayısı seçin
- **1**: Tek katmanlı (klasik RNN)
- **2+**: Çok katmanlı (deep RNN)

### 2. **Hidden Units (Nöron Sayısı)**

- Tüm katmanlar için varsayılan nöron sayısı
- Örnek: 20 seçerseniz, tüm katmanlar 20'şer nöron içerir

### 3. **Layer Sizes (Katman Boyutları)**

Virgülle ayrılmış her katmanın nöron sayısını özelleştirebilirsiniz:

**Örnekler:**
```
30,20,10     → 3 katmanlı: 30 → 20 → 10
50,40,30,20  → 4 katmanlı: 50 → 40 → 30 → 20
25,15        → 2 katmanlı: 25 → 15
```

**Not**: Boş bırakırsanız, tüm katmanlar "Hidden Units" değerini kullanır.

## 📈 Önerilen Yapılandırmalar

### Başlangıç Seviyesi (Hızlı Test)
```
Layers: 1
Hidden Units: 20
```

### Orta Seviye (İyi Performans)
```
Layers: 2
Layer Sizes: 30,20
Dropout: 0.2
Optimizer: adam
```

### İleri Seviye (Maksimum Performans)
```
Layers: 3
Layer Sizes: 50,30,20
Dropout: 0.3
Optimizer: adam
Learning Rate: 0.001
LR Schedule: exponential
```

### Çok Karmaşık Veriler
```
Layers: 4
Layer Sizes: 100,70,50,30
Dropout: 0.4
Optimizer: adam
Sequence Length: 30
```

## 🔬 Farklı Katman Yapıları

### 1. Pyramid (Piramit) Yapısı
```
Layer Sizes: 50,40,30,20
```
- Her katman azalarak gider
- **Kullanım**: En yaygın yapı, çoğu durumda iyi çalışır
- **Avantaj**: Bilgi sıkıştırması ve özellik seçimi

### 2. Uniform (Eşit) Yapısı
```
Layer Sizes: 30,30,30
```
- Tüm katmanlar eşit boyutta
- **Kullanım**: Simetrik problemler
- **Avantaj**: Her katman aynı kapasitede

### 3. Inverted Pyramid (Ters Piramit)
```
Layer Sizes: 20,30,40
```
- Her katman artarak gider
- **Kullanım**: Özellik zenginleştirme gerektiğinde
- **Avantaj**: Bilgi genişletme

### 4. Bottleneck (Darboğaz)
```
Layer Sizes: 40,10,40
```
- Ortada küçük, yanlarda büyük
- **Kullanım**: Autoencoder benzeri yapılar
- **Avantaj**: Boyut indirgeme

## 💡 İpuçları

### Katman Sayısı Seçimi:
- **1 katman**: Basit, lineer-benzeri desenler
- **2 katman**: Orta karmaşıklıkta zaman serileri (önerilen başlangıç)
- **3 katman**: Karmaşık, uzun vadeli bağımlılıklar
- **4-5 katman**: Çok karmaşık, hiyerarşik desenler (dikkatli kullanın!)

### Nöron Sayısı Seçimi:
- **Az veri** (< 500 örnek): 10-20 nöron/katman
- **Orta veri** (500-2000 örnek): 20-50 nöron/katman
- **Çok veri** (> 2000 örnek): 50-100 nöron/katman

### Dropout ile Birlikte Kullanım:
```
2-3 katman → Dropout 0.2-0.3
4+ katman  → Dropout 0.3-0.5
```

### Optimizer Önerisi:
- **Adam**: Çok katmanlı modeller için en iyi seçim
- **RMSprop**: İyi alternatif
- **SGD/Momentum**: Daha yavaş ama bazen daha iyi sonuç

## 📊 Model Bilgisi

**Model Info** butonuna tıklayarak:
- Katman sayısını
- Her katmanın boyutunu
- Toplam parametre sayısını
- Mimari yapıyı görebilirsiniz

## 🎯 Örnek Kullanım Senaryosu

### Senaryo: Karmaşık sinüs dalgası tahmin etme

1. **Data Generation**
   - Wave Type: Mixed Waves
   - Samples: 1000
   - Frequency: 2.0
   - Noise: 0.1

2. **Model Parameters**
   - Layers: 3
   - Layer Sizes: 40,30,20
   - Activation: tanh
   - Dropout: 0.3
   - Optimizer: adam
   - LR Schedule: exponential

3. **Training**
   - Epochs: 50
   - Learning Rate: 0.01

4. **Sonuç**: Çok katmanlı model, tek katmanlı modelden daha düşük loss değerine ulaşır!

## 🚀 Deneyler

Aşağıdaki deneyleri yaparak farkı görebilirsiniz:

### Deney 1: Tek vs Çok Katman
1. Tek katman (20 nöron) ile eğitin
2. İki katman (30→20) ile eğitin
3. Loss grafiklerini karşılaştırın

### Deney 2: Katman Sayısının Etkisi
1. 1, 2, 3, 4 katman ile ayrı ayrı eğitin
2. Her birinin loss'unu ve eğitim süresini karşılaştırın

### Deney 3: Farklı Yapılar
1. Piramit: 50→30→10
2. Eşit: 30→30→30
3. Ters Piramit: 10→30→50
4. Hangisi daha iyi?

## ⚙️ Teknik Detaylar

### Forward Pass (İleri Besleme):
```python
# Her katman için
for layer in range(num_layers):
    if layer == 0:
        input = x  # İlk katman: girdi verisi
    else:
        input = h[layer-1]  # Sonraki katmanlar: önceki katmanın çıktısı
    
    h[layer] = activation(Wxh[layer] @ input + Whh[layer] @ h_prev[layer])
```

### Backward Pass (Geri Yayılım):
```python
# Katmanlar ters sırada (sondan başa)
for layer in reversed(range(num_layers)):
    # Gradient hesaplama ve yayılım
    ...
```

## 🔍 Sorun Giderme

### Problem: Eğitim çok yavaş
**Çözüm**: 
- Katman sayısını azaltın
- Her katmandaki nöron sayısını azaltın
- Sequence length'i azaltın

### Problem: Overfitting (aşırı öğrenme)
**Çözüm**:
- Dropout'u artırın (0.3-0.5)
- Katman sayısını azaltın
- Daha fazla veri toplayın

### Problem: Underfitting (yetersiz öğrenme)
**Çözüm**:
- Katman sayısını artırın
- Her katmandaki nöron sayısını artırın
- Daha fazla epoch eğitin
- Learning rate'i artırın

### Problem: Gradient vanishing
**Çözüm**:
- ReLU aktivasyonu kullanın
- Learning rate'i artırın
- Katman sayısını azaltın
- Gradient clipping zaten aktif (±5)

## 📚 Daha Fazla Bilgi

- **LSTM**: Gradient vanishing problemine çözüm
- **GRU**: LSTM'in daha basit versiyonu
- **Bidirectional RNN**: İleri ve geri yönlü işleme
- **Attention Mechanism**: Önemli bölgelere odaklanma

---

**İyi Eğitimler! 🚀**
