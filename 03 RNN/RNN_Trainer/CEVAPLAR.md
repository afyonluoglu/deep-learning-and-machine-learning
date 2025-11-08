# RNN Çok Katmanlı Yapı - Hızlı Başvuru

## 🎯 Sorularınızın Cevapları

### 1. Programda Kaç Hidden Layer Var?
**Başlangıçta**: Tek bir hidden layer vardı.
**Şimdi**: 1-5 arası istediğiniz kadar hidden layer oluşturabilirsiniz!

### 2. RNN'de Birden Fazla Hidden Layer Olabilir mi?
**Evet!** Buna **Deep RNN** veya **Stacked RNN** denir.

**Yapı:**
```
Input → Layer 1 → Layer 2 → Layer 3 → ... → Output
```

### 3. Birden Fazla Layer'ın Faydaları Nelerdir?

#### ✅ Avantajlar:
1. **Hiyerarşik Öğrenme**
   - Layer 1: Basit, lokal desenler
   - Layer 2: Orta seviye desenler
   - Layer 3: Karmaşık, global desenler

2. **Daha İyi Performans**
   - Karmaşık zaman serilerinde daha düşük loss
   - Daha iyi tahmin doğruluğu

3. **Özellik Zenginleştirme**
   - Her layer bilgiyi dönüştürür
   - Daha soyut temsiller

4. **Uzun Vadeli Bağımlılıklar**
   - Uzak geçmişle daha iyi ilişki

#### ⚠️ Dezavantajlar:
1. **Daha Yavaş Eğitim**: Daha fazla hesaplama
2. **Daha Fazla Veri Gerekir**: Daha fazla parametre
3. **Overfitting Riski**: Küçük veri setlerinde
4. **Gradient Sorunları**: Vanishing/exploding

### 4. Programda Nasıl Tanımlanır?

#### A) GUI Kontrolleri:

**Model Parameters** bölümünde 3 yeni kontrol var:

1. **Hidden Layers Slider** (1-5)
   - 1: Tek katman (klasik)
   - 2-5: Çok katmanlı

2. **Hidden Units Slider** (5-100)
   - Tüm katmanlar için varsayılan nöron sayısı

3. **Layer Sizes Text Box**
   - Her katmanın nöron sayısını özelleştir
   - Örnek: `30,20,10` → 3 katman: 30→20→10

#### B) Kod İle:

```python
from rnn_model import RNNModel

# Tek katmanlı (klasik)
model = RNNModel(
    hidden_size=20,
    num_layers=1
)

# 2 katmanlı (30→20)
model = RNNModel(
    hidden_size=20,
    num_layers=2,
    hidden_sizes=[30, 20]
)

# 3 katmanlı (50→30→20)
model = RNNModel(
    hidden_size=20,
    num_layers=3,
    hidden_sizes=[50, 30, 20]
)
```

## 📊 Örnek Konfigürasyonlar

### Başlangıç (Basit)
```
Layers: 1
Hidden Units: 20
```

### Orta (Önerilen)
```
Layers: 2
Layer Sizes: 30,20
Dropout: 0.2
Optimizer: adam
```

### İleri (Karmaşık)
```
Layers: 3
Layer Sizes: 50,30,20
Dropout: 0.3
Optimizer: adam
LR Schedule: exponential
```

## 🔬 Katman Yapıları

### Pyramid (En Yaygın)
```
50 → 30 → 20 → 10
```
Bilgi sıkıştırması, özellik seçimi

### Uniform (Eşit)
```
30 → 30 → 30
```
Her katman aynı kapasitede

### Inverted Pyramid
```
10 → 20 → 30 → 40
```
Özellik zenginleştirme

### Bottleneck
```
40 → 10 → 40
```
Boyut indirgeme (autoencoder)

## 💡 Pratik Öneriler

### Ne Zaman Tek Katman?
- Basit, lineer-benzeri desenler
- Az veri (< 500 örnek)
- Hızlı prototip

### Ne Zaman 2 Katman?
- Orta karmaşıklıkta zaman serileri ⭐ (ÖNERİLEN)
- 500-2000 örnek veri
- İyi hız-performans dengesi

### Ne Zaman 3+ Katman?
- Çok karmaşık desenler
- Çok fazla veri (> 2000 örnek)
- Maksimum performans gerekli

## 🎯 Hızlı Test

1. **Veri Oluştur**:
   - Wave Type: Mixed Waves
   - Samples: 1000

2. **Model 1 - Tek Katman**:
   - Layers: 1
   - Hidden Units: 20
   - Train: 30 epochs
   - Loss'u not et

3. **Model 2 - İki Katman**:
   - Layers: 2
   - Layer Sizes: 30,20
   - Train: 30 epochs
   - Loss'u karşılaştır

**Sonuç**: İki katmanlı model genellikle daha düşük loss'a ulaşır! 🎉

## 📈 Model Bilgisini Görüntüleme

**Model Info** butonuna tıklayın:
```
Architecture:
  • Type: Multi-layer (Stacked/Deep RNN)
  • Number of Layers: 3
  • Layer Sizes: 50 → 30 → 20
  • Total Parameters: 8,540

Layer Details:
  • Layer 1: 50 hidden units
  • Layer 2: 30 hidden units
  • Layer 3: 20 hidden units
```

## 🔧 Parametre Hesaplama

**Tek Katman** (20 nöron):
```
Wxh: 1 × 20 = 20
Whh: 20 × 20 = 400
bh: 20
Why: 20 × 1 = 20
by: 1
Toplam: ~461 parametre
```

**İki Katman** (30→20):
```
Layer 1:
  Wxh: 1 × 30 = 30
  Whh: 30 × 30 = 900
  bh: 30

Layer 2:
  Wxh: 30 × 20 = 600
  Whh: 20 × 20 = 400
  bh: 20

Output:
  Why: 20 × 1 = 20
  by: 1

Toplam: ~2,001 parametre
```

**3 kat daha fazla parametre** = Daha fazla öğrenme kapasitesi!

## 🚀 Sonuç

**Evet**, programınızda artık:
- ✅ 1-5 arası hidden layer tanımlayabilirsiniz
- ✅ Her layer'ın nöron sayısını ayrı ayrı belirleyebilirsiniz
- ✅ Farklı mimariler deneyebilirsiniz
- ✅ Daha karmaşık desenleri öğrenebilirsiniz

**Mutlu kodlamalar! 🎉**
