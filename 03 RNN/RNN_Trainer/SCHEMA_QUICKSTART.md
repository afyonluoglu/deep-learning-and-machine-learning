# 🎨 Model Schema Özelliği - Hızlı Başlangıç

## Nedir?

Model Schema, RNN modelinizin yapısını **görsel olarak** gösteren bir özelliktir.

## Nasıl Kullanılır?

### 1️⃣ Model Oluştur
```
Model Parameters → Initialize Model
```

### 2️⃣ Şemayı Aç
```
Model Management → 📊 Model Schema
```

### 3️⃣ (Opsiyonel) Not Ekle
```
Alt kısımdaki metin kutusuna notlarınızı yazın:
"Best model! Loss: 0.023, LR: 0.01"
```

### 4️⃣ Kaydet
```
💾 Save Schema as PNG
→ outputs/model_schema_TARIH_SAAT.png
```

## Ne Gösterir?

### Görsel Şema:
```
INPUT → HIDDEN 1 → HIDDEN 2 → ... → OUTPUT
 (1)      (30)        (20)            (1)
```

### Detaylar:
- ✅ Her katmandaki nöron sayısı
- ✅ Toplam parametre sayısı
- ✅ Activation, Dropout, Optimizer
- ✅ Sequence Length, Learning Rate
- ✅ Recurrent bağlantılar (↻)

## Renk Kodları

- 🔵 **Mavi**: Input
- 🟢 **Yeşil**: Hidden Layers
- 🔴 **Kırmızı**: Output

## Örnek Kullanım

### Tek Katman:
```
INPUT(1) → HIDDEN(20) → OUTPUT(1)
Params: ~461
```

### İki Katman:
```
INPUT(1) → HIDDEN 1(30) → HIDDEN 2(20) → OUTPUT(1)
Params: ~2,001
```

### Üç Katman:
```
INPUT(1) → HIDDEN 1(50) → HIDDEN 2(30) → HIDDEN 3(20) → OUTPUT(1)
Params: ~4,571
```

## Not Örnekleri

**Performans:**
```
Final Loss: 0.0234
Best config: lr=0.01, dropout=0.2
Accuracy: 95%
```

**Karşılaştırma:**
```
Better than 2-layer (loss: 0.0345)
Training time: 2.5 min
Recommended for production
```

**Deney:**
```
Testing dropout effect
0.0 dropout → 0.0567 (overfitting)
0.3 dropout → 0.0289 (better!)
```

## İpuçları

### ✨ Kaydetme:
- Notlar şema ile birlikte kaydedilir
- 150 DPI yüksek kalite
- Beyaz arka plan (baskı için uygun)

### 📊 Karşılaştırma:
- Her konfigürasyonu kaydedin
- Notlarda performans yazın
- Görsel karşılaştırın

### 📝 Dokümantasyon:
- Raporlara ekleyin
- Sunumlarda kullanın
- Öğrencilere gösterin

## Kayıt Yeri

```
RNN_Trainer/
  └── outputs/
      ├── model_schema_20250101_143025.png
      └── model_schema_20250101_150312.png
```

## Hızlı Test

1. Model oluştur (2 katman, 30→20)
2. Schema aç
3. Not ekle: "Test model"
4. Kaydet
5. outputs/ klasörünü kontrol et ✅

---

**Detaylı bilgi için**: `MODEL_SCHEMA_GUIDE.md`

**İyi görselleştirmeler! 🎨✨**
