# Models Directory

Bu klasör eğitilmiş RNN modellerinizi saklamak içindir.

## 📦 Dosya Formatı

Her model kaydedildiğinde iki dosya oluşur:

```
model_ismi.pkl              # Model ağırlıkları ve parametreleri
model_ismi_config.json      # Normalizasyon ve veri bilgileri
```

## 💡 İsimlendirme Önerileri

### Veri Tipine Göre
```
sine_wave_model.pkl
mixed_waves_model.pkl
arma_model.pkl
exponential_model.pkl
```

### Parametrelere Göre
```
sine_h20_lr001_s20.pkl      # Hidden=20, LR=0.01, SeqLen=20
mixed_h40_lr008_s30.pkl     # Hidden=40, LR=0.008, SeqLen=30
```

### Versiyonlu
```
production_model_v1.pkl
production_model_v2.pkl
production_model_final.pkl
```

### Tarihli
```
model_2025_01_15.pkl
sine_model_20250115_1430.pkl
```

## 📊 Örnek Model Bilgileri

### Basit Sine Wave Model
```
Dosya: sine_basic.pkl
Hidden Units: 20
Learning Rate: 0.01
Sequence Length: 20
Training Epochs: 100
MSE: 0.018
```

### Karmaşık Mixed Waves Model
```
Dosya: mixed_advanced.pkl
Hidden Units: 40
Learning Rate: 0.008
Sequence Length: 30
Training Epochs: 200
MSE: 0.065
```

## 🔄 Model Yönetimi

### Yedekleme
Önemli modelleri düzenli olarak yedekleyin:
```
models/
  ├── production/
  │   ├── current_model.pkl
  │   └── current_model_config.json
  └── backup/
      ├── 2025_01_15/
      └── 2025_01_20/
```

### Temizlik
Kullanılmayan eski modelleri düzenli silin.

### Paylaşım
Model dosyalarını (.pkl + _config.json) birlikte paylaşın.

## ⚠️ Notlar

- Model dosyaları binary formatında (.pkl)
- Config dosyaları JSON formatında
- İkisi birlikte yedeklenmeli
- Toplam boyut genelde < 1 MB

---

**İyi eğitimler! 🚀**
