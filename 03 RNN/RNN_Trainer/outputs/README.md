# Outputs Directory

Bu klasör, RNN Trainer uygulamasından kaydedilen grafikleri saklar.

## 📊 Kaydedilen Dosyalar

### Veri ve Tahmin Grafikleri
```
data_plot_YYYYMMDD_HHMMSS.png
```
- Mavi çizgi: Gerçek veri
- Kırmızı kesikli çizgi: Model tahminleri
- Alt kısımda eğitim parametreleri

### Loss Grafikleri
```
loss_plot_YYYYMMDD_HHMMSS.png
```
- Kırmızı çizgi: Training loss
- Logaritmik ölçek
- Alt kısımda eğitim parametreleri

## 💾 Dosya Formatı

Tüm grafikler şu bilgileri içerir:

- **Model Parametreleri**:
  - Hidden Units
  - Learning Rate
  - Sequence Length
  - Activation Function
  - Total Parameters

- **Veri Parametreleri**:
  - Wave Type
  - Samples
  - Frequency
  - Noise Level

- **Eğitim Bilgileri**:
  - Epochs Trained
  - Final Loss
  - Timestamp

## 🖼️ Grafik Özellikleri

- **Çözünürlük**: 150 DPI (yüksek kalite)
- **Boyut**: 12x8 inch
- **Format**: PNG
- **Parametre Kutusu**: Sol alt köşe

## 📝 İsimlendirme

Dosya isimleri otomatik oluşturulur:
```
[tip]_plot_[tarih]_[saat].png

Örnek:
data_plot_20250930_143025.png
loss_plot_20250930_143026.png
```

## 🎯 Kullanım

1. Model eğitin veya veri oluşturun
2. İlgili grafiğin yanındaki "💾 Save Graph" butonuna tıklayın
3. Grafik otomatik olarak bu klasöre kaydedilir
4. Dosya yolu popup mesajında gösterilir

## 🔍 Grafik İzleme Özellikleri

Her iki grafikte de:
- ✅ Zoom in/out (🔍 butonu)
- ✅ Pan (el butonu)
- ✅ Home (başa dön)
- ✅ Back/Forward (geri/ileri)
- ✅ Grid toggle

**Özellikle X ekseninde zoom yaparak detayları görebilirsiniz!**

## 💡 İpuçları

### Zoom Kullanımı
1. 🔍 butonuna tıklayın
2. Fare ile alan seçin (sol üstten sağ alta çizgi)
3. Seçilen alan büyütülür
4. 🏠 Home butonu ile başa dönün

### Pan Kullanımı
1. ✋ Pan butonuna tıklayın
2. Fareyle grafiği sürükleyin
3. Grafiği istediğiniz yere taşıyın

### En İyi Pratikler
- Önemli sonuçları hemen kaydedin
- Farklı parametrelerle karşılaştırma yapın
- Parametreleri dosya adına eklemek isterseniz yeniden adlandırın
- Düzenli olarak eski dosyaları temizleyin

## 📂 Organizasyon Önerisi

```
outputs/
├── sine_wave/
│   ├── data_plot_*.png
│   └── loss_plot_*.png
├── mixed_waves/
│   ├── data_plot_*.png
│   └── loss_plot_*.png
└── experiments/
    └── comparison_*.png
```

---

**Not**: Bu klasör otomatik oluşturulur. Grafik kaydetme özelliği v1.0.2'de eklendi.
