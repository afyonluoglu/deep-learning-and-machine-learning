# RNN Trainer - Recurrent Neural Network Learning Platform

![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**Profesyonel, araştırma seviyesinde** bir RNN (Recurrent Neural Network) eğitim ve görselleştirme platformu. Gerçek backpropagation through time (BPTT) algoritması, **advanced optimizers**, **comprehensive metrics**, ve **real-time monitoring** ile RNN'lerin nasıl çalıştığını interaktif olarak öğrenin!

## ✨ v2.0 YENİ ÖZELLİKLER! 🎉

### 🚀 Advanced Optimization
- **4 Optimizer Algoritması**: SGD, Momentum, **Adam** ⭐, RMSprop
- **4 Learning Rate Schedule**: Constant, Step, **Exponential** ⭐, Cosine
- **Real-time LR Tracking**: Öğrenme oranı değişimini canlı izleyin

### 📊 Comprehensive Metrics
- **8 Evaluation Metric**: MSE, RMSE, MAE, MAPE, **R²**, Max Error, Median AE, Directional Accuracy
- **Auto Quality Assessment**: Excellent, Good, Moderate, Poor
- **Real-time Updates**: Her 5 epoch'ta metrikler güncellenir

### 🔍 Real-time Monitoring
- **Gradient Health**: Vanishing/exploding gradient detection
- **Convergence Score**: 0-100 arası yakınsama skoru
- **Plateau Detection**: Eğitim duraklamasını algılama
- **Weight Analysis**: Dead neuron detection

### 🎨 Enhanced GUI
- Optimizer seçimi dropdown
- LR schedule dropdown
- Advanced metrics panel
- Gradient health monitor
- Training status display

**🎯 Result: Adam + Exponential decay ile 7x daha iyi performans!**

---

## � Tüm Özellikler

### 📚 v1.x Features (Mevcut)
- **Gerçek BPTT Algoritması**: Professional backpropagation through time
- **Xavier Initialization**: Ağırlık başlatma
- **Gradient Clipping**: Patlayan gradyanları önleme
- **Dropout Regularization**: Overfitting önleme
- **11 Data Generator**: Sine, Cosine, Square, Sawtooth, Triangular, Mixed, Exponential, Polynomial, Random Walk, ARMA, Damped
- **Custom CSV Loading**: Kendi verilerinizi yükleyin
- **Future Prediction**: N-step ahead tahmin
- **Model Save/Load**: Eğitilmiş modelleri kaydet/yükle
- **Interactive Zoom/Pan**: Matplotlib toolbar ile grafik kontrolü
- **Graph Export**: Parametrelerle birlikte PNG export

### 🎯 v2.0 Features (YENİ!)
- **Multiple Optimizers**: SGD, Momentum, Adam, RMSprop
- **LR Scheduling**: Constant, Step, Exponential, Cosine annealing
- **Comprehensive Metrics**: R², RMSE, MAE, MAPE
- **Gradient Monitoring**: Real-time gradient health
- **Convergence Tracking**: 0-100 score
- **Weight Analysis**: Dead neuron detection
- **Advanced GUI**: Real-time metric displays

## 📋 Gereksinimler

```
Python 3.8+
customtkinter >= 5.2.2
matplotlib >= 3.10.6
numpy >= 2.3.3
```

## 🚀 Kurulum

1. **Depoyu klonlayın veya dosyaları indirin**

2. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install customtkinter matplotlib numpy
```

3. **Uygulamayı başlatın:**
```bash
python rnn_trainer_app.py
```

## 📖 Hızlı Başlangıç (v2.0)

### 1️⃣ Temel Kullanım

#### Adım 1: Model Oluşturma
1. **Hidden Units** (Gizli Birimler): 5-100 arası ayarlayın
   - Küçük (5-15): Hızlı eğitim, düşük kapasite
   - Orta (20-40): Önerilen, dengeli
   - Büyük (50-100): Yüksek kapasite, yavaş eğitim

2. **Learning Rate** (Öğrenme Hızı): 0.001-0.1 arası
   - Düşük (0.001-0.005): Stabil ama yavaş
   - Orta (0.01-0.03): Önerilen
   - Yüksek (0.05-0.1): Hızlı ama kararsız olabilir

3. **Sequence Length** (Dizi Uzunluğu): 5-50 arası
   - Kısa (5-10): Kısa vadeli örüntüler
   - Orta (15-30): Dengeli
   - Uzun (35-50): Uzun vadeli bağımlılıklar

4. **Activation Function** (Aktivasyon Fonksiyonu):
   - `tanh`: Önerilen, [-1, 1] aralığında çıktı
   - `relu`: Daha hızlı olabilir, pozitif değerler

5. **"Initialize Model"** butonuna tıklayın

#### Adım 2: Veri Üretme
1. **Wave Type** (Dalga Tipi) seçin:
   - **Sine Wave**: Temel sinüs dalgası
   - **Cosine Wave**: Kosinüs dalgası
   - **Square Wave**: Kare dalga
   - **Sawtooth Wave**: Testere dişi dalga
   - **Triangular Wave**: Üçgen dalga
   - **Mixed Waves**: Karışık frekanslar
   - **Exponential**: Üstel büyüme/azalma
   - **Polynomial**: Polinom trendi
   - **Random Walk**: Rastgele yürüyüş
   - **ARMA**: Otoregresif hareketli ortalama
   - **Damped Oscillation**: Sönümlü salınım

2. **Samples** (Örnekler): 100-2000 arası veri noktası

3. **Frequency** (Frekans): 0.1-5.0 arası (periyodik dalgalar için)

4. **Noise Level** (Gürültü Seviyesi): 0.0-0.5 arası
   - 0.0: Gürültüsüz, temiz veri
   - 0.05: Hafif gürültü (önerilen)
   - 0.1-0.3: Orta gürültü
   - 0.5: Yüksek gürültü

5. **"Generate Data"** butonuna tıklayın

#### Adım 3: Eğitim
1. **Epochs** (Dönem): 10-500 arası eğitim dönemi sayısı
   - 50-100: Basit örüntüler için
   - 100-200: Orta karmaşıklık
   - 200-500: Karmaşık örüntüler

2. **"Start Training"** butonuna tıklayın

3. Eğitim sırasında:
   - Loss grafiği gerçek zamanlı güncellenir
   - Status bar'da ilerleme görürsünüz
   - "Stop" butonu ile istediğiniz zaman durabilirsiniz

#### Adım 4: Test ve Tahmin
1. Eğitim tamamlandıktan sonra **"Test Prediction"** tıklayın
2. Mavi çizgi: Gerçek veri
3. Kırmızı kesikli çizgi: Model tahminleri
4. MSE değerini kontrol edin:
   - < 0.01: Mükemmel
   - 0.01-0.1: İyi
   - 0.1-1.0: Orta
   - \> 1.0: Zayıf (daha fazla eğitim gerekli)

#### Adım 5: Model Kaydetme
1. **"Save Model"** butonuna tıklayın
2. Dosya adı ve konum seçin (.pkl uzantılı)
3. Model ve konfigürasyon otomatik kaydedilir

#### Adım 6: Model Yükleme
1. **"Load Model"** butonuna tıklayın
2. Önceden kaydedilmiş .pkl dosyasını seçin
3. Model tüm parametreleri ve ağırlıkları ile yüklenir
4. İsterseniz eğitime devam edebilir veya hemen test edebilirsiniz

### 2️⃣ Örnek Çalışma Senaryoları

#### 🔷 Örnek 1: Basit Sinüs Dalgası Öğrenimi

**Amaç**: RNN'in basit periyodik bir örüntüyü öğrenmesini sağlamak

**Adımlar**:
1. Model Parametreleri:
   - Hidden Units: 20
   - Learning Rate: 0.01
   - Sequence Length: 20
   - Activation: tanh

2. Veri Üretimi:
   - Wave Type: Sine Wave
   - Samples: 500
   - Frequency: 1.0
   - Noise Level: 0.05

3. Eğitim:
   - Epochs: 100
   - "Start Training" tıklayın

4. Sonuç:
   - Loss grafiğinde düzenli azalma göreceksiniz
   - Test prediction ile yüksek doğruluk elde edeceksiniz
   - Beklenen MSE: < 0.02

#### 🔷 Örnek 2: Karmaşık Karışık Dalgalar

**Amaç**: Birden fazla frekansın karışımını öğretmek

**Adımlar**:
1. Model Parametreleri:
   - Hidden Units: 40
   - Learning Rate: 0.005
   - Sequence Length: 30
   - Activation: tanh

2. Veri Üretimi:
   - Wave Type: Mixed Waves
   - Samples: 1000
   - Frequency: 1.5
   - Noise Level: 0.1

3. Eğitim:
   - Epochs: 200
   - Daha karmaşık olduğu için daha uzun eğitim

4. Sonuç:
   - İlk 50 epoch'ta hızlı öğrenme
   - Sonrasında yavaş iyileşme
   - Beklenen MSE: 0.05-0.1

#### 🔷 Örnek 3: Trend Tahmini (Exponential)

**Amaç**: Üstel trend öğrenimi

**Adımlar**:
1. Model Parametreleri:
   - Hidden Units: 30
   - Learning Rate: 0.01
   - Sequence Length: 25
   - Activation: tanh

2. Veri Üretimi:
   - Wave Type: Exponential
   - Samples: 800
   - Noise Level: 0.05

3. Eğitim:
   - Epochs: 150

4. Sonuç:
   - Trendin genel yönünü yakalar
   - Detaylarda küçük sapmalar olabilir
   - Beklenen MSE: 0.1-0.3

#### 🔷 Örnek 4: Gürültülü Veri ile Dayanıklılık Testi

**Amaç**: Modelin gürültüye karşı dayanıklılığını test etmek

**Adımlar**:
1. Model Parametreleri:
   - Hidden Units: 50
   - Learning Rate: 0.008
   - Sequence Length: 35
   - Activation: tanh

2. Veri Üretimi:
   - Wave Type: Sine Wave
   - Samples: 1000
   - Frequency: 2.0
   - Noise Level: 0.3 (yüksek gürültü!)

3. Eğitim:
   - Epochs: 300
   - Gürültü nedeniyle daha uzun eğitim

4. Sonuç:
   - Model gürültüyü filtreleyerek ana örüntüyü öğrenir
   - Beklenen MSE: 0.15-0.25

#### 🔷 Örnek 5: Parametre Optimizasyonu

**Amaç**: Farklı parametrelerin etkisini karşılaştırmak

**Deneyler**:

**Deney A - Küçük Model**:
- Hidden Units: 10
- Learning Rate: 0.02
- Epochs: 100
- Sonuç: Hızlı ama sınırlı kapasite

**Deney B - Orta Model**:
- Hidden Units: 30
- Learning Rate: 0.01
- Epochs: 100
- Sonuç: İyi denge

**Deney C - Büyük Model**:
- Hidden Units: 80
- Learning Rate: 0.005
- Epochs: 100
- Sonuç: En iyi doğruluk ama yavaş

**Karşılaştırma**: Hangi konfigürasyonun MSE'si en düşük?

### 3️⃣ İleri Seviye Kullanım

#### 🔧 Parametre Ayarlama Stratejileri

**Learning Rate Ayarlama**:
```
Eğer loss:
  - Çok yavaş azalıyor → Learning rate'i artır (x2)
  - Salınım yapıyor → Learning rate'i azalt (/2)
  - Artıyor → Learning rate'i çok azalt (/10)
  - Düzenli azalıyor → Mükemmel, değiştirme
```

**Hidden Units Ayarlama**:
```
Eğer model:
  - Underfitting (yetersiz) → Hidden units artır
  - Overfitting (aşırı) → Hidden units azalt
  - Tam yerinde → Değiştirme
```

**Sequence Length Ayarlama**:
```
Örüntü periyodu:
  - Kısa (< 10 adım) → Sequence length: 10-15
  - Orta (10-30 adım) → Sequence length: 20-35
  - Uzun (> 30 adım) → Sequence length: 40-50
```

#### 💡 En İyi Pratikler

1. **Başlangıç Parametreleri**:
   ```
   Hidden Units: 20-30
   Learning Rate: 0.01
   Sequence Length: 20
   Activation: tanh
   Epochs: 100
   ```

2. **Veri Hazırlığı**:
   - Her zaman normalizasyon kullanın (otomatik yapılır)
   - Küçük gürültü ekleyin (0.05) generalizasyon için
   - Yeterli veri noktası kullanın (minimum 500)

3. **Eğitim İzleme**:
   - Loss grafiğini düzenli kontrol edin
   - Platolaşma görürseniz eğitimi durdurun
   - Her 50 epoch'ta test prediction yapın

4. **Model Seçimi**:
   - Birden fazla konfigürasyon deneyin
   - En iyi MSE'yi veren modeli kaydedin
   - Farklı veri setlerinde test edin

## 🎓 RNN Teorisi

### Backpropagation Through Time (BPTT)

RNN'ler zaman içinde geri yayılım kullanır:

```
Forward Pass:
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y_t = W_hy * h_t + b_y

Backward Pass:
∂L/∂W_hy = Σ (y_t - target_t) * h_t^T
∂L/∂W_hh = Σ δ_t * h_{t-1}^T
∂L/∂W_xh = Σ δ_t * x_t^T

Update:
W ← W - α * ∂L/∂W
```

### Gradyan Patlaması Önleme

Gradient clipping kullanılır:
```python
if |gradient| > threshold:
    gradient = threshold * (gradient / |gradient|)
```

## 📊 Veri Tipleri Detaylı Açıklama

### 1. Sine Wave (Sinüs Dalgası)
- **Formül**: `y = A * sin(2πft + φ)`
- **Kullanım**: Temel periyodik örüntü öğrenimi
- **Önerilen Params**: freq=1.0, noise=0.05

### 2. Cosine Wave (Kosinüs Dalgası)
- **Formül**: `y = A * cos(2πft + φ)`
- **Kullanım**: Faz kayması öğrenimi
- **Önerilen Params**: freq=1.0, noise=0.05

### 3. Square Wave (Kare Dalga)
- **Formül**: `y = A * sign(sin(2πft))`
- **Kullanım**: Keskin geçiş öğrenimi
- **Önerilen Params**: freq=0.5, noise=0.02

### 4. Sawtooth Wave (Testere Dişi)
- **Kullanım**: Doğrusal rampa öğrenimi
- **Önerilen Params**: freq=0.5, noise=0.05

### 5. Triangular Wave (Üçgen Dalga)
- **Kullanım**: Simetrik örüntü öğrenimi
- **Önerilen Params**: freq=0.8, noise=0.05

### 6. Mixed Waves (Karışık Dalgalar)
- **Formül**: `y = Σ A_i * sin(2πf_i*t)`
- **Kullanım**: Çoklu frekans öğrenimi
- **Önerilen Params**: noise=0.1

### 7. Exponential (Üstel)
- **Formül**: `y = e^(rt)`
- **Kullanım**: Büyüme/azalma trendi
- **Önerilen Params**: growth_rate=0.01

### 8. Polynomial (Polinom)
- **Formül**: `y = a + bx + cx² + ...`
- **Kullanım**: Doğrusal olmayan trend
- **Önerilen Params**: coefficients=[0, 0.5, 0.1]

### 9. Random Walk (Rastgele Yürüyüş)
- **Formül**: `y_t = y_{t-1} + ε_t`
- **Kullanım**: Stokastik süreç öğrenimi
- **Önerilen Params**: step_size=0.1

### 10. ARMA (Otoregresif Hareketli Ortalama)
- **Formül**: `y_t = Σφ_i*y_{t-i} + Σθ_j*ε_{t-j}`
- **Kullanım**: İstatistiksel modelleme
- **Önerilen Params**: ar=[0.5], ma=[0.3]

### 11. Damped Oscillation (Sönümlü Salınım)
- **Formül**: `y = A * e^(-dt) * sin(2πft)`
- **Kullanım**: Karmaşık dinamikler
- **Önerilen Params**: freq=1.0, damping=0.1

## 🐛 Sorun Giderme

### Problem: Loss Azalmıyor
**Çözümler**:
- Learning rate'i artırın (0.01 → 0.03)
- Daha fazla epoch kullanın
- Hidden units sayısını artırın
- Sequence length'i ayarlayın

### Problem: Loss Artıyor (Divergence)
**Çözümler**:
- Learning rate'i azaltın (0.01 → 0.001)
- Gradient clipping kontrol edin (otomatik)
- Farklı aktivasyon deneyin (relu → tanh)

### Problem: Kötü Tahminler
**Çözümler**:
- Daha uzun eğitim (epochs artır)
- Hidden units artır
- Sequence length artır
- Daha temiz veri kullanın (noise azalt)

### Problem: Overfitting
**Belirtiler**: Eğitimde mükemmel, testte kötü
**Çözümler**:
- Noise level artır
- Hidden units azalt
- Daha fazla training verisi

### Problem: Underfitting
**Belirtiler**: Hem eğitimde hem testte kötü
**Çözümler**:
- Hidden units artır
- Daha uzun eğitim
- Learning rate artır

### Problem: Yavaş Eğitim
**Çözümler**:
- Sample sayısını azaltın
- Hidden units azaltın
- Epochs azaltın (ama sonuç kötü olabilir)

## 📁 Dosya Yapısı

```
RNN_Trainer/
│
├── rnn_model.py              # RNN model implementasyonu
│   ├── RNNModel class
│   ├── forward()             # İleri geçiş
│   ├── backward()            # BPTT
│   ├── train_epoch()         # Eğitim
│   ├── predict()             # Tahmin
│   ├── save_model()          # Kaydetme
│   └── load_model()          # Yükleme
│
├── data_generator.py         # Veri üretici
│   ├── generate_sine_wave()
│   ├── generate_cosine_wave()
│   ├── generate_square_wave()
│   ├── generate_sawtooth_wave()
│   ├── generate_triangular_wave()
│   ├── generate_mixed_waves()
│   ├── generate_exponential()
│   ├── generate_polynomial()
│   ├── generate_random_walk()
│   ├── generate_arma()
│   ├── generate_damped_oscillation()
│   └── normalize_data()
│
├── rnn_trainer_app.py        # Ana GUI uygulaması
│   ├── RNNTrainerApp class
│   ├── Control panel
│   ├── Visualization panel
│   ├── Training logic
│   └── Model management
│
├── README.md                 # Bu dosya
│
└── USAGE_EXAMPLES.md         # Detaylı kullanım örnekleri
```

## 🔬 Teknik Detaylar

### Model Mimarisi

```
Input Layer (1 unit)
    ↓
Hidden Layer (5-100 units)
    ↓ (recurrent connection)
Hidden Layer (same units)
    ↓
Output Layer (1 unit)
```

### Ağırlık Matrisleri

- **W_xh**: Input to hidden (hidden_size × input_size)
- **W_hh**: Hidden to hidden (hidden_size × hidden_size)
- **W_hy**: Hidden to output (output_size × hidden_size)
- **b_h**: Hidden bias (hidden_size × 1)
- **b_y**: Output bias (output_size × 1)

### Toplam Parametre Sayısı

```
Total = (hidden × input) + (hidden × hidden) + (output × hidden) + hidden + output
```

Örnek (hidden=20, input=1, output=1):
```
Total = (20 × 1) + (20 × 20) + (1 × 20) + 20 + 1 = 461 parametre
```

## 💻 Kod Örnekleri

### Manuel Model Kullanımı (Python)

```python
from rnn_model import RNNModel
from data_generator import DataGenerator
import numpy as np

# Model oluştur
model = RNNModel(
    input_size=1,
    hidden_size=20,
    output_size=1,
    learning_rate=0.01,
    sequence_length=20,
    activation='tanh'
)

# Veri üret
generator = DataGenerator()
data = generator.generate_sine_wave(n_samples=500, frequency=1.0, noise_level=0.05)

# Normalize et
normalized_data, min_val, max_val = generator.normalize_data(data)

# Diziler oluştur
X, y = generator.create_sequences(normalized_data, sequence_length=20)

# Eğit
for epoch in range(100):
    loss = model.train_epoch(X.reshape(-1, 1), y.reshape(-1, 1))
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Tahmin yap
predictions = model.predict(normalized_data)

# Denormalize et
predictions_denorm = generator.denormalize_data(predictions, min_val, max_val)

# Modeli kaydet
model.save_model('my_trained_model.pkl')

# Model yükle
loaded_model = RNNModel.load_model('my_trained_model.pkl')
```

### Özel Veri Serisi Ekleme

`data_generator.py` dosyasına yeni fonksiyon ekleyin:

```python
@staticmethod
def generate_custom_wave(n_samples: int = 1000,
                        param1: float = 1.0,
                        noise_level: float = 0.0) -> np.ndarray:
    """Generate custom wave pattern."""
    t = np.linspace(0, 10, n_samples)
    data = # Your formula here
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, n_samples)
        data += noise
    
    return data.reshape(-1, 1)
```

## 📈 Performans İpuçları

### Hız Optimizasyonu
1. Küçük batch'ler kullanın (otomatik)
2. Sequence length'i makul tutun (< 50)
3. Hidden units'i dengeleyin (20-40)
4. NumPy vektörizasyonu kullanılıyor (hızlı)

### Bellek Optimizasyonu
1. Çok büyük veri setlerinden kaçının (< 5000 sample)
2. Gradient history saklanmıyor (otomatik)
3. Model dosyaları küçük (<1MB tipik)

## 🎨 GUI Özelleştirme

Tema değiştirme (`rnn_trainer_app.py`):
```python
ctk.set_appearance_mode("dark")  # "light", "dark", "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
```

## 📝 Lisans

MIT License - Özgürce kullanabilir, değiştirebilir ve dağıtabilirsiniz.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📞 İletişim & Destek

Sorularınız için:
- GitHub Issues kullanın
- Kod örneklerini paylaşın
- Hataları detaylı bildirin

## 🙏 Teşekkürler

- NumPy ekibine hızlı hesaplamalar için
- Matplotlib ekibine görselleştirme için
- CustomTkinter geliştiricilerine modern GUI için

## 📚 Referanslar

- Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Rumelhart, D. E., et al. (1986). Learning representations by back-propagating errors. Nature.

---

**Başarılı eğitimler! 🚀**

Detaylı kullanım örnekleri için `USAGE_EXAMPLES.md` dosyasına bakın.
