# 🧠 RNN EĞİTİM PAKETİ - KULLANIM REHBERİ

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Kurulum](#kurulum) 
3. [Hızlı Başlangıç](#hızlı-başlangıç)
4. [Dosya Yapısı](#dosya-yapısı)
5. [Eğitim Programı](#eğitim-programı)
6. [Örnekler](#örnekler)
7. [Sorun Giderme](#sorun-giderme)
8. [İleri Düzey](#ileri-düzey)

## 🔍 Genel Bakış

Bu RNN Eğitim Paketi, Recurrent Neural Networks'ü sıfırdan öğrenmek isteyenler için kapsamlı bir eğitim programıdır. Teorik bilgilerden pratik uygulamalara kadar her şeyi içerir.

### ✨ Özellikler
- 📚 **Adım adım öğretim** - Hiç deneyim gerektirmez
- 🎨 **Zengin görselleştirmeler** - Kavramları görsel olarak anlayın
- 🛠️ **Pratik projeler** - Gerçek dünya örnekleri
- 🔄 **İnteraktif deneyim** - Kendi hızınızda öğrenin
- 🎯 **Farklı zorluk seviyeleri** - Başlangıçtan ileri düzeye

### 🎓 Kimin İçin?
- Machine Learning öğrenenler
- Veri bilimciler
- Python geliştiricileri
- Akademik araştırmacılar
- AI meraklıları

## 🔧 Kurulum

### 1. Sistem Gereksinimleri
- **Python**: 3.8 veya üzeri
- **RAM**: En az 4GB (8GB önerilen)
- **Disk**: ~2GB boş alan

### 2. Otomatik Kurulum (Önerilen)
```bash
python quick_start.py
```
Program gerekli tüm paketleri otomatik kuracaktır.

### 3. Manuel Kurulum
```bash
pip install -r requirements.txt
```

### 4. Kurulum Doğrulama
```bash
python setup.py
```

## 🚀 Hızlı Başlangıç

### Seçenek 1: Hızlı Demo (5 dakika)
```bash
python quick_start.py
# Menüden "1" seçin
```

### Seçenek 2: Tam Eğitim (45 dakika)
```bash
python main_educational_rnn.py
```

### Seçenek 3: Belirli Konular
```bash
python 01_rnn_theory.py          # Teori
python 02_rnn_basic_example.py   # Basit örnek
python 05_lstm_example.py        # LSTM
```

## 📁 Dosya Yapısı

```
RNN_Educational_Package/
├── 📖 README.md                    # Ana rehber
├── 🔧 requirements.txt            # Gerekli paketler
├── ⚡ quick_start.py              # Hızlı başlangıç
├── 🎯 main_educational_rnn.py     # Ana eğitim programı
├── 🔧 setup.py                    # Kurulum kontrolü
│
├── 📚 Temel Kavramlar/
│   ├── 01_rnn_theory.py           # RNN teorisi
│   ├── 02_rnn_basic_example.py    # Basit örnek
│   └── 03_rnn_visualization.py    # Görselleştirmeler
│
├── 🤖 RNN Türleri/
│   ├── 04_vanilla_rnn.py          # Vanilla RNN
│   ├── 05_lstm_example.py         # LSTM
│   └── 06_gru_example.py          # GRU
│
├── 🎮 Uygulamalar/
│   ├── 07_text_generation.py      # Metin üretimi
│   ├── 08_sentiment_analysis.py   # Duygu analizi
│   ├── 09_time_series_prediction.py # Zaman serisi
│   └── 10_stock_price_prediction.py # Borsa tahmini
│
├── 🛠️ utils/
│   └── helpers.py                  # Yardımcı fonksiyonlar
│
└── 📊 data/
    ├── sample_text.txt            # Örnek metin
    ├── temperature_data.txt       # Sıcaklık verisi
    └── stock_data.txt            # Hisse senedi verisi
```

## 📖 Eğitim Programı

### 🟢 Seviye 1: Temel Kavramlar (30 dakika)
1. **RNN Teorisi** (`01_rnn_theory.py`)
   - RNN nedir ve nasıl çalışır?
   - Sequential data ve temporal patterns
   - Vanishing gradient problemi

2. **İlk RNN Modeliniz** (`02_rnn_basic_example.py`)
   - TensorFlow/Keras ile RNN
   - Sıcaklık tahmini projesi
   - Model eğitimi ve değerlendirme

3. **Görselleştirmeler** (`03_rnn_visualization.py`)
   - RNN mimarisi diyagramları
   - Hidden state evrimi
   - Ağırlık paylaşımı

### 🟡 Seviye 2: İleri RNN Türleri (45 dakika)
4. **Vanilla RNN** (`04_vanilla_rnn.py`)
   - Detaylı implementasyon
   - Limitasyonlar ve çözümler

5. **LSTM** (`05_lstm_example.py`)
   - Gate mekanizmaları
   - Uzun vadeli hafıza
   - Hisse senedi tahmini

6. **GRU** (`06_gru_example.py`)
   - LSTM vs GRU karşılaştırması
   - Performans analizi

### 🔴 Seviye 3: Gerçek Dünya Uygulamaları (60 dakika)
7. **Metin Üretimi** (`07_text_generation.py`)
   - Character-level modeling
   - Temperature sampling
   - Yaratıcılık kontrolü

8. **Duygu Analizi** (`08_sentiment_analysis.py`)
   - NLP preprocessing
   - Sentiment classification
   - Model interpretability

9. **Zaman Serisi Tahmini** (`09_time_series_prediction.py`)
   - Multi-step forecasting
   - Seasonal patterns
   - Feature engineering

10. **Borsa Tahmini** (`10_stock_price_prediction.py`)
    - Financial data processing
    - Risk management
    - Portfolio optimization

## 💡 Örnekler

### Basit Sıcaklık Tahmini
```python
# 7 günlük geçmiş ile yarını tahmin et
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veri hazırla
data = create_temperature_data()
X, y = create_sequences(data, window_size=7)

# Model oluştur
model = Sequential([
    LSTM(50, input_shape=(7, 1)),
    Dense(1)
])

# Eğit ve tahmin et
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50)
prediction = model.predict(last_week_data)
```

### Metin Üretimi
```python
# Shakespeare tarzı metin üret
model = create_text_model()
generated_text = generate_text(
    model, 
    seed="To be or not to be",
    length=100,
    temperature=0.8
)
print(generated_text)
```

## 🔧 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### ❌ ImportError: No module named 'tensorflow'
**Çözüm:**
```bash
pip install tensorflow>=2.8.0
```

#### ❌ Memory Error
**Çözüm:**
- Batch size'ı küçültün (`batch_size=16`)
- Model boyutunu azaltın
- Veri miktarını kısıtlayın

#### ❌ Loss patlaması (NaN değerler)
**Çözüm:**
- Learning rate'i düşürün (`lr=0.0001`)
- Gradient clipping ekleyin
- Batch normalization kullanın

#### ❌ Yavaş eğitim
**Çözüm:**
- GPU kullanın
- Mixed precision training
- Model boyutunu optimize edin

### 🩺 Sistem Kontrolü
```bash
python -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
"
```

## 🚀 İleri Düzey

### Hiperparametre Optimizasyonu
```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', 32, 512, step=32),
        dropout=hp.Float('dropout', 0.0, 0.5, step=0.1)
    ))
    model.add(Dense(1))
    
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
        loss='mse'
    )
    return model

tuner = kt.RandomSearch(build_model, objective='val_loss')
tuner.search(X_train, y_train, validation_data=(X_val, y_val))
```

### Custom Loss Functions
```python
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    condition = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * tf.abs(error) - 0.5 * tf.square(delta)
    return tf.where(condition, squared_loss, linear_loss)

model.compile(optimizer='adam', loss=huber_loss)
```

### Model Ensemble
```python
# Birden fazla modeli birleştir
models = [create_lstm_model(), create_gru_model(), create_rnn_model()]
predictions = []

for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

# Ortalama al
ensemble_pred = np.mean(predictions, axis=0)
```

## 📚 Kaynaklar

### Önerilen Kitaplar
- 📖 "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- 📖 "Hands-On Machine Learning" - Aurélien Géron
- 📖 "Pattern Recognition and Machine Learning" - Christopher Bishop

### Online Kaynaklar
- 🌐 [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
- 🌐 [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- 🌐 [Papers With Code - RNN](https://paperswithcode.com/methods/category/recurrent-neural-networks)

### Veri Setleri
- 📊 [Time Series Data](https://www.kaggle.com/datasets?search=time+series)
- 📊 [Text Datasets](https://huggingface.co/datasets)
- 📊 [Financial Data](https://finance.yahoo.com)

## 🤝 Katkıda Bulunma

Bu proje açık kaynaklıdır. Katkılarınızı bekliyoruz!

### Nasıl Katkıda Bulunabilirsiniz?
1. 🐛 Bug raporları
2. 💡 Yeni özellik önerileri  
3. 📝 Dokümantasyon iyileştirmeleri
4. 🧪 Test case'leri
5. 🎓 Eğitim materyalleri

## 📞 Destek

Sorunlarınız için:
- 📧 Email: [email protected]
- 💬 GitHub Issues
- 📱 Discord: RNN Learning Community

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

---

**🎓 İyi öğrenmeler! RNN uzmanı olma yolculuğunuzda başarılar!**