# Deep Learning Education Platform / Derin Öğrenme Eğitim Platformu

Bu repository, TensorFlow ve Keras kullanarak derin öğrenme modellerini eğitmek için tasarlanmış kapsamlı bir eğitim platformu içerir. Program hem sınıflandırma hem de regresyon görevlerini destekler ve Deep Learning öğrenmek isteyenler için interaktif eğitici araçlar sunar.

## 🎯 Eğitici Özellikler

### 📚 **Tutorial Modu**
- Sinir ağı temellerini adım adım açıklar
- Nöronlar, aktivasyon fonksiyonları ve eğitim süreci hakkında detaylı bilgi
- İnteraktif örnekler ve görsel açıklamalar

### 🔧 **Hiperparametre Optimizasyonu Rehberi**  
- Yeni başlayanlar için parametre önerileri
- Her parametrenin model üzerindeki etkilerinin açıklaması
- Otomatik parametre karşılaştırma testleri

### 📊 **Model Karşılaştırma Aracı**
- Farklı model mimarilerini karşılaştırma
- Overfitting, yakınsama hızı ve kararlılık analizi
- Görsel dashboard ile performans değerlendirme

### 🔍 **Veri Analizi ve Ön İşleme Rehberi**
- Otomatik veri kalitesi analizi
- Eksik değer ve aykırı değer tespiti
- Veri görselleştirme ve korelasyon analizi
- Kişiselleştirilmiş ön işleme kodu üretimi

## ⭐ Temel Özellikler

- **Sınıflandırma ve Regresyon Desteği**: Model ve kayıp fonksiyonunu veri setine göre otomatik ayarlar
- **Özelleştirilebilir Parametreler**: Katman sayısı, nöron sayısı, aktivasyon fonksiyonları ve daha fazlası
- **Erken Durdurma**: Validation loss iyileşmediğinde eğitimi otomatik durdurur
- **Veri Ön İşleme**: One-hot encoding, scaling ve train-test ayrımı
- **Görselleştirme**: Eğitim metrikleri için detaylı grafikler
- **Otomatik Kütüphane Yükleme**: Eksik kütüphaneleri otomatik tespit eder ve yükler

## 📦 Gereksinimler

✨ Program eksik kütüphaneleri otomatik olarak yükler. Manuel yükleme için aşağıdaki talimatları izleyebilirsiniz:

### Python Kütüphaneleri

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn pygame
```

### TensorFlow Sürümü

- GPU desteği için önerilen TensorFlow sürümü: `2.11` veya daha düşük

## 🚀 Kullanım

### 1. Veri Seti Hazırlama

Gerekli veri setlerinin script ile aynı dizinde bulunduğundan emin olun:
- `knn_purchase_history.csv` (sınıflandırma görevleri için)
- `california_housing.csv` (regresyon görevleri için)

### 2. Programı Çalıştırma

```bash
python "Deep Learning.py"
```

### 3. Menü Seçenekleri

Program çalıştırıldığında şu seçeneklerle karşılaşacaksınız:

1. **Model Eğitimi (Classification)** - Sınıflandırma modeli eğitimi
2. **Model Eğitimi (Regression)** - Regresyon modeli eğitimi  
3. **📚 Tutorial Modu** - Deep Learning kavramlarını öğrenin
4. **🔧 Hiperparametre Rehberi** - Parametre optimizasyonu öğrenin
5. **📊 Model Karşılaştırma** - Farklı modelleri karşılaştırın
6. **🔍 Veri Analizi Rehberi** - Veri ön işleme öğrenin
7. **⚙️ Gelişmiş Ayarlar** - Model parametrelerini özelleştirin
```

### 3. Selecting the Task

When prompted, select the task:
- `1` for Classification.
- `2` for Regression.

---

## Configuration

You can customize the following parameters in the script:

### Model Parameters
- `NEURON_COUNT`: Number of neurons per layer.
- `HIDDEN_LAYERS`: Number of hidden layers.
- `ACTIVATION`: Activation function (e.g., `relu`, `sigmoid`).
- `OPTIMIZER`: Optimizer (e.g., `adam`, `sgd`).
- `LOSS_ALGORITHM`: Loss function (e.g., `sparse_categorical_crossentropy`, `mean_squared_error`).
- `METRICS`: Metrics to evaluate (e.g., `accuracy`, `mean_absolute_error`).

### Training Parameters
- `EPOCHS`: Number of training epochs.
- `VALIDATION_SPLIT`: Fraction of data used for validation.
- `TEST_SPLIT`: Fraction of data used for testing.
- `LEARNING_RATE`: Learning rate for the optimizer.

### Early Stopping
- `EARLY_STOPPING_PATIENCE`: Number of epochs to wait for improvement.
- `EARLY_STOPPING_MIN_DELTA`: Minimum improvement required to reset patience.

---

## Output

### Training Metrics

The script displays the following metrics during training:
- Loss
- Accuracy (for classification)
- Mean Absolute Error (for regression)

### Graphs

After training, the script generates graphs for:
- Accuracy (or MAE) over epochs.
- Loss over epochs.

### Model Evaluation

The script evaluates the model on the test set and displays:
- Test loss.
- Test accuracy (or MAE).

### Predictions

The script demonstrates predictions using example data:
- For regression, it predicts house prices.
- For classification, it predicts purchase probabilities.

---

## Example Output

### Regression

```plaintext
✨ Predicted House Price: $325,000
```

### Classification

```plaintext
✨ Predicted Purchase Probability: 85.23%
```

---

## Notes

- Ensure the datasets are formatted correctly (e.g., no missing values).
- Modify the `ColumnTransformer` section for custom preprocessing.
- For GPU support, install the appropriate TensorFlow version and drivers.

---

## Author

Prepared by **Dr. Mustafa AFYONLUOĞLU**  
Date: September 2025  
[https://afyonluoglu.org/](https://afyonluoglu.org/)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.