# 🤖 KNN Sınıflandırma Projesi

Bu proje, K-En Yakın Komşu (K-Nearest Neighbors - KNN) algoritmasını kullanarak müşteri satın alma davranışlarını tahmin etmek için geliştirilmiş bir makine öğrenmesi uygulamasıdır. Uygulamanın amacı, makine öğrenmesi konusuna giriş yapmak isteyenler için bir örnek uygulama sunmaktır.

**Hazırlayan:** Dr. Mustafa AFYONLUOĞLU - Eylül 2025  
**Web:** https://afyonluoglu.org/

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Dosya Yapısı](#-dosya-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Programların Detaylı Açıklaması](#-programların-detaylı-açıklaması)
- [Veri Seti](#-veri-seti)
- [Model Performansı](#-model-performansı)
- [Ekran Görüntüleri](#-ekran-görüntüleri)

---

## 🎯 Proje Hakkında

Bu proje iki ana bileşenden oluşmaktadır:

1. **Model Eğitimi (ML_01_knn_classification.py)**: Müşteri verilerini kullanarak KNN modelini eğitir ve kaydeder.
2. **Model Kullanımı (ML_02_inference.py)**: Eğitilmiş modeli kullanarak yeni müşteriler için satın alma tahminleri yapar.

### 🔍 Amaç

Müşterilerin demografik bilgileri (cinsiyet, yaş, maaş) ve ürün fiyatına göre satın alma kararlarını tahmin etmek.

---

## 🛠 Kullanılan Teknolojiler

- **Python 3.x**
- **pandas**: Veri manipülasyonu ve analizi
- **numpy**: Sayısal hesaplamalar
- **scikit-learn**: Makine öğrenmesi algoritmaları
  - `KNeighborsClassifier`: KNN algoritması
  - `StandardScaler`: Veri ölçeklendirme
  - `train_test_split`: Veri setini bölme
  - `accuracy_score`: Model doğruluğu hesaplama
- **pickle**: Model kaydetme ve yükleme

---

## 📁 Dosya Yapısı

```
01 Machine Learning/
│
├── ML_01_knn_classification.py    # Model eğitim programı
├── ML_02_inference.py              # Tahmin yapma programı
├── knn_purchase_history.csv        # Eğitim veri seti
├── knn_new_customers.csv           # Test için yeni müşteri verileri
├── knn_model.pickle                # Eğitilmiş KNN modeli (otomatik oluşur)
├── knn_scaler.pickle               # Scaler objesi (otomatik oluşur)
├── knn_model_predictions.csv       # Tahmin sonuçları (otomatik oluşur)
├── outputs/                        # Ekran görüntüleri klasörü
│   ├── screenshot_training.png
│   └── screenshot_inference.png
└── README.md                       # Bu dosya
```

---

## 💻 Kurulum

### Gereksinimler

```bash
pip install pandas numpy scikit-learn
```

### Adımlar

1. Gerekli Python kütüphanelerini yükleyin
2. Veri setlerinin (`knn_purchase_history.csv`, `knn_new_customers.csv`) klasörde olduğundan emin olun
3. Programları sırasıyla çalıştırın

---

## 🚀 Kullanım

### 1️⃣ Model Eğitimi

```bash
python ML_01_knn_classification.py
```

**Bu program:**
- ✅ `knn_purchase_history.csv` dosyasından veri okur
- ✅ Veriyi işler ve ölçeklendirir
- ✅ KNN modelini eğitir (%80 eğitim, %20 test)
- ✅ Model performansını değerlendirir
- ✅ Modeli `knn_model.pickle` olarak kaydeder
- ✅ Scaler'ı `knn_scaler.pickle` olarak kaydeder

### 2️⃣ Tahmin Yapma

```bash
python ML_02_inference.py
```

**Bu program:**
- ✅ Kayıtlı modeli ve scaler'ı yükler
- ✅ `knn_new_customers.csv` dosyasından yeni müşteri verilerini okur
- ✅ Veriler üzerinde tahmin yapar
- ✅ Sonuçları `knn_model_predictions.csv` dosyasına kaydeder
- ✅ Tek bir müşteri için örnek tahmin gösterir

---

## 📚 Programların Detaylı Açıklaması

### 🔵 ML_01_knn_classification.py

#### Veri Yükleme ve Keşif
```python
df = pd.read_csv(CSV_File_Path)
```
- Veri seti yüklenir ve boyutu görüntülenir
- İlk 5 satır incelenir

#### Veri Ön İşleme

**1. Cinsiyet Kodlaması (One-Hot Encoding)**
```python
gender_encoded = pd.get_dummies(df['Gender'], drop_first=True)
```
- `Female` ve `Male` değerleri → `0` ve `1` değerlerine dönüştürülür
- `drop_first=True` ile dummy variable tuzağı önlenir

**2. Özellik ve Hedef Ayrımı**
```python
x = df[['Male','Age','Salary','Price']].to_numpy()
y = df['Purchased'].to_numpy()
```
- **X (Özellikler)**: Cinsiyet, Yaş, Maaş, Fiyat
- **y (Hedef)**: Satın alma durumu (0: Hayır, 1: Evet)

**3. Veri Setini Bölme**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
- %80 eğitim, %20 test
- `random_state=42` → Tekrarlanabilir sonuçlar

**4. Standardizasyon**
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```
- Veriler ortalaması 0, standart sapması 1 olacak şekilde ölçeklendirilir
- KNN algoritması mesafe tabanlı olduğu için ölçeklendirme kritiktir

#### Model Eğitimi
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
```
- K=5 komşu kullanılarak model eğitilir
- Her tahmin için en yakın 5 komşuya bakılır

#### Model Değerlendirme
```python
accuracy = accuracy_score(y_test, y_pred)
```
- Test seti üzerinde doğruluk skoru hesaplanır

#### Model Kaydetme
```python
pickle.dump(knn, f)
pickle.dump(scaler, f)
```
- Model ve scaler gelecekte kullanılmak üzere kaydedilir

---

### 🟢 ML_02_inference.py

#### Model ve Scaler Yükleme
```python
with open('knn_model.pickle', 'rb') as f:
    knn_new = pickle.load(f)
```
- Önceden eğitilmiş model ve scaler yüklenir

#### Yeni Veri İşleme

**1. Veri Okuma**
```python
new_df = pd.read_csv("knn_new_customers.csv")
```

**2. Cinsiyet Kodlaması**
```python
gender_encoded_new = pd.get_dummies(new_df['Gender'], drop_first=True)
```
- Eğitim verisiyle aynı şekilde işlenir

**3. Ölçeklendirme**
```python
x_new_scale2 = scaler_new.transform(x_new)
```
- ⚠️ **ÖNEMLİ**: `fit_transform` DEĞİL, sadece `transform` kullanılır
- Eğitim verisinin istatistikleri kullanılır

#### Toplu Tahmin
```python
y_new_pred = knn_new.predict(x_new_scale2)
df_new_2['will_purchase'] = y_new_pred
```
- Tüm yeni müşteriler için tahmin yapılır
- Sonuçlar CSV'ye kaydedilir

#### Tekil Tahmin Örneği
```python
row_values = [1, 32, 40000, 5000]  # Male, Age, Salary, Price
new_data_scaled = scaler_new.transform(new_data.to_numpy())
new_prediction = knn_new.predict(new_data_scaled)
```
- Tek bir müşteri için örnek tahmin gösterilir

---

## 📊 Veri Seti

### Eğitim Verisi (knn_purchase_history.csv)

| Sütun     | Açıklama                      | Değerler         |
|-----------|-------------------------------|------------------|
| Gender    | Cinsiyet                      | Male/Female      |
| Age       | Yaş                           | 18-65            |
| Salary    | Yıllık maaş                   | 15000-150000     |
| Price     | Ürün fiyatı                   | 1000-10000       |
| Purchased | Satın alma durumu (hedef)     | 0 (Hayır), 1 (Evet) |

### Yeni Müşteriler (knn_new_customers.csv)

Aynı yapıda ancak `Purchased` sütunu olmayan veri seti.

---

## 📈 Model Performansı

Model performansı, test seti üzerinde **accuracy_score** ile değerlendirilir:

```
Accuracy = (Doğru Tahminler) / (Toplam Tahminler)
```

Tipik olarak %85-95 arası doğruluk oranı elde edilir.

### 🎯 Model Hiperparametreleri

- **n_neighbors**: 5 (en yakın 5 komşu)
- **metric**: Euclidean distance (varsayılan)
- **weights**: uniform (tüm komşular eşit ağırlıklı)

---

## ⚙️ Teknik Notlar

### StandardScaler Kullanımı

```python
# ✅ DOĞRU (Eğitim verisi)
x_train = scaler.fit_transform(x_train)

# ✅ DOĞRU (Test/Yeni verisi)
x_test = scaler.transform(x_test)

# ❌ YANLIŞ (Yeni veriye fit_transform kullanmak)
x_test = scaler.fit_transform(x_test)  # Veri sızıntısına neden olur!
```

### One-Hot Encoding

```python
# drop_first=True kullanımı:
# Female, Male → Male (0: Female, 1: Male)

# drop_first=False kullanımı:
# Female, Male → Female, Male (2 sütun)
```

### Pickle ile Model Saklama

```python
# Kaydetme
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Yükleme
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
```

---

## 🔧 Hata Ayıklama

### Yaygın Hatalar ve Çözümleri

**1. FileNotFoundError**
```
❌ Model dosyası bulunamadı
✅ Önce ML_01_knn_classification.py programını çalıştırın
```

**2. Sütun Uyumsuzluğu**
```
❌ Gerekli sütunlar eksik
✅ Yeni veri setinin aynı sütunlara sahip olduğundan emin olun
```

**3. Ölçeklendirme Hatası**
```
❌ fit_transform kullanıldı
✅ Yeni veriler için sadece transform kullanın
```

---

## 📝 Sonuç

Bu proje, KNN algoritmasını kullanarak:
- ✅ Veri ön işleme tekniklerini gösterir
- ✅ Model eğitim ve değerlendirme sürecini açıklar
- ✅ Model deployment (kullanıma alma) pratiğini öğretir
- ✅ Gerçek dünya senaryolarında kullanılabilir bir çözüm sunar

---

## 📞 İletişim

**Dr. Mustafa AFYONLUOĞLU**  
Web: https://afyonluoglu.org/

---

## 📄 Lisans

Bu proje eğitim amaçlı hazırlanmıştır.

---

*Son Güncelleme: Eylül 2025*
