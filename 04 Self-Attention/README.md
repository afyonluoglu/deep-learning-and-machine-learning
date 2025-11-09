# 🎓 Self-Attention Öğrenme Aracı

Yapay sinir ağlarında **Self-Attention** mekanizmasını interaktif olarak öğrenmek için profesyonel bir eğitim aracı.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Kullanım Kılavuzu](#-kullanım-kılavuzu)
- [Self-Attention Nedir?](#-self-attention-nedir)
- [Parametreler](#️-parametreler)
- [Görselleştirmeler](#-görselleştirmeler)
- [Örnekler](#-örnekler)
- [Model Yönetimi](#-model-yönetimi)
- [Proje Yapısı](#-proje-yapısı)
- [Sorun Giderme](#-sorun-giderme)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## ✨ Özellikler

### 🎯 Eğitim Özellikleri
- ✅ **İnteraktif Öğrenme**: Query, Key, Value kavramlarını örneklerle öğrenin
- ✅ **Gerçek Zamanlı Eğitim**: Modeli canlı olarak eğitin ve sonuçları anında görün
- ✅ **Parametre Deneyleri**: Parametreleri değiştirerek etkilerini gözlemleyin
- ✅ **Multi-Head Attention**: Birden fazla attention head'in gücünü keşfedin

### 📊 Görselleştirme
- ✅ **Attention Map**: Token'lar arası ilişkileri ısı haritası olarak görün
- ✅ **Q, K, V Matrisleri**: Query, Key, Value matrislerini görselleştirin
- ✅ **Eğitim Grafikleri**: Loss değişimini gerçek zamanlı takip edin
- ✅ **Profesyonel Grafikler**: Yüksek çözünürlüklü, karanlık tema grafikleri

### 💾 Model Yönetimi
- ✅ **Model Kaydetme**: Eğitilmiş modelleri tüm parametreleriyle kaydedin
- ✅ **Model Yükleme**: Kaydedilmiş modelleri tekrar kullanın
- ✅ **Konfigürasyon Saklama**: Tüm ayarlar otomatik kaydedilir
- ✅ **Zaman Damgası**: Her model için benzersiz versiyon takibi

### 🎨 Kullanıcı Arayüzü
- ✅ **Modern Tasarım**: CustomTkinter ile modern, koyu tema arayüz
- ✅ **Kullanıcı Dostu**: Sezgisel kontroller ve açıklayıcı etiketler
- ✅ **HTML Yardım**: Detaylı, örnekli HTML yardım dökümanı
- ✅ **Türkçe Destek**: Tam Türkçe arayüz ve dökümanlar

---

## 🚀 Kurulum

### Gereksinimler

```bash
Python 3.8+
```

### Gerekli Kütüphaneler

```bash
pip install torch torchvision
pip install customtkinter
pip install matplotlib
pip install seaborn
pip install numpy
```

### Hızlı Kurulum

```bash
# Repoyu klonlayın
git clone <repo-url>
cd "04 Self-Attention"

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### requirements.txt

Program ile birlikte aşağıdaki `requirements.txt` dosyası oluşturulmuştur:

```txt
torch>=2.0.0
customtkinter>=5.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
numpy>=1.21.0
```

---

## 🎯 Hızlı Başlangıç

### 1. Programı Çalıştırma

```bash
python main.py
```

### 2. İlk Deneyiniz

1. **Örnek Veri**: "Kelime Dizisi" seçin (varsayılan)
2. **Parametreler**: Varsayılan değerleri kullanın
3. **Eğitim**: "🚀 Eğitimi Başlat" butonuna tıklayın
4. **Sonuçlar**: Görselleştirmeleri inceleyin

### 3. İlk Deneyiniz

```
Veri: Ben, Bugün, Okula, Gittim
Parametreler: d_model=64, num_heads=4, dropout=0.1
Epoch: 50
```

**Gözlem**: "Gittim" kelimesinin "Ben" ve "Okula" ile güçlü attention bağlantıları kurduğunu göreceksiniz!

---

## 📖 Kullanım Kılavuzu

### Adım 1: Veri Hazırlama

```
📊 Örnek Veri Seti seçeneğinden birini seçin:
- Kelime Dizisi
- Cümle Analizi  
- Zaman Serisi
- Görüntü Parçaları
- Özel Veri (kendi verinizi girin)
```

**Not**: Her satır bir token'ı temsil eder.

### Adım 2: Parametre Ayarlama

```
⚙️ Self-Attention Parametreleri:
- Embedding Boyutu (d_model): 32-512
- Attention Head Sayısı: 1-16
- Dropout Oranı: 0.0-0.5
- Öğrenme Hızı: 0.0001-0.01
```

### Adım 3: Eğitim

```
🎯 Eğitim Kontrolleri:
- Epoch Sayısı: 20-200 (önerilen: 50)
- Batch Size: 4-16 (önerilen: 8)
- "🚀 Eğitimi Başlat" butonuna tıklayın
```

### Adım 4: Sonuçları İnceleme

```
Tablar:
1. 🔍 Attention Map - Token'lar arası ilişkiler
2. 📊 Q, K, V Matrisleri - Vektör temsilleri
3. 📈 Eğitim Grafiği - Loss değişimi
4. 💡 Açıklama - Detaylı bilgiler
```

---

## 🧠 Self-Attention Nedir?

### Temel Kavram

Self-Attention, bir dizideki her elemanın diğer tüm elemanlarla **ilişkisini öğrenen** güçlü bir mekanizmadır.

### Ana Bileşenler

#### 1. Query (Q) - "Neyi arıyorum?"
```
Her token, Query vektörü ile diğer token'lardan 
ne tür bilgi istediğini belirtir.
```

#### 2. Key (K) - "Ben neyim?"
```
Her token, Key vektörü ile kendini tanıtır ve 
diğer token'ların sorgularına cevap verir.
```

#### 3. Value (V) - "Ne bilgi taşıyorum?"
```
Attention hesaplandıktan sonra aktarılacak 
gerçek bilgiyi içerir.
```

### Matematiksel Formül

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Adımlar**:
1. Q, K, V hesapla: `Q = X × W_q`, `K = X × W_k`, `V = X × W_v`
2. Skorları hesapla: `Scores = QK^T / √d_k`
3. Softmax uygula: `Weights = softmax(Scores)`
4. Value'ları ağırlıklandır: `Output = Weights × V`

### Multi-Head Attention

```
Birden fazla attention "head" kullanarak 
farklı ilişki türlerini paralel olarak öğrenme
```

**Avantajlar**:
- Her head farklı bir pattern öğrenir
- Daha zengin temsiller
- Daha iyi performans

---

## ⚙️ Parametreler

### d_model (Embedding Boyutu)

| Değer | Açıklama | Kullanım |
|-------|----------|----------|
| 32-64 | Küçük, hızlı | Deneyler için |
| 128-256 | Orta, dengeli | Çoğu uygulama |
| 512+ | Büyük, zengin | Karmaşık problemler |

**Etki**: ↑ = Daha zengin temsil, ↓ = Daha hızlı

### num_heads (Head Sayısı)

| Değer | Açıklama | d_model Uyumu |
|-------|----------|---------------|
| 1-2 | Az head | Basit ilişkiler |
| 4-8 | Orta | Dengeli |
| 12-16 | Çok head | Karmaşık ilişkiler |

**Önemli**: `d_model % num_heads == 0` olmalı!

### dropout

| Değer | Açıklama | Kullanım |
|-------|----------|----------|
| 0.0 | Dropout yok | Küçük veri |
| 0.1-0.2 | Hafif | Standart |
| 0.3-0.5 | Güçlü | Overfitting varsa |

**Etki**: Overfitting'i önler

### learning_rate

| Değer | Açıklama | Durum |
|-------|----------|-------|
| 0.0001 | Çok yavaş | Stabil eğitim |
| 0.001 | Orta | Standart |
| 0.01 | Hızlı | Dikkatli kullanın |

**Etki**: Öğrenme hızı ve kararlılık dengesi

---

## 📊 Görselleştirmeler

### 1. 🔍 Attention Map

```
Isı haritası formatında attention ağırlıkları

Satırlar: Query token'ları
Sütunlar: Key token'ları
Renkler: İlişki gücü (0-1)

Parlak = Güçlü ilişki
Koyu = Zayıf ilişki
```

**Örnek Yorum**:
```
"Oturdu" satırında "Kedi" ve "Mat" sütunları parlaksa,
"oturdu" kelimesi bu kelimelere güçlü attention veriyor.
```

### 2. 📊 Q, K, V Matrisleri

```
Her token için Q, K, V vektörlerinin görselleştirilmesi

Her satır: Bir token
Renkler: Vektör değerleri
- Kırmızı: Pozitif
- Mavi: Negatif  
- Beyaz: Sıfıra yakın
```

### 3. 📈 Eğitim Grafiği

```
Loss değişiminin epoch'lara göre grafiği

X Ekseni: Epoch sayısı
Y Ekseni: Loss değeri

İdeal: Azalan trend
Dikkat: Plato = durma, artış = overfitting
```

**Tüm grafikler `outputs/` klasörüne kaydedilir!**

---

## 📚 Örnekler

### Örnek 1: Basit Cümle

```python
Veri:
Kedi
Mat
Üzerinde
Oturdu

Parametreler:
- d_model: 64
- num_heads: 4
- epochs: 50

Beklenen: "Oturdu" -> "Kedi", "Mat" güçlü bağlantı
```

### Örnek 2: Zaman Serisi

```python
Veri:
Pazartesi
Salı
Çarşamba
Perşembe
Cuma

Parametreler:
- d_model: 128
- num_heads: 8
- epochs: 100

Beklenen: Ardışık günler arası güçlü bağlantı
```

### Örnek 3: Parametre Karşılaştırma

```python
Deney 1:
- num_heads: 2
- Model kaydet: "az_head"

Deney 2:
- num_heads: 8
- Model kaydet: "cok_head"

Karşılaştır: Attention map farkları
```

### Örnek 4: Dropout Etkisi

```python
Deney 1:
- dropout: 0.0
- epochs: 100
- Gözlem: Overfitting olabilir

Deney 2:
- dropout: 0.3
- epochs: 100
- Gözlem: Daha genel model
```

---

## 💾 Model Yönetimi

### Model Kaydetme

```python
1. Eğitimi tamamlayın
2. Model adı girin: "deneme_model"
3. "💾 Modeli Kaydet" tıklayın
4. Otomatik kaydedilir: "deneme_model_20250102_143052"
```

**Kaydedilen Bilgiler**:
- Model ağırlıkları
- Tüm parametreler
- Vocabulary
- Eğitim geçmişi
- Zaman damgası

### Model Yükleme

```python
1. "📂 Model Yükle" tıklayın
2. Listeden model seçin
3. Model tüm ayarlarıyla yüklenir
4. Hemen kullanıma hazır!
```

### Dosya Yapısı

```
models/
└── deneme_model_20250102_143052/
    ├── model_weights.pth      # PyTorch ağırlıkları
    ├── full_model.pth         # Tam model
    ├── config.json            # Parametreler
    └── model_info.json        # Meta bilgiler
```

### Model Karşılaştırma

```python
# Farklı konfigürasyonları kaydedin
Model 1: d_model=64, num_heads=4
Model 2: d_model=128, num_heads=8
Model 3: d_model=64, num_heads=2

# Sırayla yükleyip sonuçları karşılaştırın
```

---

## 📁 Proje Yapısı

```
04 Self-Attention/
│
├── main.py                      # Ana program
├── self_attention_module.py     # Self-Attention implementasyonu
├── visualization_module.py      # Görselleştirme fonksiyonları
├── model_manager.py             # Model yönetimi
├── help.html                    # HTML yardım dosyası
├── README.md                    # Bu dosya
├── requirements.txt             # Python bağımlılıkları
│
├── outputs/                     # Çıktı dosyaları
│   ├── attention_map.png
│   ├── qkv_matrices.png
│   └── training_history.png
│
└── models/                      # Kaydedilmiş modeller
    └── model_name_timestamp/
        ├── model_weights.pth
        ├── full_model.pth
        ├── config.json
        └── model_info.json
```

### Dosya Açıklamaları

| Dosya | Açıklama |
|-------|----------|
| `main.py` | Ana uygulama ve GUI |
| `self_attention_module.py` | Self-Attention katmanları ve eğitim |
| `visualization_module.py` | Matplotlib grafikleri |
| `model_manager.py` | Model kaydetme/yükleme |
| `help.html` | Detaylı kullanım kılavuzu |

---

## 🐛 Sorun Giderme

### Yaygın Hatalar

#### 1. "d_model must be divisible by num_heads"

```
Problem: d_model, num_heads'e tam bölünmüyor
Çözüm: d_model = num_heads × k (k bir tam sayı)
Örnek: d_model=64, num_heads=4 ✓
       d_model=64, num_heads=5 ✗
```

#### 2. Loss NaN Oldu

```
Problem: Learning rate çok yüksek
Çözüm: learning_rate'i düşürün (örn: 0.0001)
```

#### 3. Loss Düşmüyor

```
Problem: Learning rate çok düşük
Çözüm: learning_rate'i artırın (örn: 0.001)
```

#### 4. Out of Memory

```
Problem: Batch size veya model çok büyük
Çözüm: 
- batch_size'ı küçültün
- d_model'i azaltın
- num_heads'i azaltın
```

#### 5. Çok Yavaş Eğitim

```
Problem: Parametreler çok büyük
Çözüm:
- d_model'i düşürün (256 → 128)
- num_heads'i azaltın (8 → 4)
- epochs'u azaltın
```

### GPU Kullanımı

```python
# Program otomatik olarak GPU kullanır (varsa)
# Kontrol için:
import torch
print(torch.cuda.is_available())  # True = GPU var
print(torch.cuda.get_device_name(0))  # GPU adı
```

### Grafikler Görünmüyor

```
Problem: Matplotlib backend sorunu
Çözüm: Python'u yönetici olarak çalıştırın
```

---

## 💡 İpuçları ve En İyi Pratikler

### Öğrenme Stratejisi

1. **Hafta 1**: Temel kavramları anlayın
   - Q, K, V nedir?
   - Attention nasıl hesaplanır?

2. **Hafta 2**: Basit örneklerle pratik
   - Örnek veri setlerini deneyin
   - Attention map'leri inceleyin

3. **Hafta 3**: Parametre deneyleri
   - Tek tek parametreleri değiştirin
   - Etkileri gözlemleyin

4. **Hafta 4**: Kendi verileriniz
   - Özel veri setleri oluşturun
   - Gerçek problemlere uygulayın

### Deney Yapma Teknikleri

#### Tek Değişken Kuralı
```
Her seferinde sadece BİR parametreyi değiştirin
Örnek: İlk num_heads=4, sonra num_heads=8
```

#### Kayıt Tutma
```
Her deneyden sonra:
1. Modeli kaydedin (açıklayıcı isimle)
2. Ekran görüntüsü alın
3. Sonuçları not edin
```

#### Karşılaştırma
```
1. Baseline oluşturun (standart parametreler)
2. Her değişikliği baseline ile karşılaştırın
3. En iyi sonucu not edin
```

### Parametre Seçimi Rehberi

```python
# Küçük Veri (< 10 token)
d_model = 64
num_heads = 4
epochs = 50

# Orta Veri (10-50 token)
d_model = 128
num_heads = 8
epochs = 100

# Büyük Veri (> 50 token)
d_model = 256
num_heads = 16
epochs = 200
```

---

## 🎓 Ek Kaynaklar

### Önemli Makaleler

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Self-Attention'ın tanıtıldığı orijinal makale

2. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
   - BERT modelinde Self-Attention kullanımı

3. **"Language Models are Few-Shot Learners"** (Brown et al., 2020)
   - GPT-3 ve büyük dil modelleri

### Online Kaynaklar

- 📝 **The Illustrated Transformer** - Jay Alammar
- 🎥 **Attention Mechanism** - StatQuest
- 📚 **CS224N** - Stanford NLP Dersleri
- 🌐 **Hugging Face Tutorials** - Transformer'lar

### Kitaplar

- 📖 "Deep Learning" - Ian Goodfellow
- 📖 "Natural Language Processing with Transformers" - Lewis Tunstall
- 📖 "Dive into Deep Learning" - Aston Zhang

---

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak isterseniz:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

### Katkı Alanları

- 🐛 Bug düzeltmeleri
- ✨ Yeni özellikler
- 📝 Dokümantasyon iyileştirmeleri
- 🌍 Çeviri (İngilizce, vb.)
- 🎨 UI/UX iyileştirmeleri

---

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 📧 İletişim

Sorularınız veya önerileriniz için:

- 📧 Email: [your-email]
- 🐛 Issues: [GitHub Issues]
- 💬 Discussions: [GitHub Discussions]

---

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projeleri kullanır:

- **PyTorch** - Deep Learning framework
- **CustomTkinter** - Modern GUI toolkit
- **Matplotlib** - Görselleştirme
- **Seaborn** - İstatistiksel grafikler
- **NumPy** - Sayısal hesaplamalar

---


<div align="center">

### 🎓 İyi Öğrenmeler!

**Self-Attention mekanizmasını anlamak, modern yapay zeka sistemlerini anlamanın temelidir.**

Bol bol deney yapın ve keşfedin! 🚀

---

Made with ❤️ for AI Education

</div>
