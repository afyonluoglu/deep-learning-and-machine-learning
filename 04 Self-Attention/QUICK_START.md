# 🎯 Self-Attention Öğrenme Aracı - Hızlı Başlangıç Rehberi

## 🚀 Program Başlatma

```bash
cd "c:\Users\ASUS\Desktop\Python with AI\04 Self-Attention"
python main.py
```

---

## 📚 Örnek Çalışma Senaryoları

### Senaryo 1: İlk Denemeniz (5 dakika)

**Hedef**: Self-Attention'ın temel çalışma prensibini anlamak

1. **Programı Başlatın**
   - `python main.py` komutuyla başlatın

2. **Varsayılan Ayarları Kullanın**
   - Veri: "Ben, Bugün, Okula, Gittim" (otomatik yüklü)
   - d_model: 64
   - num_heads: 4
   - dropout: 0.1
   - learning_rate: 0.001

3. **Eğitimi Başlatın**
   - Epoch: 50
   - Batch Size: 8
   - "🚀 Eğitimi Başlat" butonuna tıklayın

4. **Sonuçları İnceleyin**
   - 🔍 **Attention Map** tabına gidin
   - "Gittim" satırına bakın
   - "Ben" ve "Okula" ile güçlü bağlantı görmelisiniz
   - Bu, "gittim" fiilinin özne ve yeri aradığını gösterir!

**Beklenen Süre**: ~2 dakika (eğitim + inceleme)

---

### Senaryo 2: Head Sayısının Etkisi (15 dakika)

**Hedef**: Multi-Head Attention'ın avantajlarını görmek

#### Deney 1: Az Head
```
Veri: Kedi, Mat, Üzerinde, Oturdu
d_model: 64
num_heads: 2
epochs: 50
Model Adı: "az_head_model"
```
- Eğitimi tamamlayın
- Attention map'i inceleyin
- Modeli kaydedin

#### Deney 2: Çok Head
```
Veri: Kedi, Mat, Üzerinde, Oturdu (AYNI VERİ!)
d_model: 64
num_heads: 8
epochs: 50
Model Adı: "cok_head_model"
```
- Eğitimi tamamlayın
- Attention map'i inceleyin
- Modeli kaydedin

#### Karşılaştırma
1. "az_head_model" yükleyin → Attention map ekran görüntüsü alın
2. "cok_head_model" yükleyin → Attention map ekran görüntüsü alın
3. İki grafiği yan yana koyun

**Gözlem**: Daha fazla head, daha detaylı ve çeşitli ilişkiler öğrenir!

---

### Senaryo 3: Embedding Boyutunun Etkisi (20 dakika)

**Hedef**: d_model parametresinin modelin kapasitesine etkisini görmek

#### Test 1: Küçük Model
```
Veri: Pazartesi, Salı, Çarşamba, Perşembe, Cuma, Cumartesi
d_model: 32
num_heads: 4
epochs: 100
```

#### Test 2: Orta Model
```
Veri: Pazartesi, Salı, Çarşamba, Perşembe, Cuma, Cumartesi (AYNI)
d_model: 128
num_heads: 4
epochs: 100
```

#### Test 3: Büyük Model
```
Veri: Pazartesi, Salı, Çarşamba, Perşembe, Cuma, Cumartesi (AYNI)
d_model: 256
num_heads: 4
epochs: 100
```

**Karşılaştırın**:
- 📈 Eğitim Grafiği tabında loss değerlerini karşılaştırın
- 📊 Q, K, V Matrisleri tabında vektörlerin zenginliğini gözlemleyin

**Gözlem**: Daha büyük d_model → Daha düşük loss ama daha yavaş eğitim!

---

### Senaryo 4: Dropout ve Overfitting (25 dakika)

**Hedef**: Dropout'un overfitting'i nasıl önlediğini görmek

#### Baseline (Dropout Yok)
```
Veri: Kedi, Köpek, Kuş, Balık
d_model: 64
num_heads: 4
dropout: 0.0
epochs: 200 (UZUN!)
Model: "no_dropout"
```

#### Dropout İle
```
Veri: Kedi, Köpek, Kuş, Balık (AYNI)
d_model: 64
num_heads: 4
dropout: 0.3
epochs: 200 (AYNI!)
Model: "with_dropout"
```

**Karşılaştırın**:
- 📈 Eğitim Grafiği'nde loss eğrilerini inceleyin
- Dropout=0.0 → Loss daha hızlı düşer ama overfitting riski
- Dropout=0.3 → Loss daha yavaş ama daha stabil

---

### Senaryo 5: Learning Rate Optimizasyonu (30 dakika)

**Hedef**: Optimal öğrenme hızını bulmak

#### Test 1: Çok Düşük
```
learning_rate: 0.0001
epochs: 50
Gözlem: Loss çok yavaş düşer
```

#### Test 2: Orta (Optimal)
```
learning_rate: 0.001
epochs: 50
Gözlem: Loss dengeli şekilde düşer
```

#### Test 3: Yüksek
```
learning_rate: 0.01
epochs: 50
Gözlem: Loss dalgalı, bazen patlar (NaN)
```

#### Test 4: Çok Yüksek
```
learning_rate: 0.1
epochs: 50
Gözlem: Loss hemen NaN olur!
```

**Ders**: learning_rate = 0.001 genellikle güvenli bir başlangıç noktasıdır.

---

### Senaryo 6: Gerçek Dünya Uygulaması - Cümle Analizi (40 dakika)

**Hedef**: Karmaşık bir cümlenin attention pattern'lerini analiz etmek

#### Veri
```
Bugün
Hava
Çok
Güzel
Olduğu
İçin
Parka
Gittik
```

#### Optimal Parametreler (bulacağız!)
```
Başlangıç:
d_model: 128
num_heads: 8
dropout: 0.2
learning_rate: 0.001
epochs: 100
```

#### Analiz Adımları

1. **İlk Eğitim**: Yukarıdaki parametrelerle eğitin
2. **Attention Analizi**:
   - "Gittik" satırına bakın
   - Hangi kelimelerle güçlü bağlantı var?
   - Beklenen: "Biz" (örtük özne), "Parka"
   
3. **İyileştirme Denemeleri**:
   - num_heads'i 12'ye çıkarın → Fark var mı?
   - d_model'i 256'ya çıkarın → Loss daha mı düşük?
   - dropout'u 0.1'e düşürün → Overfitting oldu mu?

4. **Sonuç**: En iyi parametreleri bulup kaydedin

---

## 🎨 GRAFİKLERİ ANLAMAK: BAŞLANGIÇ SEVIYESI REHBER

### 📊 Grafik 1: Attention Map

#### Bu Grafik Neyi Gösterir?
Attention Map, **her kelimenin diğer kelimelere ne kadar "dikkat ettiğini"** gösteren bir ısı haritasıdır. Bu, Self-Attention'ın kalbindeki mekanizmadır!

#### Grafiği Okuma Rehberi

**Eksenler:**
- **Y Ekseni (Solda, Dikey)**: Query kelimeleri - "Hangi kelime soruyor?"
- **X Ekseni (Altta, Yatay)**: Key kelimeleri - "Hangi kelimeye bakıyor?"
- **Her hücre**: Bir kelimenin başka bir kelimeye verdiği önemi gösterir

**Renkler:**
- 🟨 **Sarı/Açık Renkler**: YÜKSEK dikkat (0.7 - 1.0)
  - Bu kelimeler birbirine çok önemli!
  - Güçlü ilişki var
  - Örnek: "Kedi" → "Oturdu" (özne-fiil ilişkisi)

- 🟧 **Turuncu Renkler**: ORTA dikkat (0.3 - 0.7)
  - Var olan bir ilişki
  - Önemli ama birincil değil
  - Örnek: "Kedi" → "Mat" (özne-yer ilişkisi)

- 🟦 **Mavi/Koyu Renkler**: DÜŞÜK dikkat (0.0 - 0.3)
  - Zayıf veya yok ilişki
  - Kelimeler birbirini görmezden geliyor
  - Örnek: "Pazartesi" → "Cuma" (uzak günler)

#### Adım Adım Analiz Örneği

**Örnek Veri**: "Ben, Bugün, Okula, Gittim"

**1. İlk Bakış:**
```
Grafiği açın → 4x4'lük bir tablo göreceksiniz
Her satır bir kelime, her sütun bir kelime
16 hücre toplam (4 kelime x 4 kelime)
```

**2. Bir Satırı İnceleyin (Örnek: "Gittim"):**
```
"Gittim" satırını bulun (en altta)
Bu satır "Gittim" kelimesinin bakış açısı

Sütunlara bakın:
┌─────────────────────────────────────┐
│ Gittim → Ben      : 0.52 (PARLAK!) │ ← Özne arıyor!
│ Gittim → Bugün    : 0.21 (Orta)    │ ← Zaman önemli
│ Gittim → Okula    : 0.25 (Orta)    │ ← Yer önemli
│ Gittim → Gittim   : 0.02 (Koyu)    │ ← Kendine bakmıyor
└─────────────────────────────────────┘

YORUM: "Gittim" fiili, özneyi (Ben) arıyor! 
Bu DOĞRU bir dilbilgisel ilişki!
```

**3. Çapraz İlişkileri İnceleyin:**
```
"Ben" satırı, "Gittim" sütunu: 0.48 (Parlak)
"Gittim" satırı, "Ben" sütunu: 0.52 (Parlak)

YORUM: İki yönlü güçlü ilişki! 
Özne ve fiil birbirini tanıyor.
```

**4. Köşegen (Diagonal) İnceleyin:**
```
Köşegen = Her kelimenin kendisine dikkat etmesi

Ben → Ben      : Genellikle DÜŞÜK (0.1-0.2)
Bugün → Bugün  : Genellikle DÜŞÜK (0.1-0.2)
Okula → Okula  : Genellikle DÜŞÜK (0.1-0.2)
Gittim → Gittim: Genellikle DÜŞÜK (0.1-0.2)

YORUM: İyi! Kelimeler kendi içlerine değil, 
diğer kelimelere bakıyor. Self-attention çalışıyor!

⚠️ UYARI: Köşegen ÇOK PARLAK ise sorun var!
Bu, modelin diğer kelimeleri görmezden geldiği anlamına gelir.
```

#### Gerçek Dünya Örnekleri

**Örnek 1: Fiil-Özne İlişkisi**
```
Veri: "Kedi, Mat, Üzerinde, Oturdu"

Attention Map'e bakın:
"Oturdu" (satır) → "Kedi" (sütun): 0.65 (ÇOK PARLAK!)

Neden? Çünkü "oturdu" fiili bir özne arıyor!
"Oturdu" → "Mat": 0.25 (orta, yer belirtir)
"Oturdu" → "Üzerinde": 0.08 (düşük, ek bilgi)

✅ Model dilbilgisel ilişkiyi DOĞRU öğrenmiş!
```

**Örnek 2: Zaman Serisi İlişkisi**
```
Veri: "Pazartesi, Salı, Çarşamba, Perşembe"

Attention Map'te bakın:
"Salı" (satır) → "Pazartesi" (sütun): 0.45
"Salı" (satır) → "Salı" (sütun): 0.30
"Salı" (satır) → "Çarşamba" (sütun): 0.20
"Salı" (satır) → "Perşembe" (sütun): 0.05

YORUM: Salı, en çok önceki güne (Pazartesi) bakıyor!
Ardışıklık ilişkisi öğrenilmiş. Uzak günler (Perşembe) az ilgili.

✅ Model zamansal sıralamayı DOĞRU öğrenmiş!
```

**Örnek 3: Anlamsız Veri**
```
Veri: "Elma, Araba, Pazartesi, Kırmızı" (ilgisiz kelimeler)

Attention Map'te bakın:
Tüm hücreler benzer değerler (~0.25)
Hepsi eşit derecede "belirsiz"
Köşegen daha parlak (kendi kendine bakıyor)

❌ Model anlamlı ilişki bulamıyor!
Bu NORMAL, çünkü kelimeler gerçekten ilgisiz.
```

#### Sık Karşılaşılan Durumlar ve Anlamları

**Durum 1: Tüm Satır Eşit Dağılımlı**
```
Ben → Ben    : 0.25
Ben → Bugün  : 0.25
Ben → Okula  : 0.25
Ben → Gittim : 0.25

Anlam: Bu kelime herkese eşit dikkat ediyor
Yorum: Model net bir ilişki öğrenememiş
Çözüm: Daha fazla epoch, daha büyük d_model deneyin
```

**Durum 2: Bir Sütun Çok Parlak**
```
Tüm kelimeler → "Gittim": PARLAK

Anlam: "Gittim" kelimesi merkezi bir rol oynuyor
Yorum: Fiil/ana kelime olduğu için DOĞRU!
Çözüm: Sorun yok, bu iyi bir şey
```

**Durum 3: Sadece Köşegen Parlak**
```
Ben → Ben       : 0.90 ❌
Bugün → Bugün   : 0.85 ❌
Okula → Okula   : 0.80 ❌

Anlam: Kelimeler sadece kendilerine bakıyor!
Yorum: Self-attention çalışmıyor, model öğrenememiş
Çözüm: Learning rate artırın, epoch sayısını artırın
```

---

### 📊 Grafik 2: Q, K, V Matrisleri (İleri Seviye)

#### Bu Grafik Neyi Gösterir?
Self-Attention'ın "kaputun altı"! Her kelimenin Query, Key, Value vektörlerini gösterir.

#### Önce Teori: Q, K, V Nedir?

**Gerçek Dünya Analojisi:**
```
🔍 Query (Q): "Ne arıyorum?"
   Örnek: Kütüphanede "tarih kitabı arıyorum"
   
🔑 Key (K): "Ben kimim? Ne sunuyorum?"
   Örnek: Kitap rafında "Ben bir tarih kitabıyım"
   
💎 Value (V): "Bulunursam ne veriyorum?"
   Örnek: "Osmanlı İmparatorluğu hakkında bilgi"

Attention = Q ve K'nin uyumu × V'nin içeriği
```

**Matematiksel (Basitleştirilmiş):**
```
1. Her kelime → Q, K, V vektörlerine dönüştürülür
2. Q ve K karşılaştırılır → Benzerlik skoru (attention ağırlığı)
3. Bu skor ile V çarpılır → Bağlamsal temsil

Formül: Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

#### Grafiği Okuma Rehberi

**Görsel Yapı:**
```
Üç adet heatmap göreceksiniz:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Q Matrix  │  │   K Matrix  │  │   V Matrix  │
│  (Query)    │  │   (Key)     │  │  (Value)    │
└─────────────┘  └─────────────┘  └─────────────┘

Her matris:
- Satırlar: Kelimeler (örn: Ben, Bugün, Okula, Gittim)
- Sütunlar: Embedding boyutları (d_model kadar, örn: 64 sütun)
- Renkler: Vektör değerleri (-1 ile +1 arası)
```

**Renk Yorumlama:**
- 🔴 **Kırmızı/Sarı (+0.5 ile +1.0)**: POZİTİF aktivasyon, güçlü özellik
- ⚪ **Beyaz/Açık Gri (-0.2 ile +0.2)**: NÖTR, önemsiz boyut
- 🔵 **Mavi/Lacivert (-1.0 ile -0.5)**: NEGATİF aktivasyon, ters özellik

#### Adım Adım Analiz

**1. Q Matrisi (Query) Analizi:**
```
Soru: "Her kelime ne arıyor?"

Örnek: "Gittim" kelimesinin satırı
┌───────────────────────────────────────────┐
│ Pozisyon 0-15:  Kırmızı bantlar (0.8)     │ ← Özne özellikleri arıyor
│ Pozisyon 16-31: Mavi bantlar (-0.6)       │ ← Zaman özelliklerini es geçiyor
│ Pozisyon 32-47: Karışık pattern           │ ← Nesne özellikleri arıyor
│ Pozisyon 48-63: Nötr (beyaz)              │ ← Bu boyutlar önemsiz
└───────────────────────────────────────────┘

YORUM: "Gittim" fiili, özne ve nesne pattern'leri arıyor.
Zaman bilgisini es geçiyor (mavi = negatif).
```

**2. K Matrisi (Key) Analizi:**
```
Soru: "Her kelime kendini nasıl tanıtıyor?"

Örnek: "Ben" kelimesinin satırı
┌───────────────────────────────────────────┐
│ Pozisyon 0-15:  ÇOK kırmızı (0.9)         │ ← "Ben özneyim!"
│ Pozisyon 16-31: Nötr                      │ ← Zaman değilim
│ Pozisyon 32-47: Hafif kırmızı (0.3)       │ ← Belki nesne de olabilirim
│ Pozisyon 48-63: Mavi (-0.4)               │ ← Fiil kesinlikle değilim!
└───────────────────────────────────────────┘

YORUM: "Ben" özne olduğunu güçlü bir şekilde belirtiyor.
Q ve K'nin 0-15 pozisyonları uyuşuyor → Yüksek attention!
```

**3. V Matrisi (Value) Analizi:**
```
Soru: "Her kelime hangi bilgiyi taşıyor?"

Örnek: "Okula" kelimesinin satırı
┌───────────────────────────────────────────┐
│ Tüm boyutlar zengin pattern'ler          │
│ Çok renkli, karışık (iyi bir şey!)       │
│ Pozitif ve negatif değerler karışık      │
└───────────────────────────────────────────┘

YORUM: V matrisi en zengin olmalı!
Çünkü tüm bağlamsal bilgi burada saklanır.
```

#### Pattern'leri Karşılaştırma

**Benzer Kelimeler → Benzer Pattern'ler**
```
Veri: "Elma, Armut, Muz" (meyveler) + "Araba" (farklı)

K Matrisi'nde:
"Elma" satırı:  ████████░░░░░░░░ (pattern A)
"Armut" satırı: ████████░░░░░░░░ (pattern A - benzer!)
"Muz" satırı:   ████████░░░░░░░░ (pattern A - benzer!)
"Araba" satırı: ░░░░████████████ (pattern B - farklı!)

✅ Model benzer kelimeleri benzer encode ediyor!
```

**Farklı Roller → Farklı Pattern'ler**
```
Veri: "Kedi" (özne) vs "Oturdu" (fiil)

Q Matrisi'nde:
"Kedi":   █░░░█░░░███░ (pattern X - özne arıyor)
"Oturdu": ░░░░█████░░░ (pattern Y - özne arıyor)

"Kedi" ve "Oturdu" farklı roller ama Q pattern'leri benzer!
Neden? İkisi de özne arıyor → Q pattern'leri benzer olmalı!
```

#### Ne Zaman Endişelenmelisiniz?

**❌ Sorun 1: Tüm Satırlar Aynı**
```
Q/K/V'de tüm kelimeler aynı pattern → Model öğrenmemiş!
Çözüm: Daha fazla epoch, farklı learning rate
```

**❌ Sorun 2: Sadece Nötr Renkler (Beyaz)**
```
Hiç kırmızı/mavi yok, hep beyaz → Model aktivasyon üretemiyor!
Çözüm: Learning rate artırın, model çok küçük olabilir
```

**❌ Sorun 3: Aşırı Kırmızı veya Aşırı Mavi**
```
Sadece çok parlak kırmızı veya çok koyu mavi → Exploding gradients!
Çözüm: Learning rate düşürün, gradient clipping ekleyin
```

**✅ İyi Görünüm:**
```
- Karışık renkler (kırmızı, mavi, beyaz)
- Her satır farklı pattern
- V matrisi en zengin (en karışık)
- Benzer kelimeler benzer pattern'ler
```

---

### 📊 Grafik 3: Eğitim Geçmişi (Training Loss)

#### Bu Grafik Neyi Gösterir?
Modelin **öğrenme sürecini** gösterir. Her epoch'ta modelin ne kadar "hata yaptığını" (loss) görürsünüz.

#### Grafiği Okuma Rehberi

**Eksenler:**
- **X Ekseni (Yatay)**: Epoch sayısı (zaman çizelgesi)
  - Başlangıç: 0
  - Son: Belirlediğiniz epoch (örn: 50, 100, 200)
  
- **Y Ekseni (Dikey)**: Loss değeri (hata miktarı)
  - Yüksek değer = Çok hata (kötü)
  - Düşük değer = Az hata (iyi)
  - Aralık: Genellikle 0.0 - 3.0

**Çizgi:**
- Kahverengi çizgi: Modelin öğrenme yolculuğu
- Başlangıç (sol): Genellikle yüksek (model cahil)
- Son (sağ): Genellikle düşük (model öğrendi)

#### Adım Adım Analiz

**1. Başlangıç Değeri (İlk Epoch):**
```
Epoch 0'da loss: 2.5 - 3.5 arası

Ne anlama gelir?
- Model henüz hiçbir şey öğrenmemiş
- Rastgele tahmin yapıyor
- NORMAL bir durum, endişelenmeyin!

Örnek:
Epoch 1: Loss = 2.87 ← İyi bir başlangıç
```

**2. Öğrenme Eğrisi (İlk 10-20 Epoch):**
```
İdeal Görünüm:
Epoch 1:  2.87
Epoch 2:  2.45 ↓ (azalıyor - iyi!)
Epoch 3:  2.12 ↓
Epoch 4:  1.85 ↓
Epoch 5:  1.62 ↓
...
Epoch 10: 0.95 ↓

✅ Loss hızla düşüyor = Model aktif öğreniyor!

Sorunlu Görünüm:
Epoch 1:  2.87
Epoch 2:  2.85 (çok az düştü)
Epoch 3:  2.84 (hala çok az)
Epoch 4:  2.83
Epoch 5:  2.82 (çok yavaş!)

❌ Loss çok yavaş düşüyor = Learning rate çok düşük!
Çözüm: learning_rate'i 0.001'den 0.005'e çıkarın
```

**3. Stabilizasyon (Orta Dönem):**
```
Epoch 20-50 arası:

İdeal Görünüm:
Epoch 20: 0.75
Epoch 25: 0.68
Epoch 30: 0.64
Epoch 35: 0.61
Epoch 40: 0.59
Epoch 45: 0.58
Epoch 50: 0.57 (çok az düşüyor artık)

✅ Loss yavaş yavaş düzleşiyor = Model optimuma yaklaşıyor!

Sorunlu Görünüm 1: Dalgalanma
Epoch 20: 0.75
Epoch 25: 0.65
Epoch 30: 0.80 ↑ (yükseldi!)
Epoch 35: 0.60
Epoch 40: 0.85 ↑ (tekrar yükseldi!)

❌ Loss dalgalanıyor = Learning rate çok yüksek VEYA batch size çok küçük!
Çözüm: learning_rate'i 0.001'den 0.0005'e düşürün

Sorunlu Görünüm 2: Erken Durdurma
Epoch 20: 0.75
Epoch 25: 0.75 (değişmedi)
Epoch 30: 0.75 (hala aynı)
Epoch 35: 0.75

❌ Loss düşmüyor = Model yerel minimumda sıkıştı!
Çözüm: Modeli yeniden başlatın, farklı learning rate deneyin
```

**4. Final Değer (Son Epoch):**
```
Epoch 50'de loss ne olmalı?

Mükemmel: < 0.3
  ├─ Model çok iyi öğrendi!
  └─ Attention map'ler çok net olacak

İyi: 0.3 - 0.6
  ├─ Model yeterli öğrendi
  └─ Pratik kullanım için uygun

Orta: 0.6 - 1.0
  ├─ Model kısmen öğrendi
  └─ Daha fazla epoch deneyin

Kötü: > 1.0
  ├─ Model yeterince öğrenemedi
  └─ Parametreleri değiştirin

Örnek:
Epoch 50: Loss = 0.42 ← İyi bir sonuç!
```

#### Gerçek Dünya Örnekleri

**Örnek 1: Mükemmel Eğitim**
```
Veri: "Ben, Bugün, Okula, Gittim" (4 token, basit)
Parametreler: d_model=64, num_heads=4, lr=0.001

Grafik:
     3.0 |●
         |  ●
     2.0 |    ●●
         |       ●●●
Loss 1.0 |           ●●●●
         |                ●●●●●
     0.0 |________________________●●●●●●●●
         0   10   20   30   40   50
              Epoch

✅ Düzgün azalış, son değer 0.25, MÜKEMMELl!
```

**Örnek 2: Learning Rate Çok Yüksek**
```
Parametreler: lr=0.01 (10x fazla!)

Grafik:
     3.0 |●
         | ● ●
     2.0 |  ●  ●
         |●  ● ●  ●
Loss 1.0 | ●    ●  ●
         |●  ●   ●
     0.0 |________________________
         0   10   20   30   40   50
              Epoch

❌ Dalgalı, düzensiz, düşmüyor!
Çözüm: lr'yi 0.001'e düşürün
```

**Örnek 3: Learning Rate Çok Düşük**
```
Parametreler: lr=0.00001 (100 kat az!)

Grafik:
     3.0 |●───────────────────────
         |
     2.0 |
         |
Loss 1.0 |
         |
     0.0 |________________________
         0   10   20   30   40   50
              Epoch

❌ Hiç düşmüyor, düz çizgi!
Çözüm: lr'yi 0.001'e çıkarın
```

**Örnek 4: Overfitting**
```
Veri: 3 token (çok az!)
Parametreler: d_model=256 (çok büyük!), epochs=200

Grafik:
     3.0 |●
         |  ●●
     2.0 |     ●●
         |        ●●●●●
Loss 1.0 |              ●●●●●●
         |                     ●●●●●●─── (düzleşti)
     0.0 |_____________________________  ╱ (sonra yükselmeye başladı!)
         0   50   100  150  200
              Epoch

❌ Önce düştü, sonra tekrar yükseldi!
Çözüm: Dropout artırın (0.3'e çıkarın), erken durdurun
```

#### Grafikteki Anomaliler

**Anomali 1: Ani Sıçrama**
```
Epoch 35: 0.65
Epoch 36: 0.64
Epoch 37: 0.63
Epoch 38: 0.62
Epoch 39: 15.87 ← ANİ SIÇRAMA!
Epoch 40: NaN

Anlam: Exploding gradient! Model patladı!
Çözüm: Learning rate'i yarıya indirin, gradient clipping ekleyin
```

**Anomali 2: Plato (Düzlük)**
```
Epoch 20-50: Loss hep 1.25 civarında

Anlam: Model yerel minimumda sıkıştı
Çözüm: Learning rate'i artırın veya modeli sıfırdan başlatın
```

**Anomali 3: Negatif Loss**
```
Epoch 50: Loss = -0.45

Anlam: Kodda hata var! Loss negatif olamaz!
Çözüm: Bunu görürseniz kodu inceleyin, bu bir bug
```

#### Parametrelerin Etkisi (Karşılaştırma)

**d_model Etkisi:**
```
d_model=32:  Final loss = 0.75 (yüksek)
d_model=64:  Final loss = 0.42 (orta) ← STANDART
d_model=128: Final loss = 0.28 (düşük)
d_model=256: Final loss = 0.15 (çok düşük ama yavaş)

Kural: Daha büyük model → Daha düşük loss (ama daha yavaş)
```

**num_heads Etkisi:**
```
num_heads=2:  Final loss = 0.55 (orta)
num_heads=4:  Final loss = 0.42 (iyi) ← STANDART
num_heads=8:  Final loss = 0.38 (daha iyi)
num_heads=16: Final loss = 0.36 (en iyi ama yavaş)

Kural: Daha fazla head → Daha düşük loss (ama azalan getiri)
```

**dropout Etkisi:**
```
dropout=0.0: Loss hızla düşer ama sonra yükselir (overfitting)
dropout=0.1: Dengeli düşüş ← STANDART
dropout=0.3: Yavaş ama stabil düşüş
dropout=0.5: Çok yavaş, belki hiç düşmez

Kural: Dropout daha stabil ama daha yavaş öğrenme sağlar
```

---

## 🔬 Tüm Grafikleri Birlikte Değerlendirme

### Senaryo 1: Mükemmel Sonuç

**Eğitim Grafiği:**
- Loss 2.8'den 0.25'e düştü ✅
- Düzgün, dalgasız eğri ✅
- Son 10 epoch düzleşmiş ✅

**Attention Map:**
- Anlamlı pattern'ler (fiil-özne ilişkisi) ✅
- Köşegen koyu (kendi kendine bakmıyor) ✅
- Net, parlak bağlantılar ✅

**Q, K, V Matrisleri:**
- Her kelime farklı pattern ✅
- Benzer kelimeler benzer vector'ler ✅
- V matrisi zengin ve karışık ✅

**SONUÇ**: Model mükemmel öğrenmiş! 🎉

---

### Senaryo 2: Kötü Sonuç

**Eğitim Grafiği:**
- Loss 2.8'de kaldı, düşmedi ❌
- Düz bir çizgi ❌

**Attention Map:**
- Tüm hücreler ~0.25 (eşit dağılım) ❌
- Köşegen parlak ❌
- Net pattern yok ❌

**Q, K, V Matrisleri:**
- Tüm satırlar aynı pattern ❌
- Nötr renkler (beyaz), aktivasyon yok ❌

**SONUÇ**: Model öğrenemedi!
**Çözüm**: Learning rate artırın, daha fazla epoch deneyin

---

### Senaryo 3: Overfitting

**Eğitim Grafiği:**
- Loss başta düştü, sonra tekrar yükseldi ⚠️

**Attention Map:**
- Aşırı keskin pattern'ler ⚠️
- Sadece 1-2 bağlantı çok güçlü ⚠️

**Q, K, V Matrisleri:**
- Aşırı parlak renkler (çok kırmızı/mavi) ⚠️

**SONUÇ**: Model ezberlemiş, genelleştiremiyor!
**Çözüm**: Dropout artırın (0.3), daha az epoch kullanın

---

## 📌 Hızlı Referans Tablosu

### Grafik Kontrol Listesi

| Grafik | Ne Kontrol Edin | İyi Görünüm | Kötü Görünüm |
|--------|----------------|-------------|--------------|
| **Attention Map** | İlişki pattern'leri | Net, parlak bağlantılar | Eşit dağılım, köşegen parlak |
| **Q, K, V** | Vektör çeşitliliği | Karışık renkler, farklı satırlar | Tüm satırlar aynı, nötr |
| **Training Loss** | Düşüş trendi | Düzgün azalış, düşük final | Dalgalı, yüksek final |

### Loss Değeri Yorumlama

| Final Loss | Yorum | Yapılacak |
|------------|-------|-----------|
| < 0.3 | Mükemmel! | Hiçbir şey, devam edin |
| 0.3 - 0.6 | İyi | Belki d_model artırın |
| 0.6 - 1.0 | Orta | Daha fazla epoch, lr ayarlayın |
| 1.0 - 2.0 | Zayıf | Parametreleri değiştirin |
| > 2.0 | Çok kötü | Her şeyi değiştirin |
| NaN | HATA! | lr'yi çok düşürün |

### Attention Ağırlık Yorumlama

| Değer | Renk | Anlam |
|-------|------|-------|
| 0.0 - 0.1 | Koyu mavi | İlişki yok, görmezden geliyor |
| 0.1 - 0.3 | Açık mavi | Zayıf ilişki, az önemli |
| 0.3 - 0.5 | Turuncu | Orta ilişki, kısmen önemli |
| 0.5 - 0.7 | Sarı | Güçlü ilişki, çok önemli |
| 0.7 - 1.0 | Parlak sarı | Çok güçlü ilişki, kritik |

---

## 🎯 Pratik Egzersizler

### Egzersiz 1: Attention Map Okuma
1. Programı açın, "Ben, Bugün, Okula, Gittim" veriyle eğitin
2. Attention Map'i açın
3. "Gittim" satırını bulun
4. En parlak hücre hangisi? (Cevap: "Ben" sütunu olmalı)
5. Bu neden mantıklı? (Cevap: Fiil özne arıyor!)

### Egzersiz 2: Loss Grafiği Yorumlama
1. İlk loss değerini not alın (örn: 2.87)
2. Final loss değerini not alın (örn: 0.42)
3. Farkı hesaplayın: 2.87 - 0.42 = 2.45 azalma
4. Yüzde hesaplama: (2.45 / 2.87) × 100 = %85.4 iyileşme!
5. Grafik düzgün mü yoksa dalgalı mı?

### Egzersiz 3: Q, K, V Pattern Karşılaştırma
1. Q matrisinde "Gittim" satırını bulun
2. K matrisinde "Ben" satırını bulun
3. İlk 10 sütunu karşılaştırın
4. Pattern'ler benzer mi? (Benzer olmalı, ikisi de özne-fiil ilişkisi!)

---

## 💡 Son Tavsiyeler

### Grafik Analizi İçin Altın Kurallar

1. **Önce Eğitim Grafiğini İnceleyin**
   - Loss düşmediyse, diğer grafikler anlamsız!
   - Loss düştüyse, attention'a bakmaya değer

2. **Attention Map'te Hikaye Arayın**
   - "Bu kelime neden bu kelimeye bakıyor?"
   - Dilbilgisel veya anlamsal mantık var mı?

3. **Q, K, V'yi İleri Seviye İçin Saklayın**
   - Başlangıçta sadece Attention Map yeterli
   - Detaylı analiz için sonra inceleyin

4. **Parametreleri Tek Tek Değiştirin**
   - Her seferinde sadece bir parametreyi değiştirin
   - Etkiyi net görmek için karşılaştırın

5. **Grafikleri Kaydedin ve Karşılaştırın**
   - PNG dosyalarını yan yana koyun
   - Zamanla pattern'leri anlamaya başlayacaksınız!

**Bu rehber ile artık Self-Attention grafiklerin profesyonel değerlendiricisisiniz! 🎓**

---

## 📊 Veri Hazırlama İpuçları

### İyi Veri Örnekleri

#### Dil İşleme
```
✓ İyi: Anlamlı cümleler
"Kedi mat üzerinde oturdu"
"Ben bugün okula gittim"

✗ Kötü: Rastgele kelimeler
"Masa araba gökyüzü bilgisayar"
```

#### Zaman Serisi
```
✓ İyi: Sıralı veriler
"Ocak, Şubat, Mart, Nisan"
"Sabah, Öğle, Akşam, Gece"

✗ Kötü: Karışık sıra
"Mart, Ocak, Nisan, Şubat"
```

#### Kategorik Veriler
```
✓ İyi: İlişkili kategoriler
"Elma, Armut, Muz, Üzüm" (meyveler)
"Kırmızı, Mavi, Yeşil, Sarı" (renkler)

✗ Kötü: İlgisiz kategoriler
"Elma, Araba, Pazartesi, Mavi"
```

### Veri Boyutu Önerileri

```
Minimum: 3-4 token
Optimal: 5-10 token
Maksimum: 20-30 token (performans için)

Not: Çok az token → Basit ilişkiler
     Çok fazla token → Yavaş eğitim
```

---

## 🎯 Öğrenme Hedefleri ve Kontrol Listesi

### Temel Seviye (1-2 hafta)
- [ ] Self-Attention'ın ne olduğunu anlıyorum
- [ ] Q, K, V kavramlarını açıklayabiliyorum
- [ ] Attention map'i okuyabiliyorum
- [ ] Basit parametreleri ayarlayabiliyorum
- [ ] Model kaydedip yükleyebiliyorum

### Orta Seviye (3-4 hafta)
- [ ] Multi-head attention'ın avantajlarını anlıyorum
- [ ] Parametrelerin etkilerini tahmin edebiliyorum
- [ ] Attention pattern'lerini yorumlayabiliyorum
- [ ] Kendi verilerimle deney yapabiliyorum
- [ ] Eğitim grafiklerini analiz edebiliyorum

### İleri Seviye (5+ hafta)
- [ ] Optimal hiperparametreleri bulabiliyorum
- [ ] Overfitting/underfitting'i tespit edebiliyorum
- [ ] Karmaşık ilişkileri modelleyebiliyorum
- [ ] Farklı attention pattern'lerini karşılaştırabiliyorum
- [ ] Gerçek problemlere uygulayabiliyorum

---

## 💡 Sık Sorulan Sorular

### Q: Eğitim ne kadar sürmeli?
**A**: Veri boyutuna bağlı:
- 3-5 token: 20-50 epoch (~1 dakika)
- 6-10 token: 50-100 epoch (~2-3 dakika)
- 10+ token: 100-200 epoch (~5-10 dakika)

### Q: En iyi parametreler neler?
**A**: Veri boyutuna göre değişir:
```
Küçük veri (<10 token):
d_model=64, num_heads=4, dropout=0.1, lr=0.001

Orta veri (10-20 token):
d_model=128, num_heads=8, dropout=0.2, lr=0.001

Büyük veri (>20 token):
d_model=256, num_heads=16, dropout=0.3, lr=0.0005
```

### Q: GPU gerekli mi?
**A**: Hayır! CPU ile de rahatça çalışır. GPU varsa otomatik kullanılır.

### Q: Modeller ne kadar yer kaplar?
**A**: Model başına ~1-10 MB (parametrelere bağlı)

### Q: Kaç model kaydedebilirim?
**A**: Sınırsız! Ama düzenli temizlik yapın.

---

## 🚀 İleri Seviye Deneyler

### Deney 1: Positional Encoding Etkisi
```
Aynı kelimeleri farklı sıralarda deneyin:
Veri 1: "Kedi köpek kuş"
Veri 2: "Kuş köpek kedi"

Gözlem: Farklı attention pattern'leri göreceksiniz!
Bu, positional encoding'in çalıştığını gösterir.
```

### Deney 2: Uzun Mesafe Bağımlılıklar
```
Veri: "Ben, Bugün, Çok, Erken, Kalktım, Ve, İşe, Gittim"

Gözlem: "Ben" ve "Gittim" arasındaki ilişki
7 kelime mesafeden tespit edilebilir mi?
```

### Deney 3: Eş Anlamlı Kelimeler
```
Veri: "Güzel, Çirkin, Hoş, İğrenç"

Gözlem: "Güzel" ve "Hoş" benzer attention pattern'leri
oluşturur mu? (Eş anlamlı oldukları için olmalı!)
```

---

## 📈 İlerleme Takibi

### Günlük Kontrol Listesi
```
Gün 1: [ ] Programı çalıştır, temel kavramları öğren
Gün 2: [ ] İlk örneği dene, attention map'i incele
Gün 3: [ ] num_heads'i değiştir, farkı gözle
Gün 4: [ ] d_model'i değiştir, etkiyi analiz et
Gün 5: [ ] Kendi verini oluştur, dene
Gün 6: [ ] Optimal parametreleri bul
Gün 7: [ ] Öğrendiklerini not al, özetle
```

### Haftalık Hedefler
```
Hafta 1: Temel kavramlar ve ilk deneyler
Hafta 2: Parametre etkilerini anlama
Hafta 3: Karmaşık deneyler yapma
Hafta 4: Gerçek problemlere uygulama
```

---

## 🎓 Sonuç

Bu program ile:
- ✅ Self-Attention'ı **görsel olarak** anlayacaksınız
- ✅ Q, K, V kavramlarını **örneklerle** öğreneceksiniz
- ✅ Parametrelerin etkilerini **aktif olarak** gözleyeceksiniz
- ✅ Modern AI sistemlerinin temelini **deneyerek** öğreneceksiniz

**İyi öğrenmeler! 🚀**

---

## 📝 Not Alma Şablonu

Her deney için bu şablonu kullanın:

```markdown
## Deney: [Deney Adı]
Tarih: [Tarih]

### Parametreler
- Veri: [Token listesi]
- d_model: [Değer]
- num_heads: [Değer]
- dropout: [Değer]
- learning_rate: [Değer]
- epochs: [Değer]

### Sonuçlar
- Final Loss: [Değer]
- Eğitim Süresi: [Dakika]
- Gözlemler: [Notlar]

### Attention Pattern'leri
- En güçlü bağlantı: [Token1] → [Token2] (ağırlık: X.XX)
- En zayıf bağlantı: [Token3] → [Token4] (ağırlık: X.XX)

### Öğrenilen Dersler
1. [Ders 1]
2. [Ders 2]
3. [Ders 3]

### Sonraki Adımlar
- [ ] [Yapılacak 1]
- [ ] [Yapılacak 2]
```

**Bu şablonu her deney için kullanarak ilerlemenizi takip edin!**
