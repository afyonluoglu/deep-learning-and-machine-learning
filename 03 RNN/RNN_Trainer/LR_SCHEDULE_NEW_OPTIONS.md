# Learning Rate Schedule Yeni Seçenekler

## 📋 Eklenen LR Schedule Stratejileri

### 1. **Reduce on Plateau**
Kayıp (loss) değeri düzelmediğinde learning rate'i azaltır.

#### Parametreler:
- **patience**: Kaç epoch beklenecek (default: 10)
- **factor**: LR azaltma faktörü (default: 0.5)
- **min_lr**: Minimum learning rate (default: 1e-6)

#### Çalışma Prensibi:
```python
if loss < best_loss:
    best_loss = loss
    wait = 0
else:
    wait += 1
    if wait >= patience:
        current_lr = max(current_lr * factor, min_lr)
        wait = 0
```

#### Ne Zaman Kullanılır:
- Loss değeri platoya ulaştığında
- Otomatik LR ayarlaması istendiğinde
- Model öğrenmeyi yavaşladığında

### 2. **Cyclical LR (CLR)**
Learning rate'i periyodik olarak minimum ve maksimum değerler arasında gidip geldirir.

#### Parametreler:
- **base_lr**: Minimum learning rate (default: initial_lr * 0.1)
- **max_lr**: Maximum learning rate (default: initial_lr)
- **step_size_cycle**: Yarım döngü adım sayısı (default: 2000)

#### Çalışma Prensibi:
```python
cycle = floor(1 + cycle_step / (2 * step_size_cycle))
x = abs(cycle_step / step_size_cycle - 2 * cycle + 1)
lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
```

#### Formül Açıklaması:
- Her döngü `2 * step_size_cycle` adımdan oluşur
- İlk yarıda: base_lr → max_lr (artış)
- İkinci yarıda: max_lr → base_lr (azalış)
- Üçgensel dalga formu

#### Ne Zaman Kullanılır:
- Hızlı yakınsama istendiğinde
- Lokal minimumlarden kaçmak için
- Saddle point'lerden geçmek için

#### Avantajları:
- Daha hızlı eğitim
- Daha iyi genelleme
- Lokal minimumlardan kaçış

### 3. **Warmup + Decay**
Başlangıçta yavaş yavaş artırır (warmup), sonra üstel olarak azaltır (decay).

#### Parametreler:
- **warmup_steps**: Warmup adım sayısı (default: 1000)
- **decay_steps**: Decay adım sayısı (default: 10000)
- **min_lr**: Minimum learning rate (default: 1e-6)

#### Çalışma Prensibi:
```python
# Warmup Phase (Linear)
if epoch < warmup_steps:
    lr = initial_lr * (epoch + 1) / warmup_steps

# Decay Phase (Exponential)
else:
    decay_rate = (min_lr / initial_lr) ** (1 / decay_steps)
    steps_after_warmup = epoch - warmup_steps
    lr = initial_lr * (decay_rate ** steps_after_warmup)
    lr = max(lr, min_lr)
```

#### İki Aşama:
1. **Warmup (Isınma)**: 
   - Linear olarak 0'dan initial_lr'ye
   - Modelin kararlı başlaması için
   - Warmup_steps kadar sürer

2. **Decay (Azalma)**:
   - Üstel azalma
   - Min_lr'ye kadar iner
   - Decay_steps kadar sürer

#### Ne Zaman Kullanılır:
- Adam optimizer ile
- Transformer modellerde
- Büyük batch size'larda
- Kararlı başlangıç istendiğinde

#### Avantajları:
- Kararlı başlangıç
- Gradient explosion önleme
- Daha iyi yakınsama

## 📊 Mevcut LR Schedule'lar (Güncellendi)

### 1. Constant
- Sabit learning rate
- En basit yöntem

### 2. Step
- Belirli adımlarda azaltma
- `step_size` her X epoch'ta
- `gamma` ile çarpılır

### 3. Exponential
- Üstel azalma
- Her epoch'ta `gamma` ile çarpılır
- Sürekli azalma

### 4. Cosine
- Kosinüs fonksiyonu
- Yumuşak azalma
- T_max epoch'a kadar

## 🔧 Kod Değişiklikleri

### 1. rnn_trainer_app.py
**Line ~217-221**: LR Schedule dropdown güncellendi
```python
lr_schedule_menu = ctk.CTkOptionMenu(
    frame,
    values=["constant", "step", "exponential", "cosine", 
            "reduce_on_plateau", "cyclical", "warmup_decay"],
    variable=self.lr_schedule_var
)
```

### 2. optimizers.py
**LearningRateScheduler Sınıfı Güncellendi:**

#### __init__ Metodu:
```python
# Yeni state değişkenleri eklendi
self.best_loss = float('inf')      # reduce_on_plateau için
self.wait = 0                       # reduce_on_plateau için
self.current_lr = initial_lr        # reduce_on_plateau için
self.cycle_step = 0                 # cyclical için
```

#### get_lr Metodu:
```python
def get_lr(self, epoch: int = None, loss: float = None) -> float:
    # loss parametresi eklendi (reduce_on_plateau için)
    
    # 3 yeni schedule tipi eklendi:
    elif self.schedule_type == 'reduce_on_plateau':
        # ... kod ...
    
    elif self.schedule_type == 'cyclical':
        # ... kod ...
    
    elif self.schedule_type == 'warmup_decay':
        # ... kod ...
```

#### step Metodu:
```python
def step(self, loss: float = None):
    # loss parametresi eklendi
    self.current_epoch += 1
    self.cycle_step += 1  # cyclical için
```

### 3. rnn_model.py
**Line ~375-380**: Scheduler çağrıları güncellendi
```python
# Önceki:
self.optimizer.learning_rate = self.lr_scheduler.get_lr()
self.lr_scheduler.step()

# Yeni:
self.optimizer.learning_rate = self.lr_scheduler.get_lr(loss=avg_loss)
self.lr_scheduler.step(loss=avg_loss)
```

**Line ~48-50**: Docstring güncellendi
```python
lr_schedule: Learning rate schedule ('constant', 'step', 'exponential', 'cosine', 
            'reduce_on_plateau', 'cyclical', 'warmup_decay')
```

## 📈 Kullanım Örnekleri

### Reduce on Plateau
```python
# Model eğitimi sırasında
model = RNNModel(
    learning_rate=0.01,
    lr_schedule='reduce_on_plateau',
    patience=10,         # 10 epoch bekle
    factor=0.5,          # Yarıya indir
    min_lr=1e-6          # En az bu kadar
)
```

**Senaryo:**
- Başlangıç: lr = 0.01
- 10 epoch boyunca loss düzelmezse: lr = 0.005
- 10 epoch daha loss düzelmezse: lr = 0.0025
- ... devam eder ...

### Cyclical LR
```python
model = RNNModel(
    learning_rate=0.01,
    lr_schedule='cyclical',
    base_lr=0.001,        # Minimum
    max_lr=0.01,          # Maximum
    step_size_cycle=2000  # 2000 adımda bir tam döngü
)
```

**Senaryo:**
- Adım 0-2000: 0.001 → 0.01 (artış)
- Adım 2000-4000: 0.01 → 0.001 (azalış)
- Adım 4000-6000: 0.001 → 0.01 (tekrar)
- ... devam eder ...

### Warmup + Decay
```python
model = RNNModel(
    learning_rate=0.01,
    lr_schedule='warmup_decay',
    warmup_steps=1000,    # İlk 1000 adım warmup
    decay_steps=10000,    # 10000 adımda decay
    min_lr=1e-6           # Minimum lr
)
```

**Senaryo:**
- Adım 0-1000: 0 → 0.01 (linear artış)
- Adım 1000-11000: 0.01 → 0.000001 (üstel azalma)
- Adım 11000+: 0.000001 (sabit minimum)

## 🎯 Hangi Schedule'u Ne Zaman Kullanmalı?

### Constant
✅ **Kullan:**
- Küçük modellerde
- Basit problemlerde
- LR zaten iyi ayarlandıysa

### Step
✅ **Kullan:**
- Geleneksel CNN/RNN'lerde
- Epoch sayısı belli ise
- Manuel kontrol istenirse

### Exponential
✅ **Kullan:**
- Sürekli azalan LR istenirse
- Transfer learning'de fine-tuning
- Uzun eğitimlerde

### Cosine
✅ **Kullan:**
- Modern deep learning'de
- Yumuşak azalma istenirse
- SOTA modellerde

### Reduce on Plateau ⭐ YENİ
✅ **Kullan:**
- Loss platoya ulaştığında
- Otomatik ayarlama istenirse
- Uzun eğitim süreçlerinde
- Validation loss takip edilirse

❌ **Kullanma:**
- Çok gürültülü loss'larda
- Kısa eğitimlerde

### Cyclical LR ⭐ YENİ
✅ **Kullan:**
- Lokal minimumlardan kaçmak için
- Hızlı yakınsama istenirse
- Saddle point problemlerinde
- ResNet, DenseNet gibi modellerde

❌ **Kullanma:**
- Çok hassas eğitimlerde
- Fine-tuning'de

### Warmup + Decay ⭐ YENİ
✅ **Kullan:**
- Adam optimizer ile
- Transformer modellerde
- Büyük batch size'larda
- BERT, GPT gibi modellerde

❌ **Kullanma:**
- SGD optimizer ile
- Küçük batch size'larda

## 📊 LR Schedule Karşılaştırması

| Schedule | Karmaşıklık | Otomatik | Hız | Kararlılık | Kullanım |
|----------|-------------|----------|-----|------------|----------|
| Constant | ⭐ | ❌ | ⭐⭐ | ⭐⭐⭐ | Basit |
| Step | ⭐⭐ | ❌ | ⭐⭐⭐ | ⭐⭐⭐ | Orta |
| Exponential | ⭐⭐ | ❌ | ⭐⭐⭐ | ⭐⭐ | Orta |
| Cosine | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐ | İleri |
| Reduce on Plateau | ⭐⭐⭐ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ | İleri |
| Cyclical | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐ | İleri |
| Warmup+Decay | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | İleri |

## ✅ Test Sonuçları

### Başarılı Testler:
1. ✅ GUI'ye 3 yeni schedule eklendi
2. ✅ LearningRateScheduler sınıfı güncellendi
3. ✅ Reduce on Plateau implementasyonu
4. ✅ Cyclical LR implementasyonu
5. ✅ Warmup + Decay implementasyonu
6. ✅ Model eğitimi sırasında loss geçişi
7. ✅ Program hatasız çalışıyor

### Test Edilen Özellikler:
- Dropdown menüde tüm seçenekler görünüyor
- Model başarıyla initialize ediliyor
- LR scheduler doğru parametrelerle çalışıyor

## 📝 Önemli Notlar

### Reduce on Plateau:
- Loss değeri her epoch'ta geçilmeli
- `patience` değeri veri setine göre ayarlanmalı
- Çok küçük `patience` → Erken azalma
- Çok büyük `patience` → Geç azalma

### Cyclical LR:
- `step_size_cycle` epoch sayısına göre ayarlanmalı
- Genellikle: total_epochs / 8 civarı
- base_lr/max_lr oranı: 1:10 veya 1:3 arası

### Warmup + Decay:
- `warmup_steps` genellikle total_steps'in %5-10'u
- Adam optimizer için önerilen
- Büyük modellerde etkili

## 🔄 Gelecek İyileştirmeler (Opsiyonel)

1. **OneCycle Policy**: Cyclical'ın gelişmiş versiyonu
2. **Polynomial Decay**: Polinom fonksiyonu ile azalma
3. **Linear Schedule with Warmup**: BERT'te kullanılan
4. **Cosine with Restarts**: Periyodik restart'lar
5. **Custom Schedule**: Kullanıcı tanımlı fonksiyon

## 📚 Referanslar

### Cyclical Learning Rates:
- Paper: "Cyclical Learning Rates for Training Neural Networks" (Leslie Smith, 2017)
- Fikir: LR'yi düzenli olarak değiştirmek daha iyi sonuç verir

### Warmup:
- Paper: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Goyal et al., 2017)
- Kullanım: BERT, GPT, Transformer modellerde standart

### Reduce on Plateau:
- PyTorch: ReduceLROnPlateau
- Keras: ReduceLROnPlateau callback
- Otomatik LR ayarlaması

---

**Tamamlanma Tarihi**: 2025-10-01
**Durum**: ✅ Başarıyla Tamamlandı
**Test Sonucu**: ✅ Tüm Testler Geçti
**Yeni Özellikler**: 3 yeni LR schedule stratejisi
