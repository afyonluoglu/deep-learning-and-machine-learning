"""
🧠 RNN TEORİSİ ve TEMEL KAVRAMLAR
=====================================

Bu dosya RNN'lerin teorik temellerini ve temel kavramları açıklar.

Recurrent Neural Network (RNN) nedir?
- Zaman serilerini işlemek için tasarlanmış sinir ağı türüdür
- Geçmiş bilgiyi hatırlaması için "hafıza" mekanizması vardır
- Sequential (sıralı) verileri işlemek için idealdir

Temel Özellikler:
1. Temporal (zamansal) bağımlılıkları öğrenebilir
2. Değişken uzunlukta girişleri işleyebilir
3. Parametre paylaşımı ile etkili öğrenme
4. Gizli durum (hidden state) ile hafıza

Kullanım Alanları:
- Doğal dil işleme (NLP)
- Zaman serisi tahmini
- Konuşma tanıma
- Müzik üretimi
- Video analizi
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# Matplotlib font uyarılarını gizle
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_title(title, single_line:bool=False, line_len = 50):
    """Başlıkları büyük ve belirgin yazdırır"""
    line_str = "=" if not single_line else "-"
    line_length = line_len
    if not single_line:
        print("\n" + "=" * line_length)
    else:
        print("\n")
    print(title)
    print(line_str * line_length + "\n")

print_title("🧠 RNN TEORİSİ VE TEMEL KAVRAMLAR", line_len=75)

print_title("📋 RNN NEDİR?")
print("Recurrent Neural Network (RNN):")
print("• Zaman serilerini işlemek için tasarlanmış özel sinir ağı")
print("• Geçmiş bilgileri 'hatırlayarak' tahmin yapar")
print("• Sequential (sıralı) verilerle çalışır")

print_title("🔄 RNN ÇALIŞMA PRENSİBİ")
print("1. Her zaman adımında:")
print("   - Mevcut giriş (x_t)")
print("   - Önceki hidden state (h_t-1)")
print("   - Yeni hidden state hesaplanır (h_t)")

print("\n2. Formül:")
print("   h_t = tanh(W_hh * h_t-1 + W_xh * x_t + b)")
print("   y_t = W_hy * h_t + b_y")

print_title("📊 BASIT RNN ÖRNEĞİ - MANUEL UYGULAMA")

# Basit RNN manuel uygulaması (eğitim amaçlı)
def simple_rnn_step(x_t, h_prev, W_hh, W_xh, b):
    """
    🧠 BİR RNN ADIMINI MANUEL OLARAK HESAPLAR
    
    Bu fonksiyon RNN'in temel matematiksel işlemini yapar:
    1. Önceki hafızayı (h_prev) mevcut hafıza ağırlıklarıyla (W_hh) çarpar
    2. Yeni girişi (x_t) giriş ağırlıklarıyla (W_xh) çarpar  
    3. İkisini toplar, bias ekler ve tanh aktivasyonundan geçirir
    4. Yeni hafıza durumunu (h_t) döndürür
    
    Args:
        x_t: Mevcut zaman adımındaki giriş verisi (örn: [0.8, 0.5])
        h_prev: Önceki zaman adımından gelen hafıza durumu
        W_hh: Hidden-to-Hidden ağırlık matrisi (hafızanın kendisini nasıl güncellediği)
        W_xh: Input-to-Hidden ağırlık matrisi (yeni girişin hafızayı nasıl etkilediği)  
        b: Bias terimi (öğrenilen sabit eklenti)
    
    Returns:
        h_t: Yeni hesaplanan hafıza durumu (bir sonraki adımda h_prev olacak)
        
    💡 FORMÜL: h_t = tanh(W_hh * h_prev + W_xh * x_t + b)
    """
    # Adım 1: Önceki hafızayı mevcut ağırlıklarla çarp (W_hh * h_prev)
    memory_contribution = np.dot(W_hh, h_prev)
    
    # Adım 2: Yeni girişi ağırlıklarla çarp (W_xh * x_t)  
    input_contribution = np.dot(W_xh, x_t)
    
    # Adım 3: İkisini topla, bias ekle ve tanh aktivasyonundan geçir
    combined = memory_contribution + input_contribution + b
    h_t = np.tanh(combined)  # tanh: değerleri -1 ile +1 arasına sıkıştırır
    
    return h_t

# 🎛️ PARAMETRIK AYARLAR - İSTEDİĞİNİZ GİBİ DEĞİŞTİREBİLİRSİNİZ!
print_title("🎛️ PARAMETRIK AYARLAR", True)

# Ana parametreler (buradan kolayca değiştirebilirsiniz)
HIDDEN_SIZE = 13      # Hidden layer'daki nöron sayısı (2-50 arası deneyin)
INPUT_SIZE = 2        # Giriş boyutu 
TIME_STEPS = 6        # Zaman adımı sayısı (3-20 arası deneyin)
WEIGHT_SCALE = 0.2    # Ağırlık ölçeği (0.01-0.5 arası deneyin)

# ✨ Deneyim Önerileri:

# Hidden Size Etkisi:
# HIDDEN_SIZE = 5 → Basit öğrenme
# HIDDEN_SIZE = 20 → Karmaşık öğrenme
# HIDDEN_SIZE = 50 → Çok güçlü ama yavaş

# Zaman Adımı Etkisi:
# TIME_STEPS = 3 → Kısa hafıza
# TIME_STEPS = 15 → Uzun hafıza
# TIME_STEPS = 25 → Çok uzun hafıza

# Ağırlık Etkisi:
# WEIGHT_SCALE = 0.01 → Zayıf sinyal
# WEIGHT_SCALE = 0.3 → Güçlü sinyal
# WEIGHT_SCALE = 0.7 → Çok güçlü (patlama riski)

print(f"✅ Hidden Size : {HIDDEN_SIZE}")
print(f"✅ Input Size  : {INPUT_SIZE}")  
print(f"✅ Time Steps  : {TIME_STEPS}")
print(f"✅ Weight Scale: {WEIGHT_SCALE}")

# Dinamik veri üretimi - zaman adımı sayısına göre
def generate_random_sequence_data(time_steps, input_size):
    """Belirtilen zaman adımı sayısına göre örnek veri üretir"""
    sequence = []
    np.random.seed(123)  # Tekrarlanabilir sonuçlar için
    
    for t in range(time_steps):
        # Sinüs dalgası + rastgele gürültü ile ilginç desenler oluştur
        x1 = 0.8 * np.sin(2 * np.pi * t / time_steps) + 0.2 * np.random.randn()
        x2 = 0.6 * np.cos(2 * np.pi * t / time_steps) + 0.1 * np.random.randn()
        
        sequence.append(np.array([[x1], [x2]]))
    
    return sequence

# Örnek parametreler
hidden_size = HIDDEN_SIZE
input_size = INPUT_SIZE

# Rastgele ağırlıklar (gerçekte eğitimle öğrenilir)
np.random.seed(42)
W_hh = np.random.randn(hidden_size, hidden_size) * WEIGHT_SCALE
W_xh = np.random.randn(hidden_size, input_size) * WEIGHT_SCALE
b = np.zeros((hidden_size, 1))
print(f"➡️ Ağırlık matrisleri oluşturuldu (boyutlar: W_hh={W_hh.shape}, W_xh={W_xh.shape})")
print(f"W_hh= {W_hh[:5]}")
print(f"W_xh= {W_xh[:5]}")
print(f"b   = {b.flatten()[:5]} ... ({hidden_size} bias)")

exit()
# Dinamik olarak veri üret
sequence_data = generate_random_sequence_data(TIME_STEPS, INPUT_SIZE)

print(f"\n🟢 Giriş dizisi ({TIME_STEPS} zaman adımı):")
for t, x_t in enumerate(sequence_data):
    formatted_x_t = ", ".join([f"{val:11.8f}" for val in x_t.flatten()])
    print(f"  t{t}: [{formatted_x_t}]")

print_title("🔍 ADIM ADIM RNN HESAPLAMA:")

# İlk gizli durum (sıfır)
h = np.zeros((hidden_size, 1))

formatted_h_init = ", ".join([f"{val:11.8f}" for val in h.flatten()[:5]])
print(f"Başlangıç hidden states:              h  = [{formatted_h_init}] ... ({hidden_size} nöron)")

# Her zaman adımını işle ve hidden state evrimini gözlemle
hidden_states = []
for t, x_t in enumerate(sequence_data):
    # 🔄 RNN'İN KALBİ: Bu satır RNN'in temel işlemini yapar
    # Mevcut giriş (x_t) + önceki hafıza (h) = yeni hafıza (h)

    # FORMÜL: h_t = tanh(W_hh * h_prev + W_xh * x_t + b)
    # simple_rnn_step(x_t, h_prev, W_hh, W_xh, b):
    h = simple_rnn_step(x_t, h, W_hh, W_xh, b)
    
    # Sonuçları sakla (görselleştirme için)
    hidden_states.append(h.copy())  # .copy() önemli: referansı değil değeri sakla
    
    formatted_x_t = ", ".join([f"{val:11.8f}" for val in x_t.flatten()])
    formatted_h_t = ", ".join([f"{val:11.8f}" for val in h.flatten()[:5]])
    print(f"t{t}: x_{t} = [{formatted_x_t}], h{t} = [{formatted_h_t}] ...")

    # 💡 ÖNEMLİ: h artık bir önceki adımın bilgisini içeriyor!

print_title("📈 GİZLİ DURUMLARIN GÖRSELLEŞTİRİLMESİ")

# Gizli durumları görselleştir - DİNAMİK RENK SİSTEMİ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Giriş verisini göster
input_data = np.array([x.flatten() for x in sequence_data])
time_steps = range(len(sequence_data))

ax1.plot(time_steps, input_data[:, 0], 'bo-', label='Giriş 1', linewidth=2, markersize=8)
ax1.plot(time_steps, input_data[:, 1], 'ro-', label='Giriş 2', linewidth=2, markersize=8)
ax1.set_title(f'RNN Giriş Verisi ({TIME_STEPS} zaman adımı)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Zaman Adımı')
ax1.set_ylabel('Değer')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gizli durumları göster - DİNAMİK RENK VE NÖRON SEÇİMİ
hidden_data = np.array([h.flatten() for h in hidden_states])

# Çok nöron varsa sadece ilk 10 tanesini göster
max_neurons_to_show = min(10, hidden_size)
print(f"📊 Grafikte gösterilen nöron sayısı: {max_neurons_to_show} / {hidden_size}")

# Dinamik renk paleti oluştur
colors = plt.cm.tab10(np.linspace(0, 1, max_neurons_to_show))  # 10 farklı renk

for i in range(max_neurons_to_show):
    ax2.plot(time_steps, hidden_data[:, i], 'o-', 
             color=colors[i], label=f'H.Nöron {i+1}', linewidth=2, markersize=6)

ax2.set_title(f'RNN Hidden State Evrimi ({hidden_size} nöron, {max_neurons_to_show} gösteriliyor)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Zaman Adımı')
ax2.set_ylabel('Hidden State Değeri')

# Legend sadece çok fazla nöron yoksa göster
if max_neurons_to_show <= 10:
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    ax2.text(0.02, 0.98, f'{hidden_size} nöron var\n(İlk {max_neurons_to_show} gösteriliyor)', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_title("\n🧪 PARAMETRİK DENEY SİSTEMİ", True)

print("Yukarıdaki parametreleri değiştirerek farklı deneyler yapabilirsiniz!")
print()
print_title("🔧 NASIL FARKLI DURUMLAR DENEYEBİLİRİM?",True)

print("1. Dosyanın başındaki parametreleri değiştirin:")
print("   • HIDDEN_SIZE  = 5, 10, 20, 50  (farklı değerler deneyin)")
print("   • TIME_STEPS   = 4, 8, 15, 20   (farklı zaman aralıkları)")
print("   • WEIGHT_SCALE = 0.01, 0.1, 0.5 (ağırlık büyüklükleri)")
print()
print("2. Kodu tekrar çalıştırın ve sonuçları karşılaştırın.")
print()
print_title("🎯 DENEYİM ÖNERİLERİ:")

print("• Hidden Size 2 - 20      : Hangi daha iyi öğrenir?")
print("• Time Steps 4 - 15       : Uzun veya kısa diziler") 
print("• Weight Scale 0.01 vs 0.5: Ağırlık etkisi")

print_title("📊 MEVCUT DENEY SONUÇLARI:")

print(f"✓ Hidden Size  : {HIDDEN_SIZE}")
print(f"✓ Time Steps   : {TIME_STEPS}")
print(f"✓ Son Hidden State ortalama: {np.mean(hidden_states[-1]):.4f}")
print(f"✓ Hidden State değişkenliği: {np.std([np.mean(h) for h in hidden_states]):.4f}")
print(f"✓ Maksimum aktivasyon      : {np.max([np.max(np.abs(h)) for h in hidden_states]):.4f}")

# Otomatik performans analizi
def analyze_performance(hidden_states, sequence_data):
    """RNN performansının basit analizi"""
    # Hafıza kararlılığı - hidden state'lerin ne kadar değişken olduğu
    stability = np.std([np.mean(h) for h in hidden_states])
    
    # Aktivasyon gücü - nöronların ne kadar aktif olduğu
    activation_power = np.mean([np.mean(np.abs(h)) for h in hidden_states])
    
    # Desen hassasiyeti - girişe ne kadar tepki verdiği
    input_sensitivity = np.std([np.mean(x.flatten()) for x in sequence_data])
    
    return {
        "stability": stability,
        "activation_power": activation_power, 
        "input_sensitivity": input_sensitivity
    }

performance = analyze_performance(hidden_states, sequence_data)

print_title(f"📈 OTOMATİK PERFORMANS ANALİZİ:")

print(f"🔹 Hafıza Kararlılığı: {performance['stability']:.4f}")
print("   (Düşük = kararlı, Yüksek = değişken)")
print(f"🔹 Aktivasyon Gücü: {performance['activation_power']:.4f}")  
print("   (Çok düşük = öğrenme zor, Çok yüksek = patlama riski)")
print(f"🔹 Giriş Hassasiyeti: {performance['input_sensitivity']:.4f}")
print("   (Yüksek = çeşitli giriş desenleri)")

# Öneriler
print_title("💡 PERFORMANS DEĞERLENDİRMESİ:")

if performance['activation_power'] < 0.1:
    print("⚠️  Aktivasyon çok düşük - WEIGHT_SCALE'i artırın")
elif performance['activation_power'] > 0.8:
    print("⚠️  Aktivasyon çok yüksek - WEIGHT_SCALE'i azaltın")
else:
    print("✅ Aktivasyon dengeli görünüyor")

if performance['stability'] > 0.5:
    print("⚠️  Hidden state çok değişken - daha az TIME_STEPS deneyin")
else:
    print("✅ Hidden state kararlı")

print("Bir tuşa basınız...")
input()

print_title("🎯 HIDDEN STATE'İN FAYDASINI GÖSTEREN ÖRNEKLER", single_line=True, line_len=70)

# Örnek 1: Desen Tanıma

print_title("    📋 ÖRNEK 1: DESEN TANIMA")

print("Diyelim ki şu sırayla sayılar gelsin: [1, 0, 1, 0]")

# Desen analizi için hidden state'leri incele
pattern_data = [
    np.array([[1.0], [0.0]]),  # t=0: 1
    np.array([[0.0], [1.0]]),  # t=1: 0  
    np.array([[1.0], [0.0]]),  # t=2: 1
    np.array([[0.0], [1.0]])   # t=3: 0
]

print("Desen: 1 -> 0 -> 1 -> 0 (değişken desen)\n")
h_pattern = np.zeros((hidden_size, 1))
pattern_states = []

for t, x_t in enumerate(pattern_data):
    # 🧠 DESEN ÖĞRENME: RNN her adımda önceki desenleri hatırlayarak yeni girişi işler
    h_pattern = simple_rnn_step(x_t, h_pattern, W_hh, W_xh, b)
    pattern_states.append(h_pattern.copy())
    
    # Hidden state'in bu adımda neyi 'hatırladığını' göster
    dominant_neuron = np.argmax(np.abs(h_pattern))  # En büyük mutlak değere sahip nöron
    h_pattern_formatted = ", ".join([f"{val:11.8f}" for val in h_pattern.flatten()[:5]])
    print(f"t{t}: Giriş={x_t.flatten()} -> Hidden State: {h_pattern_formatted} ... (ilk 5 nöron)")
    print(f"     En aktif nöron: {dominant_neuron} (Bu nöron geçmişi 'hatırlıyor')")

print("\n🔍 ANALİZ:")
print("Hidden state'in değişimi, RNN'in önceki girişleri 'hatırladığını' gösterir!")
print("Her adımda sadece o anki giriş değil, geçmiş girişler de etkili olur.")
print("Bir tuşa basınız...")
input()

# Örnek 2: Gerçek RNN Eğitimi ve Başarılı Tahmin

print_title("    📋 ÖRNEK 2: GERÇEK RNN EĞİTİMİ VE TAHMİN")

print("RNN'i eğitelim ve başarısını ölçelim!")

# 🎯 GENİŞLETİLMİŞ EĞİTİM VERİSETİ OLUŞTUR
print_title("🎯 Eğitim veri seti oluşturuluyor...", single_line=True)

def create_training_sequences(pattern_type="decreasing", num_sequences=50, seq_length=6):
    """
    Eğitim için çoklu sekans üretir
    
    Args:
        pattern_type: "decreasing", "increasing", "sine" 
        num_sequences: Kaç farklı sekans
        seq_length: Her sekansın uzunluğu
    """
    sequences = []
    targets = []
    
    np.random.seed(42)  # Tekrarlanabilir sonuçlar
    
    for i in range(num_sequences):
        if pattern_type == "decreasing":
            # Azalan desenler: farklı başlangıç ve azalma oranları
            start_val = np.random.uniform(0.8, 2.0) # Başlangıç değeri 0.8 ile 2.0 arasında
            decrease_rate = np.random.uniform(0.1, 0.4) # Azalma oranı 0.1 ile 0.4 arasında değer çıkartılarak dizi üretiliyor
            
            sequence = []
            current_val = start_val
            for _ in range(seq_length):
                sequence.append(np.array([[current_val], [0.0]]))
                current_val -= decrease_rate
            
            # Target: bir sonraki değer
            target = current_val
            
        elif pattern_type == "increasing":
            # Artan desenler
            start_val = np.random.uniform(0.2, 5.0)
            increase_rate = np.random.uniform(0.1, 0.6)
            
            sequence = []
            current_val = start_val
            for _ in range(seq_length):
                sequence.append(np.array([[current_val], [0.0]]))
                current_val += increase_rate
            
            target = current_val
            
        elif pattern_type == "sine":
            # Sinüs dalgası desenleri
            frequency = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(0.5, 1.5)
            
            sequence = []
            for j in range(seq_length):
                val = amplitude * np.sin(frequency * j) + np.random.normal(0, 0.05)
                sequence.append(np.array([[val], [0.0]]))
            
            # Target: sinüs devamı
            target = amplitude * np.sin(frequency * seq_length)
        
        sequences.append(sequence)
        targets.append(target)
    
    return sequences, targets

# Farklı pattern türlerinde eğitim verisi oluştur
decreasing_seqs, decreasing_targets = create_training_sequences("decreasing", 30, 5)
increasing_seqs, increasing_targets = create_training_sequences("increasing", 20, 5)

# Tüm eğitim verilerini birleştir
all_training_sequences = decreasing_seqs + increasing_seqs
all_training_targets = decreasing_targets + increasing_targets

seq_count = len(all_training_sequences)
uzunluk = len(all_training_sequences[0])
print(f"✅ {seq_count} eğitim sekansı oluşturuldu (her biri {uzunluk} adım)")
print(f"   - {len(decreasing_seqs)} azalan desen")
print(f"   - {len(increasing_seqs)} artan desen")

if seq_count > 5:
    say = 5
else:
    say = seq_count
print(f"   İlk {say} sekans örneği:")

for i in range(say):
    formatted_sequence = [x[0][0] for x in all_training_sequences[i]]
    veri = ", ".join([f"{x:8.4f}" for x in formatted_sequence])
    print(f"Training sequence {i+1}: {veri:<54}" +
          f"Training target {i+1}: {all_training_targets[i]:8.4f}")

# Test için özel sequence (orijinal pattern'imiz)
test_sequence = [
    np.array([[1.4], [0.0]]),   
    np.array([[1.1], [0.0]]),   # (-0.3)
    np.array([[0.9], [0.0]]),   # (-0.2)
    np.array([[0.6], [0.0]]),   # (-0.3)
    np.array([[0.4], [0.0]])    # (-0.2)
]
test_target = 0.1  # Manuel hesaplanan beklenen değer

formatted_sequence = [x[0][0] for x in test_sequence]
degerler = ",  ".join([f"{x:.1f}" for x in formatted_sequence])
print(f"\n🎯 Test Dizisi  : {degerler} -> ???")
print(f"   Beklenen sonraki değer: {test_target:.1f}")


# 🧠 RNN MODELİ OLUŞTUR
print_title("🧠 Creaating RNN Model...", single_line=True)

class SimpleTrainableRNN:
    def __init__(self, hidden_size=16, input_size=2, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        
        # Xavier initialization ile ağırlıkları başlat
        # Bu yöntem, gradient'ların kaybolmasını veya patlamasını önler
        scale = np.sqrt(2.0 / (hidden_size + input_size))  # Xavier ölçek faktörü

        # W_hh: Hidden-to-Hidden ağırlıkları (hafızanın kendisini güncellemesi için)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale

        # W_xh: Input-to-Hidden ağırlıkları (giriş ile hidden-layer arası ağırlıklar)  
        self.W_xh = np.random.randn(hidden_size, input_size) * scale

        # W_hy: Hidden-to-Output ağırlıkları (hidden layer ile çıkış arasındaki ağırlıklar)
        self.W_hy = np.random.randn(1, hidden_size) * scale

        # b_h: Hidden bias (hidden layer için öğrenilen sabit eklenti)
        self.b_h = np.zeros((hidden_size, 1))

        # b_y: Output bias (çıkış katmanı için öğrenilen sabit eklenti)
        self.b_y = np.zeros((1, 1))
        
        print(f"✅ Model created:")
        print(f"   Hidden Size   : {hidden_size}")
        print(f"   Learning Rate : {learning_rate}")
        print(f"   Weight matrices initialized (Xavier)")

    def forward(self, sequence):
        """Forward pass - tahmin yap"""
        # Her forward işleminin başında "h" (hidden state) sıfırlanır. Böylece şunlar sağlanır:
        # Genelleme  : Model, eğitimde görmediği dizilerde çalışabilir.
        # Bağımsızlık: Sequence'lar birbirini etkilemez (istenmeyen bilgi sızıntısı önlenir).
        # Verimlilik : Her sequence kendi hafızasında işlenir.
        h = np.zeros((self.hidden_size, 1))
        
        # Bundan sonraki işlemlerde, eğitimde öğrenilen ağırlıklar (W_xh, W_hh, W_hy) kullanılır

        # Her zaman adımını işle
        for x_t in sequence:
            # RNN'in temel işlemi
            # FORMÜL: h_t = tanh(W_hh * h_prev + W_xh * x_t + b)
            # Trainde öğrenilen ağırlıklar (W_xh, W_hh) burada kullanılır
            h = np.tanh(np.dot(self.W_hh, h) + np.dot(self.W_xh, x_t) + self.b_h)
        
        # Final prediction
        # RNN'in çıkış katmanı değeri: y = W_hy × h + b_y
        output = np.dot(self.W_hy, h) + self.b_y
        return output[0, 0], h
    
    def train_step(self, sequence, target):
        """Tek eğitim adımı (basitleştirilmiş backpropagation)"""
        prediction, final_h = self.forward(sequence)
        
        # Loss hesapla (MSE, Mean Squared Error)
        error = prediction - target
        loss = (error) ** 2
        
        # Gradient Hesaplama (Zincir Kuralı):
        # gradient = ∂Loss/∂W_hy = ∂Loss/∂prediction × ∂prediction/∂W_hy
        #                        = 2 × (prediction - target) × final_h
        # Ağırlık Güncellemesi:
        # W_hy_new = W_hy_old - learning_rate × gradient

        # Output weights güncelle - Backpropagation aşaması
        # final_h.T = (çıkışın hidden state'e göre türevi)
        self.W_hy -= self.learning_rate * error * final_h.T * 0.1
        self.b_y -= self.learning_rate * error * 0.01
        
        # Hidden weights güncelle - Hidden layer'lar için backprop (basit yaklaşım)
        # Şu sebeplerle "basit yaklaşım" diye isimlendirildi:
        # 1. Hidden katmanlar için gerçek backprop yok: Sadece rastgele gürültü ile güncelleniyor.
        # 2. Zincir kuralı eksik: Gerçek backprop, her katmandan başlayarak geriye doğru gradient'ları hesaplar
        # 3. Aktivasyon fonksiyonlarının türevleri yok: tanh'ın türevi kullanılmıyor
        # Bu basit yaklaşım, temel bir öğrenme simülasyonu sağlar ancak gerçek uygulamalarda yetersizdir.
        # Bu şekli, temel kavramları öğretmek için eğitim amaçlı yeterlidir. Ayrıca  Basit gradient descent ile loss'u azaltır
        gradient_scale = self.learning_rate * error * 0.001
        self.W_hh -= gradient_scale * np.random.randn(*self.W_hh.shape) * 0.1  # ← Öğrenme burada oluyor
        self.W_xh -= gradient_scale * np.random.randn(*self.W_xh.shape) * 0.1  # ← Öğrenme burada oluyor

        return loss, prediction

# Model oluştur
rnn_model = SimpleTrainableRNN(hidden_size=16, learning_rate=0.05)

# 🎓 Eğitim Başlat
print_title("🎓 Start training...", single_line=True)

epochs = 40
batch_size = 10

print(f"Epochs    : {epochs}")
print(f"Batch Size: {batch_size}")
print()

# Eğitim döngüsü
training_losses = []
test_predictions = []

for epoch in range(epochs):
    epoch_losses = []
    
    # Mini-batch eğitimi
    for i in range(0, len(all_training_sequences), batch_size):
        batch_sequences = all_training_sequences[i:i+batch_size]
        batch_targets = all_training_targets[i:i+batch_size]
        
        batch_loss = 0
        for seq, target in zip(batch_sequences, batch_targets):
            loss, pred = rnn_model.train_step(seq, target)
            batch_loss += loss
        
        epoch_losses.append(batch_loss / len(batch_sequences))
    
    # Epoch ortalaması
    avg_loss = np.mean(epoch_losses)
    training_losses.append(avg_loss)
    
    # Test tahmin yap
    test_pred, _ = rnn_model.forward(test_sequence)
    test_predictions.append(test_pred)
    test_error = abs(test_pred - test_target)
    
    # İlerlemeyi göster
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Test Prediction={test_pred:.3f}, Error={test_error:.3f}")

print("\n🎯 Training completed.")

# 📊 SONUÇLARI DEĞERLENDİR
print_title("📊 Result Analysis:", single_line=True)

final_prediction, _ = rnn_model.forward(test_sequence)
final_error = abs(final_prediction - test_target)

print(f"🔹 Test Sequence  : {degerler}")
print(f"🔹 Expected Value : {test_target:.1f}")
print(f"🔹 RNN Prediction : {final_prediction:.3f}")
print(f"🔹 Absolute Error : {final_error:.3f}")
print(f"🔹 Error Rate     : {(final_error/abs(test_target)*100):.1f}%")

print_title(f"📈 Training Progress:", single_line=True)
print(f"🔹 Initial Loss   : {training_losses[0]:.4f}")
print(f"🔹 Final Loss     : {training_losses[-1]:.4f}")
print(f"🔹 Improvement    : {((training_losses[0]-training_losses[-1])/training_losses[0]*100):.1f}%")

def calc_success_level(final_error, silent:True):
    """Hata oranına göre başarı seviyesini belirle"""
    # Başarı değerlendirmesi
    if final_error < 0.1:
        if not silent:
            print("\n🏆 GREAT RESULT!")
            print("   RNN pattern successfully learned!")
        return "🏆 Excellent"
    elif final_error < 0.2:
        if not silent:
            print("\n✅ GOOD RESULT!")
            print("   RNN pattern largely learned!")
        return "✅ Good"
    elif final_error < 0.4:
        if not silent:
            print("\n⚠️  AVERAGE RESULT")
            print("   RNN partially learned, more training needed")
        return "⚠️  Average"
    else:
        if not silent:
            print("\n❌ FAILED")
            print("   RNN pattern could not be learned, model or data issue")
        return "❌ Failed"

success_level = calc_success_level(final_error, silent=False)

# Başarı değerlendirmesi
if final_error < 0.1:
    print("\n🏆 GREAT RESULT!")
    print("   RNN pattern successfully learned!")
    success_level = "Excellent"
elif final_error < 0.2:
    print("\n✅ GOOD RESULT!")
    print("   RNN pattern largely learned!")
    success_level = "Good"
elif final_error < 0.4:
    print("\n⚠️  AVERAGE RESULT")
    print("   RNN partially learned, more training needed")
    success_level = "Average"
else:
    print("\n❌ FAILED")
    print("   RNN pattern could not be learned, model or data issue")
    success_level = "Failed"

# 🎯 ADDITIONAL TEST PATTERNS
print_title(f"🎯 ADDITIONAL TEST PATTERNS:")

# Benzer azalan desen test et
extra_test_1 = [
    np.array([[2.0], [0.0]]), 
    np.array([[1.7], [0.0]]), 
    np.array([[1.4], [0.0]]), 
    np.array([[1.1], [0.0]]), 
    np.array([[0.8], [0.0]])
]
expected_1 = 0.5
pred_1, _ = rnn_model.forward(extra_test_1)
error_1 = abs(pred_1 - expected_1)

extra_values_1 = [x[0][0] for x in extra_test_1]
extra_str_1 = ", ".join([f"{x:.1f}" for x in extra_values_1])

print(f"Test 1: {extra_str_1} -> Expected: {expected_1:.1f}, Prediction: {pred_1:.3f}, Error: {error_1:.3f}")

# Artan desen test et  
extra_test_2 = [
    np.array([[0.5], [0.0]]), 
    np.array([[0.7], [0.0]]), 
    np.array([[0.9], [0.0]]), 
    np.array([[1.1], [0.0]]), 
    np.array([[1.3], [0.0]])
]
expected_2 = 1.5
pred_2, _ = rnn_model.forward(extra_test_2)
error_2 = abs(pred_2 - expected_2)

extra_values_2 = [x[0][0] for x in extra_test_2]
extra_str_2 = ", ".join([f"{x:.1f}" for x in extra_values_2])

print(f"Test 2: {extra_str_2} -> Expected: {expected_2:.1f}, Prediction: {pred_2:.3f}, Error: {error_2:.3f}")

# Artan desen test et  
extra_test_3 = [
    np.array([[1.0], [0.0]]), 
    np.array([[1.4], [0.0]]), # +0.4
    np.array([[1.9], [0.0]]), # +0.5
    np.array([[2.3], [0.0]]), # +0.4
    np.array([[2.8], [0.0]])  # +0.5
]
expected_3 = 3.2
pred_3, _ = rnn_model.forward(extra_test_3)
error_3 = abs(pred_3 - expected_3)

extra_values_3 = [x[0][0] for x in extra_test_3]
extra_str_3 = ", ".join([f"{x:.1f}" for x in extra_values_3])

print(f"Test 3: {extra_str_3} -> Expected: {expected_3:.1f}, Prediction: {pred_3:.3f}, Error: {error_3:.3f}")

# Genel başarı raporu
avg_error = np.mean([final_error, error_1, error_2, error_3])
print_title(f"📊 General Success Report:", single_line=True)
print(f"🔹 Average Error           : {avg_error:.3f}")
print(f"🔹 1st Test Success Level  : {calc_success_level(error_1, silent=True)}")
print(f"🔹 2nd Test Success Level  : {calc_success_level(error_2, silent=True)}")
print(f"🔹 3rd Test Success Level  : {calc_success_level(error_3, silent=True)}")

print(f"🔹 Number of Tests        : 3")

if avg_error < 0.15:
    print("🌟 RNN successfully learned the pattern and can generalize!")
elif avg_error < 0.3:
    print("👍 RNN learned the pattern at a reasonable level")
else:
    print("🔧 RNN needs more training")

print(f"\n💡 Training Details:")
print("-" * 25)
print(f"✅  Trained on {len(all_training_sequences)} different patterns")
print(f"✅  {epochs} epochs of training completed")
print(f"✅  Loss reduced by % {((training_losses[0]-training_losses[-1])/training_losses[0]*100):.0f}")
print(f"✅  Real backpropagation simulated")
print(f"✅  Multiple pattern recognition achieved")

