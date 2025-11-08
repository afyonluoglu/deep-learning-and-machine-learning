"""
🔤 METİN ÜRETİMİ İLE RNN ÖĞRENİMİ
==================================

Bu dosya RNN'leri metin üretimi ile öğretir.
Character-level language model kullanarak RNN'lerin
nasıl sequential pattern'leri öğrendiğini gösterir.

Öğreneceğiniz konular:
1. Character-level text processing
2. One-hot encoding
3. Text generation with RNN
4. Temperature sampling
5. Model creativity control
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import random

print("=" * 60)
print("🔤 METİN ÜRETİMİ İLE RNN ÖĞRENİMİ")
print("=" * 60)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("ÖRNEK METİN HAZIRLIĞI")

# Türkçe örnek metin - Yunus Emre'den
sample_text = """
miskince gel ey canlar bu canlara gaş olam
dost yoluna varıcak yol dostlara gaş olam
gerek yüriyem tenha göçem ıssız yerde
aşk elinden ölicem dostlara gaş olam

bu gönül şehr-i cananın kuşu iken
nice olur ayrı düşem dostlara gaş olam
bu gönül dost darağında bülbül iken
nice olur karakuşa yar keremin gaş olam

bu başım kurbanı olsun dostların hakkına
severem ben anların yüzü suyu hakkına
uş gitmezem el çekip dostların yolundan
tut elim yunus miskin dostlara gaş olam
""".strip()

# Metni temizle ve küçük harfe çevir
text = sample_text.lower()
text = ''.join([char for char in text if char.isalpha() or char.isspace() or char in '.,!?'])

print("📖 Örnek metin:")
print(text[:200] + "...")
print(f"\n📊 Metin istatistikleri:")
print(f"   Toplam karakter sayısı: {len(text)}")
print(f"   Benzersiz karakter sayısı: {len(set(text))}")

# Karakter setini oluştur
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

n_chars = len(chars)
n_vocab = len(chars)

print(f"\n🔤 Karakter seti: {chars}")
print(f"📊 Vocabulary boyutu: {n_vocab}")

print_section("VERİ HAZIRLAMA VE SEQUENCE OLUŞTURMA")

# Sequence parametreleri
seq_length = 40  # 40 karakterlik sequence'ler kullan
step = 3         # Her 3 karakterde bir yeni sequence başlat

print(f"⚙️ Parametreler:")
print(f"   Sequence uzunluğu: {seq_length}")
print(f"   Adım boyutu: {step}")

# Sequence'leri oluştur
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

print(f"📊 Toplam {len(sequences)} sequence oluşturuldu")

# İlk birkaç sequence'i göster
print(f"\n🔍 İlk 3 sequence örneği:")
for i in range(3):
    print(f"  Sequence {i+1}: '{sequences[i]}'")
    print(f"  Sonraki char: '{next_chars[i]}'")
    print()

# One-hot encoding için veri hazırla
print("🔢 One-hot encoding yapılıyor...")

X = np.zeros((len(sequences), seq_length, n_vocab), dtype=np.bool_)
y = np.zeros((len(sequences), n_vocab), dtype=np.bool_)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

print(f"✅ Encoding tamamlandı!")
print(f"   X shape: {X.shape} (örnekler, sequence_length, vocab_size)")
print(f"   y shape: {y.shape} (örnekler, vocab_size)")

print_section("RNN MODELİ TASARIMI")

print("🏗️ Character-level RNN modeli oluşturuluyor...")

model = Sequential([
    LSTM(128, input_shape=(seq_length, n_vocab), return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(n_vocab, activation='softmax')
])

# Learning rate scheduler
def lr_schedule(epoch):
    """Epoch'a göre learning rate ayarlar"""
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.005
    else:
        return 0.001

# Model derle
optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print("✅ Model hazırlandı!")
print(f"\n📋 MODEL ÖZETİ:")
model.summary()

print_section("METİN ÜRETME FONKSİYONLARI")

def sample_with_temperature(predictions, temperature=1.0):
    """
    Temperature ile sampling yapar
    
    Args:
        predictions: Model tahminleri (probability distribution)
        temperature: Yaratıcılık kontrolü
                    - Düşük (0.2-0.5): Muhafazakar, tekrarlanan
                    - Orta (0.8-1.2): Dengeli
                    - Yüksek (1.5-2.0): Yaratıcı, rastgele
    
    Returns:
        Seçilen karakterin index'i
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, length=200, temperature=1.0):
    """
    Verilen seed text ile yeni metin üretir
    
    Args:
        model: Eğitilmiş RNN modeli
        seed_text: Başlangıç metni
        length: Üretilecek karakter sayısı
        temperature: Yaratıcılık seviyesi
    
    Returns:
        Üretilen metin
    """
    generated = seed_text.lower()
    seed = seed_text.lower()
    
    for i in range(length):
        # Son seq_length karakteri al
        if len(seed) < seq_length:
            # Padding with spaces
            padded_seed = ' ' * (seq_length - len(seed)) + seed
        else:
            padded_seed = seed[-seq_length:]
        
        # One-hot encode
        x_pred = np.zeros((1, seq_length, n_vocab))
        for t, char in enumerate(padded_seed):
            if char in char_to_int:
                x_pred[0, t, char_to_int[char]] = 1
        
        # Tahmin yap
        predictions = model.predict(x_pred, verbose=0)[0]
        
        # Temperature ile sampling
        next_index = sample_with_temperature(predictions, temperature)
        next_char = int_to_char[next_index]
        
        generated += next_char
        seed = seed + next_char
    
    return generated

# Farklı temperature'ları demo et
def demonstrate_temperature():
    """Farklı temperature değerlerini gösterir"""
    
    print("🌡️ TEMPERATURE ÖRNEKLERİ:")
    print("-" * 40)
    
    seed = "miskince gel"
    temperatures = [0.2, 0.5, 1.0, 1.5, 2.0]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        if temp <= 0.5:
            print("   Beklenen: Muhafazakar, güvenli seçimler")
        elif temp <= 1.2:
            print("   Beklenen: Dengeli, okunabilir")
        else:
            print("   Beklenen: Yaratıcı, riskli")
        
        # Bu aşamada model henüz eğitilmediği için demo çıktısı
        print(f"   Örnek: '{seed}' + [model üretimi]")

demonstrate_temperature()

print_section("MODEL EĞİTİMİ")

print("🚀 Model eğitimi başlıyor...")

# Training callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)

class TextGenerationCallback(tf.keras.callbacks.Callback):
    """Her epoch sonunda örnek metin üretir"""
    
    def __init__(self, seed_text="miskince"):
        self.seed_text = seed_text
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Her 5 epoch'ta bir
            print(f"\n📝 Epoch {epoch+1} - Örnek üretim:")
            generated = generate_text(self.model, self.seed_text, 100, temperature=0.8)
            print(f"   '{generated[:80]}...'")

# Callbacks
text_callback = TextGenerationCallback()

# Model eğitimi
history = model.fit(
    X, y,
    batch_size=64,
    epochs=30,
    callbacks=[lr_scheduler, text_callback],
    verbose=1
)

print("✅ Eğitim tamamlandı!")

print_section("METİN ÜRETME DENEYİMLERİ")

print("🎨 Farklı temperature'larla metin üretimi:")

seed_texts = ["miskince gel", "bu gönül", "dost yoluna"]
temperatures = [0.3, 0.8, 1.5]

results = []

for seed in seed_texts:
    print(f"\n🌱 Seed: '{seed}'")
    print("-" * 30)
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        generated = generate_text(model, seed, 150, temperature=temp)
        print(f"📝 Üretilen metin:")
        print(f"   {generated}")
        
        results.append({
            'seed': seed,
            'temperature': temp,
            'generated': generated
        })

print_section("METİN KALİTE ANALİZİ")

def analyze_text_quality(generated_text, original_chars):
    """Üretilen metnin kalitesini analiz eder"""
    
    # Karakter çeşitliliği
    unique_chars = len(set(generated_text))
    diversity = unique_chars / len(original_chars)
    
    # Kelime sayısı
    words = generated_text.split()
    word_count = len(words)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Tekrar analizi
    char_counts = {}
    for char in generated_text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # En sık kullanılan karakterler
    most_common = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'length': len(generated_text),
        'unique_chars': unique_chars,
        'diversity': diversity,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'most_common_chars': most_common
    }

print("📊 METİN KALİTE ANALİZİ:")
print("-" * 30)

# Orijinal metin analizi
original_analysis = analyze_text_quality(text, chars)
print(f"📖 Orijinal metin:")
print(f"   Uzunluk: {original_analysis['length']}")
print(f"   Benzersiz karakter: {original_analysis['unique_chars']}")
print(f"   Çeşitlilik: {original_analysis['diversity']:.3f}")
print(f"   Kelime sayısı: {original_analysis['word_count']}")
print(f"   Ort. kelime uzunluğu: {original_analysis['avg_word_length']:.1f}")

# Üretilen metinlerin analizi
print(f"\n📝 Üretilen metinler:")
for i, result in enumerate(results[:3]):  # İlk 3 sonuç
    analysis = analyze_text_quality(result['generated'], chars)
    print(f"\n   Örnek {i+1} (T={result['temperature']}):")
    print(f"     Çeşitlilik: {analysis['diversity']:.3f}")
    print(f"     Kelime sayısı: {analysis['word_count']}")
    print(f"     Ort. kelime uzunluğu: {analysis['avg_word_length']:.1f}")

print_section("GÖRSELLEŞTİRME VE ANALİZ")

# Training loss görselleştirmesi
plt.figure(figsize=(15, 10))

# Loss grafiği
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], 'b-', linewidth=2)
plt.title('📉 Model Eğitim Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Karakter frekans analizi
plt.subplot(2, 2, 2)
char_freq = {}
for char in text:
    char_freq[char] = char_freq.get(char, 0) + 1

# En sık 10 karakter
top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]
chars_list, freqs_list = zip(*top_chars)

plt.bar(range(len(chars_list)), freqs_list, alpha=0.7)
plt.title('📊 En Sık Kullanılan Karakterler', fontweight='bold')
plt.xlabel('Karakterler')
plt.ylabel('Frekans')
plt.xticks(range(len(chars_list)), chars_list)
plt.grid(True, alpha=0.3)

# Temperature karşılaştırması
plt.subplot(2, 2, 3)
temp_values = [0.3, 0.8, 1.5]
diversities = []

for temp in temp_values:
    # Her temperature için diversity hesapla
    sample_text = generate_text(model, "miskince", 200, temperature=temp)
    analysis = analyze_text_quality(sample_text, chars)
    diversities.append(analysis['diversity'])

plt.plot(temp_values, diversities, 'ro-', linewidth=2, markersize=8)
plt.title('🌡️ Temperature vs Çeşitlilik', fontweight='bold')
plt.xlabel('Temperature')
plt.ylabel('Karakter Çeşitliliği')
plt.grid(True, alpha=0.3)

# Model karmaşıklığı
plt.subplot(2, 2, 4)
layers = ['LSTM-1', 'Dropout-1', 'LSTM-2', 'Dropout-2', 'Dense']
params = [128*4*128, 0, 128*4*128, 0, 128*n_vocab]  # Yaklaşık parametre sayıları

plt.bar(layers, params, alpha=0.7, color=['blue', 'gray', 'blue', 'gray', 'green'])
plt.title('🏗️ Model Katman Parametreleri', fontweight='bold')
plt.xlabel('Katmanlar')
plt.ylabel('Parametre Sayısı')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_section("İNTERAKTİF METİN ÜRETME")

def interactive_text_generation():
    """Kullanıcı ile interaktif metin üretme"""
    
    print("🎮 İNTERAKTİF METİN ÜRETME")
    print("-" * 30)
    print("Önerilen seed'ler:")
    suggestions = ["miskince gel", "bu gönül", "dost yoluna", "severem ben"]
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. '{suggestion}'")
    
    print("\n💡 Kullanım örnekleri:")
    print("  - Düşük temperature (0.2-0.5): Güvenli, okunabilir")
    print("  - Orta temperature (0.8-1.2): Dengeli, yaratıcı")
    print("  - Yüksek temperature (1.5-2.0): Çok yaratıcı, riskli")
    
    # Demo için örnekler
    demo_examples = [
        {"seed": "miskince gel", "temp": 0.5, "length": 100},
        {"seed": "bu gönül", "temp": 1.0, "length": 120},
        {"seed": "dost yoluna", "temp": 1.5, "length": 80}
    ]
    
    print(f"\n🎯 DEMO ÖRNEKLERİ:")
    for i, example in enumerate(demo_examples, 1):
        print(f"\nÖrnek {i}:")
        print(f"  Seed: '{example['seed']}'")
        print(f"  Temperature: {example['temp']}")
        print(f"  Uzunluk: {example['length']}")
        
        generated = generate_text(model, example['seed'], example['length'], example['temp'])
        print(f"  Sonuç: '{generated[:60]}...'")

interactive_text_generation()

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu metin üretimi örneğinde öğrendikleriniz:")
print("  1. 🔤 Character-level text processing")
print("  2. 🔢 One-hot encoding ve vocabulary oluşturma")
print("  3. 🧠 RNN ile sequence modeling")
print("  4. 🌡️ Temperature sampling ve creativity control")
print("  5. 📊 Text quality analysis")
print("  6. 🎮 Interactive text generation")
print("")
print("💡 RNN'in metin üretimindeki yetenekleri:")
print("  ✅ Sequential pattern learning")
print("  ✅ Context-aware generation")
print("  ✅ Controllable creativity")
print("  ✅ Language modeling")
print("")
print("🎨 Temperature etkisi:")
print("  • Düşük temperature → Muhafazakar, tutarlı")
print("  • Yüksek temperature → Yaratıcı, riskli")
print("  • Optimal değer problem ve context'e bağlı")
print("")
print("🚀 İyileştirme önerileri:")
print("  1. Daha büyük dataset kullanın")
print("  2. Word-level modeling deneyin")
print("  3. Attention mechanism ekleyin")
print("  4. Beam search kullanın")
print("  5. Fine-tuning ile özelleştirin")
print("")
print("📚 Sonraki dosya: 08_sentiment_analysis.py")
print("RNN ile duygu analizi yapacağız!")

print("\n" + "=" * 60)
print("✅ METİN ÜRETİMİ ÖRNEĞİ TAMAMLANDI!")
print("=" * 60)