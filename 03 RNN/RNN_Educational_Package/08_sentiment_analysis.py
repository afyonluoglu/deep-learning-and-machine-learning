"""
💬 SENTIMENT ANALİZİ - RNN İLE DUYGU ANALİZİ
============================================

Bu dosya RNN ile duygu analizi yapmayı detaylıca açıklar.
Metin sınıflandırma, word embedding ve sequence modeling öğrenin.

Duygu Analizi Nedir?
- Metinlerdeki duyguları (pozitif/negatif) tespit etme
- Sosyal medya monitoring
- Müşteri geri bildirim analizi
- Ürün yorumu değerlendirme

RNN Avantajları:
- Sequential bilgiyi korur
- Variable length inputs
- Context-aware predictions
- Memory of past words
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Embedding, Dropout, 
                                   Bidirectional, GlobalMaxPooling1D, 
                                   Conv1D, MaxPooling1D, Flatten,
                                   Input, Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("💬 SENTIMENT ANALİZİ - RNN İLE DUYGU ANALİZİ")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("VERİ SETİ OLUŞTURMA VE HAZIRLIK")

def create_synthetic_sentiment_dataset():
    """Sentetik duygu analizi veri seti oluşturur"""
    
    print("📊 Sentetik duygu analizi veri seti oluşturuluyor...")
    
    # Pozitif cümleler
    positive_samples = [
        # Temel pozitif
        "This movie is absolutely fantastic and amazing!",
        "I love this product, it works perfectly.",
        "Great service and friendly staff!",
        "Excellent quality and fast delivery.",
        "Amazing experience, highly recommended!",
        "Perfect solution for my needs.",
        "Outstanding performance and great value.",
        "Wonderful customer support team.",
        "Brilliant idea and execution!",
        "Fantastic results exceeded expectations.",
        
        # Orta seviye pozitif
        "The movie was enjoyable and entertaining.",
        "Good product with decent features.",
        "Nice experience overall.",
        "Satisfied with the purchase.",
        "Works well for basic needs.",
        "Pleasant interface and design.",
        "Reasonable price and quality.",
        "Helpful and informative content.",
        "Smooth and reliable performance.",
        "User-friendly and intuitive.",
        
        # Güçlü pozitif
        "This is the best product ever made!",
        "Absolutely incredible and mind-blowing!",
        "Phenomenal quality and superb craftsmanship.",
        "Extraordinary service beyond expectations.",
        "Magnificent results and outstanding value.",
        "Spectacular performance and reliability.",
        "Exceptional experience that exceeded hopes.",
        "Brilliant innovation and perfect execution.",
        "Remarkable achievement and superior quality.",
        "Astonishing features and incredible value.",
        
        # Duygusal pozitif
        "This made my day so much better!",
        "I'm so happy with this purchase.",
        "Brings joy and happiness to life.",
        "Feel grateful for this experience.",
        "Makes me smile every time I use it.",
        "Brings peace of mind and comfort.",
        "Feels like a warm hug.",
        "Creates wonderful memories.",
        "Inspires confidence and positivity.",
        "Fills me with excitement and joy."
    ]
    
    # Negatif cümleler
    negative_samples = [
        # Temel negatif
        "This movie is terrible and boring.",
        "Worst product I have ever bought.",
        "Completely useless and waste of money.",
        "Awful service and rude staff.",
        "Poor quality and overpriced.",
        "Disappointing and frustrating experience.",
        "Broken and defective product.",
        "Slow and unreliable performance.",
        "Confusing and difficult to use.",
        "Not worth the money spent.",
        
        # Orta seviye negatif
        "The product has some issues.",
        "Not what I expected from reviews.",
        "Could be better in some aspects.",
        "Has limitations and drawbacks.",
        "Price is too high for quality.",
        "Interface could be more intuitive.",
        "Some features don't work properly.",
        "Delivery was delayed significantly.",
        "Customer support was unhelpful.",
        "Instructions were unclear and confusing.",
        
        # Güçlü negatif
        "Absolute disaster and complete failure!",
        "Horrible experience that ruined everything.",
        "Disgusting quality and terrible service.",
        "Completely broken and utterly useless.",
        "Nightmare experience and total waste.",
        "Pathetic attempt at product design.",
        "Catastrophic failure and poor execution.",
        "Abysmal performance and awful quality.",
        "Horrendous service and rude behavior.",
        "Dreadful experience from start to finish.",
        
        # Duygusal negatif
        "This ruined my entire day.",
        "I'm so disappointed and upset.",
        "Makes me angry and frustrated.",
        "Feels like a complete betrayal.",
        "Brings stress and anxiety.",
        "Made me feel stupid and confused.",
        "Causes headaches and irritation.",
        "Creates more problems than solutions.",
        "Leaves me feeling empty and sad.",
        "Brings nothing but trouble and pain."
    ]
    
    # Nötr cümleler (isteğe bağlı - 3-class için)
    neutral_samples = [
        "The product has standard features.",
        "It works as described in manual.",
        "Basic functionality is available.",
        "Meets minimum requirements.",
        "Standard quality and price.",
        "Regular performance and features.",
        "As expected from product description.",
        "Normal experience with no issues.",
        "Basic service and support.",
        "Standard delivery time and process.",
        "Product functions as intended.",
        "Adequate for basic use cases.",
        "Typical quality for this price range.",
        "Nothing special but gets job done.",
        "Standard features and performance.",
        "Meets basic expectations.",
        "Regular customer service experience.",
        "Standard packaging and delivery.",
        "Basic instructions provided.",
        "Functional but not exceptional."
    ]
    
    # Veri setini genişlet
    def expand_samples(samples, multiplier=3):
        expanded = []
        for sample in samples:
            # Orijinal
            expanded.append(sample)
            
            # Varyasyon 1: Noktalama değişiklikleri
            var1 = sample.replace('!', '.').replace('.', '!')
            if var1 != sample:
                expanded.append(var1)
            
            # Varyasyon 2: Büyük-küçük harf değişiklikleri
            if np.random.random() > 0.5:
                var2 = sample.lower()
                expanded.append(var2)
            
            # Varyasyon 3: Ekstra kelimeler
            if len(expanded) < len(samples) * multiplier:
                extras = ["really ", "very ", "quite ", "extremely ", "totally "]
                extra = np.random.choice(extras)
                var3 = extra + sample.lower()
                expanded.append(var3)
        
        return expanded[:len(samples) * multiplier]
    
    # Samples'ları genişlet
    positive_expanded = expand_samples(positive_samples, 4)
    negative_expanded = expand_samples(negative_samples, 4)
    neutral_expanded = expand_samples(neutral_samples, 2)
    
    # DataFrame oluştur
    all_texts = positive_expanded + negative_expanded + neutral_expanded
    all_labels = (['positive'] * len(positive_expanded) + 
                 ['negative'] * len(negative_expanded) + 
                 ['neutral'] * len(neutral_expanded))
    
    df = pd.DataFrame({
        'text': all_texts,
        'sentiment': all_labels
    })
    
    # Karıştır
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✅ Veri seti oluşturuldu:")
    print(f"   Toplam: {len(df)} örnek")
    print(f"   Positive: {len(positive_expanded)}")
    print(f"   Negative: {len(negative_expanded)}")
    print(f"   Neutral: {len(neutral_expanded)}")
    
    return df

# Veri seti oluştur
df = create_synthetic_sentiment_dataset()

print("\n📊 VERİ SETİ İNCELEMESİ:")
print("-" * 30)
print(df.head(10))
print(f"\nSentiment dağılımı:")
print(df['sentiment'].value_counts())

# Metin istatistikleri
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nMetin istatistikleri:")
print(f"Ortalama karakter sayısı: {df['text_length'].mean():.1f}")
print(f"Ortalama kelime sayısı: {df['word_count'].mean():.1f}")

print_section("METİN ÖN İŞLEME VE TEMİZLEME")

def comprehensive_text_preprocessing(texts):
    """Kapsamlı metin ön işleme"""
    
    print("🧹 Metin ön işleme başlıyor...")
    
    processed_texts = []
    
    for text in texts:
        # 1. Küçük harfe çevir
        text = text.lower()
        
        # 2. HTML etiketlerini temizle (varsa)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3. URL'leri temizle
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        
        # 4. Mention ve hashtag'leri temizle
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # 5. Sayıları normalize et
        text = re.sub(r'\d+', 'NUMBER', text)
        
        # 6. Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # 7. Noktalama işaretlerini koru ama normalize et
        text = text.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
        
        # 8. Başta ve sonda boşluk temizle
        text = text.strip()
        
        processed_texts.append(text)
    
    print(f"✅ {len(texts)} metin işlendi")
    return processed_texts

# Metinleri işle
processed_texts = comprehensive_text_preprocessing(df['text'].tolist())

print("\n🔍 ÖRNEK İŞLEME SONUÇLARI:")
print("-" * 40)
for i in range(5):
    print(f"Orijinal: {df['text'].iloc[i]}")
    print(f"İşlenmiş: {processed_texts[i]}")
    print()

print_section("TOKENIZATION VE SEQUENCE HAZIRLIĞI")

def prepare_sequences_for_rnn(texts, labels, max_vocab_size=10000, max_sequence_length=100):
    """RNN için sequence'leri hazırla"""
    
    print("🔤 Tokenization ve sequence hazırlığı...")
    
    # Tokenizer oluştur
    tokenizer = Tokenizer(num_words=max_vocab_size, 
                         oov_token='<OOV>',
                         filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    
    # Tokenizer'ı fit et
    tokenizer.fit_on_texts(texts)
    
    # Metinleri sequence'lere çevir
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, 
                                   maxlen=max_sequence_length,
                                   padding='post',
                                   truncating='post')
    
    # Label encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"✅ Sequence hazırlığı tamamlandı:")
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")
    print(f"   Max sequence length: {max_sequence_length}")
    print(f"   Number of classes: {len(label_encoder.classes_)}")
    print(f"   Classes: {label_encoder.classes_}")
    
    # İstatistikler
    sequence_lengths = [len(seq) for seq in sequences]
    print(f"   Ortalama sequence length: {np.mean(sequence_lengths):.1f}")
    print(f"   Max sequence length: {np.max(sequence_lengths)}")
    print(f"   Min sequence length: {np.min(sequence_lengths)}")
    
    return padded_sequences, encoded_labels, tokenizer, label_encoder

# Sequence'leri hazırla
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 50

X, y, tokenizer, label_encoder = prepare_sequences_for_rnn(
    processed_texts, 
    df['sentiment'].tolist(),
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH
)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Visualize sequence lengths
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
original_lengths = [len(text.split()) for text in processed_texts]
plt.hist(original_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('📊 Orijinal Metin Uzunlukları', fontweight='bold')
plt.xlabel('Kelime Sayısı')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
sequence_lengths = [len([w for w in seq if w != 0]) for seq in X]
plt.hist(sequence_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('📊 Sequence Uzunlukları', fontweight='bold')
plt.xlabel('Token Sayısı')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
sentiment_counts = pd.Series(y).value_counts()
colors = ['green', 'red', 'gray']
plt.pie(sentiment_counts.values, 
        labels=[label_encoder.classes_[i] for i in sentiment_counts.index],
        colors=colors[:len(sentiment_counts)],
        autopct='%1.1f%%',
        startangle=90)
plt.title('🥧 Sentiment Dağılımı', fontweight='bold')

plt.tight_layout()
plt.show()

print_section("RNN MODEL MİMARİLERİ KARŞILAŞTIRMASI")

def create_sentiment_models(vocab_size, embedding_dim=100, max_length=50, num_classes=3):
    """Farklı RNN mimarilerini oluşturur"""
    
    models = {}
    
    print("🏗️ Farklı RNN modelleri oluşturuluyor...")
    
    # 1. Simple RNN
    simple_rnn = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ], name='Simple_RNN')
    
    # 2. LSTM
    lstm_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ], name='LSTM')
    
    # 3. GRU
    gru_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GRU(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ], name='GRU')
    
    # 4. Bidirectional LSTM
    bidirectional_lstm = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ], name='Bidirectional_LSTM')
    
    # 5. Stacked LSTM
    stacked_lstm = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ], name='Stacked_LSTM')
    
    # 6. CNN-RNN Hybrid
    cnn_rnn = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ], name='CNN_RNN')
    
    models = {
        'Simple RNN': simple_rnn,
        'LSTM': lstm_model,
        'GRU': gru_model,
        'Bidirectional LSTM': bidirectional_lstm,
        'Stacked LSTM': stacked_lstm,
        'CNN-RNN Hybrid': cnn_rnn
    }
    
    # Modelleri compile et ve build et
    for name, model in models.items():
        # Modeli build et
        model.build(input_shape=(None, max_length))
        
        if num_classes > 2:
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        print(f"✅ {name}: {model.count_params():,} parameters")
    
    return models

# Modelleri oluştur
num_classes = len(label_encoder.classes_)
models = create_sentiment_models(
    vocab_size=MAX_VOCAB_SIZE,
    embedding_dim=100,
    max_length=MAX_SEQUENCE_LENGTH,
    num_classes=num_classes
)

print_section("MODEL EĞİTİMİ VE PERFORMANS KARŞILAŞTIRMASI")

# Train/validation/test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, 
                                                  random_state=42, stratify=y_temp)

print(f"📊 Veri bölümleme:")
print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

def train_and_evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    """Modelleri eğit ve değerlendir"""
    
    print("🚀 Modeller eğitiliyor...")
    
    histories = {}
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 {name} eğitiliyor...")
        
        # Early stopping ve reduce LR
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,
            min_lr=0.0001
        )
        
        # Eğitim
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Test değerlendirmesi
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        
        if num_classes > 2:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Sonuçları kaydet
        histories[name] = history
        results[name] = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'predictions': y_pred_classes,
            'parameters': model.count_params(),
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"   ✅ Test Accuracy: {test_acc:.4f}")
        print(f"   📊 Test Loss: {test_loss:.4f}")
        print(f"   🏃 Epochs: {len(history.history['loss'])}")
    
    return histories, results

# Modelleri eğit
histories, results = train_and_evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test)

print_section("SONUÇLAR VE GÖRSELLEŞTİRME")

# En iyi modeli bul
best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
print(f"🏆 EN İYİ MODEL: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"   Parameters: {results[best_model_name]['parameters']:,}")

# Sonuçları görselleştir
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('💬 Sentiment Analysis - Model Karşılaştırması', fontsize=16, fontweight='bold')

# 1. Training curves
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for i, (name, history) in enumerate(histories.items()):
    axes[0, 0].plot(history.history['accuracy'], color=colors[i], 
                   label=f'{name}', linewidth=2)
    axes[0, 0].set_title('📈 Training Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

for i, (name, history) in enumerate(histories.items()):
    axes[0, 1].plot(history.history['val_accuracy'], color=colors[i], 
                   label=f'{name}', linewidth=2)
    axes[0, 1].set_title('📊 Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

# 2. Test performance
model_names = list(results.keys())
test_accuracies = [results[name]['test_accuracy'] for name in model_names]
param_counts = [results[name]['parameters'] for name in model_names]

bars = axes[0, 2].bar(range(len(model_names)), test_accuracies, 
                     color=colors[:len(model_names)], alpha=0.7)
axes[0, 2].set_title('🎯 Test Accuracy Comparison', fontweight='bold')
axes[0, 2].set_xlabel('Model')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].set_xticks(range(len(model_names)))
axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 2].grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, test_accuracies):
    height = bar.get_height()
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Parameter counts
axes[1, 0].barh(range(len(model_names)), param_counts, 
               color=colors[:len(model_names)], alpha=0.7)
axes[1, 0].set_title('🔧 Parameter Counts', fontweight='bold')
axes[1, 0].set_xlabel('Parameters')
axes[1, 0].set_ylabel('Model')
axes[1, 0].set_yticks(range(len(model_names)))
axes[1, 0].set_yticklabels(model_names)
axes[1, 0].grid(True, alpha=0.3)

# 4. Accuracy vs Parameters scatter
axes[1, 1].scatter(param_counts, test_accuracies, 
                  s=150, c=colors[:len(model_names)], alpha=0.7)
for i, name in enumerate(model_names):
    axes[1, 1].annotate(name, (param_counts[i], test_accuracies[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1, 1].set_title('⚖️ Accuracy vs Parameters', fontweight='bold')
axes[1, 1].set_xlabel('Parameters')
axes[1, 1].set_ylabel('Test Accuracy')
axes[1, 1].grid(True, alpha=0.3)

# 5. Confusion Matrix for best model
best_model = models[best_model_name]
y_pred_best = results[best_model_name]['predictions']

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_,
           ax=axes[1, 2])
axes[1, 2].set_title(f'🎯 Confusion Matrix - {best_model_name}', fontweight='bold')
axes[1, 2].set_xlabel('Predicted')
axes[1, 2].set_ylabel('Actual')

# 6. Loss curves
for i, (name, history) in enumerate(histories.items()):
    axes[2, 0].plot(history.history['loss'], color=colors[i], 
                   label=f'{name} Train', linewidth=2, alpha=0.7)
    axes[2, 0].plot(history.history['val_loss'], color=colors[i], 
                   linestyle='--', label=f'{name} Val', linewidth=2, alpha=0.7)
axes[2, 0].set_title('📉 Loss Curves', fontweight='bold')
axes[2, 0].set_xlabel('Epoch')
axes[2, 0].set_ylabel('Loss')
axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_yscale('log')

# 7. Training epochs comparison
epochs_trained = [results[name]['epochs_trained'] for name in model_names]
bars = axes[2, 1].bar(range(len(model_names)), epochs_trained,
                     color=colors[:len(model_names)], alpha=0.7)
axes[2, 1].set_title('⏱️ Training Epochs', fontweight='bold')
axes[2, 1].set_xlabel('Model')
axes[2, 1].set_ylabel('Epochs')
axes[2, 1].set_xticks(range(len(model_names)))
axes[2, 1].set_xticklabels(model_names, rotation=45, ha='right')
axes[2, 1].grid(True, alpha=0.3)

# 8. Classification report for best model
report = classification_report(y_test, y_pred_best, 
                             target_names=label_encoder.classes_,
                             output_dict=True)

# Convert to heatmap data
classes = label_encoder.classes_
metrics = ['precision', 'recall', 'f1-score']
report_data = []
for metric in metrics:
    row = []
    for class_name in classes:
        row.append(report[class_name][metric])
    report_data.append(row)

sns.heatmap(report_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
           xticklabels=classes, yticklabels=metrics,
           ax=axes[2, 2])
axes[2, 2].set_title(f'📊 Classification Report - {best_model_name}', fontweight='bold')
axes[2, 2].set_xlabel('Classes')
axes[2, 2].set_ylabel('Metrics')

plt.tight_layout()
plt.show()

print_section("MODEL PERFORMANS ANALİZİ")

print("📊 DETAYLANDIRILMIŞ PERFORMANS ANALİZİ:")
print("="*50)

for name, result in results.items():
    print(f"\n🎯 {name}:")
    print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"   Test Loss: {result['test_loss']:.4f}")
    print(f"   Parameters: {result['parameters']:,}")
    print(f"   Epochs Trained: {result['epochs_trained']}")
    
    # Classification report
    y_pred = result['predictions']
    report = classification_report(y_test, y_pred, 
                                 target_names=label_encoder.classes_,
                                 output_dict=True)
    
    print(f"   Precision (macro avg): {report['macro avg']['precision']:.4f}")
    print(f"   Recall (macro avg): {report['macro avg']['recall']:.4f}")
    print(f"   F1-Score (macro avg): {report['macro avg']['f1-score']:.4f}")

print_section("PRAKTİK SENTIMENT TAHMİN FONKSİYONU")

def predict_sentiment(text, model, tokenizer, label_encoder, max_length=50):
    """Verilen metin için sentiment tahmini yapar"""
    
    # Metni işle
    processed_text = comprehensive_text_preprocessing([text])
    
    # Sequence'e çevir
    sequence = tokenizer.texts_to_sequences(processed_text)
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Tahmin yap
    prediction = model.predict(padded_sequence, verbose=0)
    
    if len(label_encoder.classes_) > 2:
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
    else:
        predicted_class = (prediction[0][0] > 0.5).astype(int)
        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
    
    sentiment = label_encoder.classes_[predicted_class]
    
    return sentiment, confidence, prediction[0]

# Test örnekleri
test_examples = [
    "This movie is absolutely fantastic and amazing!",
    "Terrible product, waste of money.",
    "The product works fine, nothing special.",
    "I love this so much, best purchase ever!",
    "Completely disappointed with the service.",
    "Average quality, meets basic expectations.",
    "Outstanding performance, highly recommend!",
    "Worst experience of my life.",
    "It's okay, does what it's supposed to do."
]

print("🔮 SENTIMENT TAHMİN ÖRNEKLERİ:")
print("-" * 40)

best_model = models[best_model_name]

for example in test_examples:
    sentiment, confidence, full_pred = predict_sentiment(
        example, best_model, tokenizer, label_encoder, MAX_SEQUENCE_LENGTH
    )
    
    print(f"\nMetin: '{example}'")
    print(f"Tahmin: {sentiment.upper()} (Güven: {confidence:.3f})")
    
    # Tüm sınıf olasılıklarını göster
    class_probs = ""
    for i, class_name in enumerate(label_encoder.classes_):
        class_probs += f"{class_name}: {full_pred[i]:.3f}  "
    print(f"Detaylar: {class_probs}")

print_section("WORD EMBEDDİNG ANALİZİ")

def analyze_word_embeddings(model, tokenizer, top_words=20):
    """Word embedding'leri analiz et"""
    
    print("📚 Word embedding analizi...")
    
    # Embedding layer'ını al
    embedding_layer = model.layers[0]  # İlk layer embedding olmalı
    embeddings = embedding_layer.get_weights()[0]
    
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    # En sık kullanılan kelimeleri al
    word_freq = tokenizer.word_counts
    most_common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_words]
    
    print(f"\nEn sık kullanılan {top_words} kelime:")
    for i, (word, freq) in enumerate(most_common_words):
        word_index = tokenizer.word_index.get(word, 0)
        if word_index < len(embeddings):
            embedding_vec = embeddings[word_index]
            print(f"{i+1:2d}. {word:15s} (freq: {freq:3d}) - embedding norm: {np.linalg.norm(embedding_vec):.3f}")

# En iyi modelin embedding'lerini analiz et
analyze_word_embeddings(best_model, tokenizer)

print_section("GELİŞTİRME ÖNERİLERİ VE İPUÇLARI")

improvement_tips = """
🚀 SENTIMENT ANALİZİ İYİLEŞTİRME ÖNERİLERİ:

📊 VERİ İYİLEŞTİRMELERİ:
• Daha büyük ve çeşitli veri setleri kullanın
• Gerçek sosyal medya verileri ekleyin
• Data augmentation teknikleri uygulayın
• Sarcasm ve irony örnekleri ekleyin
• Domain-specific terminoloji dahil edin

🧠 MODEL MİMARİ İYİLEŞTİRMELERİ:
• Pre-trained embeddings (Word2Vec, GloVe, BERT)
• Attention mechanism ekleyin
• Transformer modelleri deneyin
• Ensemble methods kullanın
• Multi-task learning uygulayın

🔧 HİPERPARAMETRE OPTİMİZASYONU:
• Grid search veya Bayesian optimization
• Learning rate scheduling
• Dropout rate tuning
• Embedding dimension optimization
• Sequence length analysis

📈 EVALUATION İYİLEŞTİRMELERİ:
• Cross-validation kullanın
• Stratified sampling uygulayın
• Class imbalance handling
• Per-class analysis yapın
• Error analysis ve case study

🛠️ PRODUKSİYON HAZİRLIĞI:
• Model compression/quantization
• Inference optimization
• Batch prediction support
• API wrapper development
• Monitoring ve logging
"""

print(improvement_tips)

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu Sentiment Analysis modülünde öğrendikleriniz:")
print("  1. 💬 Metin sınıflandırma temelleri")
print("  2. 🧹 Comprehensive text preprocessing")
print("  3. 🔤 Tokenization ve sequence preparation")
print("  4. 🧠 Farklı RNN mimarileri")
print("  5. 📊 Model comparison ve evaluation")
print("  6. 🎯 Practical prediction pipeline")
print("  7. 📚 Word embedding analysis")
print("  8. 🚀 Production deployment considerations")

print(f"\n🏆 PERFORMANS ÖZETİ:")
print(f"   En iyi model: {best_model_name}")
print(f"   Test accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"   Parameters: {results[best_model_name]['parameters']:,}")
print(f"   Classes: {', '.join(label_encoder.classes_)}")

print("\n💡 Ana çıkarımlar:")
print("  • Bidirectional modeller genelde daha iyi performans gösterir")
print("  • Dropout ve regularization overfitting'i önler")
print("  • Balanced dataset önemli")
print("  • Text preprocessing kritik rol oynar")
print("  • Model complexity vs performance trade-off")

print("\n📚 Sonraki modül: 09_time_series_prediction.py")
print("RNN ile gelişmiş zaman serisi tahmini öğreneceğiz!")

print("\n" + "=" * 70)
print("✅ SENTIMENT ANALİZİ MODÜLÜ TAMAMLANDI!")
print("=" * 70)