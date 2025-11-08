"""
🔄 ENCODER-DECODER ARCHITECTURE - KODLAYICI-ÇÖZÜCÜ MİMARİ
========================================================

Bu dosya Encoder-Decoder architecture'in detaylı teorisi ve uygulamalarını açıklar.
Sequence-to-sequence learning'in temelini oluşturan güçlü mimari.

Encoder-Decoder Nedir?
- Input sequence'i context vector'a encode eder
- Context vector'ı output sequence'e decode eder  
- Farklı uzunluktaki input/output sequence'ler için ideal
- Machine translation'ın temelini oluşturur

Kullanım Alanları:
- Machine Translation
- Text Summarization
- Chatbots & Question Answering
- Image Captioning
- Speech Recognition
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                   Embedding, TimeDistributed, Bidirectional,
                                   Concatenate, BatchNormalization, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔄 ENCODER-DECODER ARCHITECTURE - KODLAYICI-ÇÖZÜCÜ MİMARİ")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("ENCODER-DECODER TEORİSİ")

print("🧠 Encoder-Decoder Nedir?")
print("-" * 40)
print("• Input sequence'i fixed-size context vector'a sıkıştırır (Encoder)")
print("• Context vector'ı output sequence'e çevirir (Decoder)")
print("• Seq2Seq learning'in temel mimarisi")
print("• Variable length input/output destekler")

print("\n🔄 ENCODER-DECODER YAPISI:")
print("-" * 30)
print("INPUT:  [x₁, x₂, x₃, x₄]")
print("   ↓")
print("ENCODER: RNN/LSTM/GRU")
print("   ↓")  
print("CONTEXT: c (fixed-size vector)")
print("   ↓")
print("DECODER: RNN/LSTM/GRU")
print("   ↓")
print("OUTPUT: [y₁, y₂, y₃, y₄, y₅]")

print("\n⚖️ AVANTAJ VE DEZAVANTAJLAR:")
print("-" * 30)
print("✅ AVANTAJLAR:")
print("  • Variable length sequences")
print("  • End-to-end learning")
print("  • Flexible architectures")
print("  • Many applications")

print("\n❌ DEZAVANTAJLAR:")
print("  • Information bottleneck (context vector)")
print("  • Long sequence problems")
print("  • Training complexity")
print("  • Gradient vanishing")

def visualize_encoder_decoder_concept():
    """Encoder-Decoder kavramını görselleştirir"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔄 Encoder-Decoder Architecture Concept', fontsize=16, fontweight='bold')
    
    # 1. Basic encoder-decoder flow
    ax = axes[0, 0]
    
    # Input sequence
    input_seq = ['Hello', 'World', 'Good', 'Morning']
    input_positions = np.arange(len(input_seq))
    
    # Output sequence (different length)
    output_seq = ['Merhaba', 'Dünya', 'Günaydın']
    output_positions = np.arange(len(output_seq)) + 6
    
    # Draw encoder
    for i, word in enumerate(input_seq):
        ax.scatter(i, 2, s=200, c='lightblue', edgecolor='blue')
        ax.text(i, 1.7, word, ha='center', fontweight='bold', fontsize=8)
        if i < len(input_seq) - 1:
            ax.arrow(i + 0.1, 2, 0.8, 0, head_width=0.1, head_length=0.05, 
                    fc='blue', ec='blue')
    
    # Context vector
    ax.scatter(4.5, 2, s=300, c='yellow', edgecolor='orange')
    ax.text(4.5, 1.7, 'Context\nVector', ha='center', fontweight='bold', fontsize=8)
    
    # Arrow to decoder
    ax.arrow(4.5, 1.8, 0.8, 0, head_width=0.1, head_length=0.05, 
            fc='purple', ec='purple', linewidth=2)
    
    # Draw decoder
    for i, word in enumerate(output_seq):
        pos = output_positions[i]
        ax.scatter(pos, 2, s=200, c='lightcoral', edgecolor='red')
        ax.text(pos, 2.3, word, ha='center', fontweight='bold', fontsize=8)
        if i < len(output_seq) - 1:
            ax.arrow(pos + 0.1, 2, 0.8, 0, head_width=0.1, head_length=0.05, 
                    fc='red', ec='red')
    
    ax.set_title('🔄 Basic Encoder-Decoder Flow', fontweight='bold')
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(1.5, 2.5)
    ax.text(1.5, 2.4, 'ENCODER', ha='center', fontweight='bold', color='blue', fontsize=12)
    ax.text(7, 2.4, 'DECODER', ha='center', fontweight='bold', color='red', fontsize=12)
    ax.axis('off')
    
    # 2. Context vector bottleneck visualization
    ax = axes[0, 1]
    
    sequence_lengths = [5, 10, 15, 20, 25, 30]
    context_size = [4] * len(sequence_lengths)  # Fixed context size
    information_loss = [0, 5, 15, 30, 50, 70]  # Increasing information loss
    
    ax.bar(sequence_lengths, context_size, width=2, alpha=0.7, 
          color='orange', label='Context Vector Size')
    ax.plot(sequence_lengths, information_loss, 'ro-', linewidth=2, 
           label='Information Loss %', markersize=6)
    
    ax.set_title('📊 Information Bottleneck Problem', fontweight='bold')
    ax.set_xlabel('Input Sequence Length')
    ax.set_ylabel('Information Capacity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training process visualization
    ax = axes[1, 0]
    
    # Teacher forcing vs inference time
    teacher_seq = ['<start>', 'Merhaba', 'Dünya']
    target_seq = ['Merhaba', 'Dünya', '<end>']
    
    for i, (inp, target) in enumerate(zip(teacher_seq, target_seq)):
        # Input to decoder
        ax.scatter(i, 1, s=150, c='lightgreen', edgecolor='green')
        ax.text(i, 0.7, inp, ha='center', fontweight='bold', fontsize=8)
        
        # Target output
        ax.scatter(i, 2, s=150, c='lightcoral', edgecolor='red')
        ax.text(i, 2.3, target, ha='center', fontweight='bold', fontsize=8)
        
        # Arrow from input to output
        ax.arrow(i, 1.2, 0, 0.6, head_width=0.05, head_length=0.05,
                fc='purple', ec='purple')
    
    ax.set_title('🎓 Teacher Forcing Training', fontweight='bold')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.5, 2.8)
    ax.text(1, 0.5, 'Decoder Input', ha='center', fontweight='bold', color='green')
    ax.text(1, 2.6, 'Target Output', ha='center', fontweight='bold', color='red')
    ax.axis('off')
    
    # 4. Architecture comparison
    ax = axes[1, 1]
    
    architectures = ['Basic\nSeq2Seq', 'With\nAttention', 'Bidirectional\nEncoder', 'Multi-layer\nEncoder-Decoder']
    performance = [75, 88, 83, 91]
    complexity = [1, 3, 2, 4]
    
    colors = ['red', 'orange', 'blue', 'purple']
    
    for i, (arch, perf, comp) in enumerate(zip(architectures, performance, complexity)):
        ax.scatter(comp, perf, s=200, c=colors[i], alpha=0.7, edgecolor='black')
        ax.annotate(arch, (comp, perf), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold')
    
    ax.set_title('⚖️ Architecture Trade-offs', fontweight='bold')
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Performance (%)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(70, 95)
    
    plt.tight_layout()
    plt.show()
    
    print("🔍 Görselleştirme Açıklaması:")
    print("1. Basic Flow: Input → Context → Output transformation")
    print("2. Bottleneck: Fixed context size causes information loss")
    print("3. Teacher Forcing: Training time vs inference time difference")
    print("4. Trade-offs: Performance vs complexity comparison")

visualize_encoder_decoder_concept()

print_section("BASIC ENCODER-DECODER İMPLEMENTASYONU")

def create_basic_encoder_decoder():
    """Temel Encoder-Decoder model oluşturur"""
    
    print("🏗️ Basic Encoder-Decoder oluşturuluyor...")
    
    # Hyperparameters
    vocab_size = 1000
    embedding_dim = 128
    hidden_units = 256
    max_encoder_len = 20
    max_decoder_len = 25
    
    # ENCODER
    encoder_inputs = Input(shape=(max_encoder_len,), name='encoder_inputs')
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    
    # Encoder LSTM
    encoder_lstm = LSTM(hidden_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]  # Context vector
    
    # DECODER
    decoder_inputs = Input(shape=(max_decoder_len,), name='decoder_inputs')
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    
    # Decoder LSTM
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Dense layer for output
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    print(f"✅ Basic Encoder-Decoder:")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Encoder length: {max_encoder_len}")
    print(f"   Decoder length: {max_decoder_len}")
    
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense

basic_model, enc_inputs, enc_states, dec_inputs, dec_lstm, dec_dense = create_basic_encoder_decoder()

print_section("INFERENCE MODEL OLUŞTURMA")

def create_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense):
    """Inference için ayrı encoder ve decoder modelleri oluşturur"""
    
    print("🔮 Inference models oluşturuluyor...")
    
    # ENCODER MODEL (for inference)
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # DECODER MODEL (for inference)
    # Decoder state inputs
    decoder_state_input_h = Input(shape=(256,))
    decoder_state_input_c = Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Single token input for decoder
    decoder_single_input = Input(shape=(1,))
    decoder_embedding = Embedding(1000, 128, mask_zero=True)
    decoder_embedded = decoder_embedding(decoder_single_input)
    
    # Decoder LSTM step
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedded, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    
    # Output prediction
    decoder_outputs = dec_dense(decoder_outputs)
    
    decoder_model = Model(
        [decoder_single_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    
    print("✅ Inference models:")
    print(f"   Encoder model: {encoder_model.count_params():,} params")
    print(f"   Decoder model: {decoder_model.count_params():,} params")
    
    return encoder_model, decoder_model

encoder_model, decoder_model = create_inference_models(
    enc_inputs, enc_states, dec_inputs, dec_lstm, dec_dense
)

print_section("SEQ2SEQ TRAINING DEMONSTRATION")

def demonstrate_seq2seq_training():
    """Seq2Seq training demonstration"""
    
    print("🚀 Seq2Seq training demonstration...")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    vocab_size = 1000
    max_encoder_len = 20
    max_decoder_len = 25
    n_samples = 1000
    
    # Reserve indices for special tokens (within vocab_size range)
    START_TOKEN = vocab_size - 2  # 998
    END_TOKEN = vocab_size - 1    # 999
    VOCAB_SIZE_ACTUAL = vocab_size - 2  # Use 0-997 for regular tokens
    
    # Generate synthetic sequence pairs
    encoder_sequences = []
    decoder_sequences = []
    target_sequences = []
    
    for i in range(n_samples):
        # Random encoder sequence
        enc_len = np.random.randint(5, max_encoder_len)
        enc_seq = np.random.randint(1, VOCAB_SIZE_ACTUAL, enc_len)  # Use reduced vocab for regular tokens
        enc_seq_padded = np.pad(enc_seq, (0, max_encoder_len - enc_len), 'constant')
        encoder_sequences.append(enc_seq_padded)
        
        # Decoder sequence (reverse of encoder + some noise for variety)
        dec_len = np.random.randint(5, max_decoder_len - 1)
        
        # Start with <START> token
        dec_seq = [START_TOKEN]  # <START> token
        target_seq = []
        
        # Create some pattern (e.g., reverse first few elements)
        reverse_len = min(3, enc_len)
        for j in range(reverse_len):
            if j < enc_len:
                token = enc_seq[enc_len - 1 - j]
                dec_seq.append(token)
                target_seq.append(token)
        
        # Fill remaining with random tokens
        remaining = dec_len - reverse_len
        for j in range(remaining):
            token = np.random.randint(1, VOCAB_SIZE_ACTUAL)  # Use reduced vocab
            dec_seq.append(token)
            target_seq.append(token)
        
        # Add <END> token
        target_seq.append(END_TOKEN)  # <END> token
        
        # Pad sequences
        dec_seq_padded = np.pad(dec_seq, (0, max_decoder_len - len(dec_seq)), 'constant')
        target_seq_padded = np.pad(target_seq, (0, max_decoder_len - len(target_seq)), 'constant')
        
        decoder_sequences.append(dec_seq_padded)
        target_sequences.append(target_seq_padded)
    
    encoder_data = np.array(encoder_sequences)
    decoder_data = np.array(decoder_sequences)
    target_data = np.array(target_sequences)
    
    print(f"Training data shapes:")
    print(f"  Encoder: {encoder_data.shape}")
    print(f"  Decoder: {decoder_data.shape}")
    print(f"  Target: {target_data.shape}")
    
    # Create and compile model
    model = basic_model
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Quick training (limited epochs for demo)
    print("\n🎓 Training started...")
    
    from sklearn.model_selection import train_test_split
    
    encoder_train, encoder_val, decoder_train, decoder_val, target_train, target_val = train_test_split(
        encoder_data, decoder_data, target_data, test_size=0.2, random_state=42
    )
    
    history = model.fit(
        [encoder_train, decoder_train], target_train,
        validation_data=([encoder_val, decoder_val], target_val),
        batch_size=32,
        epochs=5,  # Limited for demonstration
        verbose=1
    )
    
    print("✅ Training completed!")
    
    return model, history, encoder_val, decoder_val, target_val

trained_model, history, encoder_val, decoder_val, target_val = demonstrate_seq2seq_training()

print_section("GELİŞMİŞ ENCODER-DECODER VARIANTS")

def create_advanced_encoder_decoder():
    """Gelişmiş Encoder-Decoder variants"""
    
    print("🚀 Advanced Encoder-Decoder variants...")
    
    # 1. Bidirectional Encoder
    def bidirectional_encoder_model():
        vocab_size = 1000
        embedding_dim = 128
        hidden_units = 128  # Smaller since bidirectional doubles the size
        
        # Encoder with bidirectional LSTM
        encoder_inputs = Input(shape=(20,))
        encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        
        # Bidirectional encoder
        encoder_bi = Bidirectional(LSTM(hidden_units, return_state=True))
        encoder_outputs, fwd_h, fwd_c, bwd_h, bwd_c = encoder_bi(encoder_embedding)
        
        # Combine states from both directions
        encoder_h = Concatenate()([fwd_h, bwd_h])
        encoder_c = Concatenate()([fwd_c, bwd_c])
        encoder_states = [encoder_h, encoder_c]
        
        # Decoder
        decoder_inputs = Input(shape=(25,))
        decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(hidden_units * 2, return_sequences=True)  # Match encoder size
        decoder_outputs = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model
    
    # 2. Multi-layer Encoder-Decoder
    def multilayer_encoder_decoder():
        vocab_size = 1000
        embedding_dim = 128
        hidden_units = 256
        num_layers = 2
        
        # Multi-layer encoder
        encoder_inputs = Input(shape=(20,))
        encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        
        # Stack multiple LSTM layers in encoder
        encoder_outputs = encoder_embedding
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer returns states
                encoder_lstm = LSTM(hidden_units, return_state=True)
                encoder_outputs, state_h, state_c = encoder_lstm(encoder_outputs)
                encoder_states = [state_h, state_c]
            else:
                encoder_lstm = LSTM(hidden_units, return_sequences=True)
                encoder_outputs = encoder_lstm(encoder_outputs)
        
        # Multi-layer decoder
        decoder_inputs = Input(shape=(25,))
        decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
        
        # Stack multiple LSTM layers in decoder
        decoder_outputs = decoder_embedding
        for i in range(num_layers):
            if i == 0:  # First layer uses encoder states
                decoder_lstm = LSTM(hidden_units, return_sequences=True)
                decoder_outputs = decoder_lstm(decoder_outputs, initial_state=encoder_states)
            else:
                decoder_lstm = LSTM(hidden_units, return_sequences=True)
                decoder_outputs = decoder_lstm(decoder_outputs)
        
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model
    
    # 3. Encoder-Decoder with Skip Connections
    def skip_connection_model():
        vocab_size = 1000
        embedding_dim = 128
        hidden_units = 256
        
        # Encoder
        encoder_inputs = Input(shape=(20,))
        encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        
        # Two encoder layers
        encoder_layer1 = LSTM(hidden_units, return_sequences=True)(encoder_embedding)
        encoder_layer2, state_h, state_c = LSTM(hidden_units, return_state=True)(encoder_layer1)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(25,))
        decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
        
        # Decoder with skip connection
        decoder_layer1 = LSTM(hidden_units, return_sequences=True)(
            decoder_embedding, initial_state=encoder_states
        )
        
        # Skip connection: combine with encoder layer1 information
        # For simplicity, we'll use the final encoder output as skip connection
        encoder_final = Lambda(lambda x: tf.expand_dims(x, 1))(encoder_layer2)
        encoder_repeated = Lambda(lambda x: tf.tile(x, [1, 25, 1]))(encoder_final)
        
        # Combine decoder output with encoder information
        combined = Concatenate()([decoder_layer1, encoder_repeated])
        
        decoder_layer2 = LSTM(hidden_units, return_sequences=True)(combined)
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_layer2)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model
    
    # Create all variants
    bi_model = bidirectional_encoder_model()
    multi_model = multilayer_encoder_decoder()
    skip_model = skip_connection_model()
    
    models = {
        'Bidirectional Encoder': bi_model,
        'Multi-layer': multi_model,
        'Skip Connections': skip_model
    }
    
    print("✅ Advanced variants created:")
    for name, model in models.items():
        print(f"   {name}: {model.count_params():,} parameters")
    
    return models

advanced_models = create_advanced_encoder_decoder()

print_section("ENCODER-DECODER PERFORMANCE ANALİZİ")

def analyze_encoder_decoder_performance():
    """Encoder-Decoder performance analysis"""
    
    print("📊 Performance analysis...")
    
    # Visualize different architectural choices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔄 Encoder-Decoder Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Parameter count comparison
    model_names = ['Basic', 'Bidirectional', 'Multi-layer', 'Skip Connections']
    param_counts = [
        basic_model.count_params(),
        advanced_models['Bidirectional Encoder'].count_params(),
        advanced_models['Multi-layer'].count_params(),
        advanced_models['Skip Connections'].count_params()
    ]
    
    bars = axes[0, 0].bar(model_names, param_counts, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[0, 0].set_title('📊 Parameter Count Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Parameter Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 10000,
                       f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 2. Theoretical performance vs complexity
    complexity_scores = [1, 2, 3, 2.5]
    performance_scores = [70, 78, 85, 80]  # Theoretical scores
    
    colors = ['blue', 'green', 'orange', 'red']
    for i, (name, comp, perf) in enumerate(zip(model_names, complexity_scores, performance_scores)):
        axes[0, 1].scatter(comp, perf, s=200, c=colors[i], alpha=0.7, edgecolor='black')
        axes[0, 1].annotate(name, (comp, perf), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9, fontweight='bold')
    
    axes[0, 1].set_title('⚖️ Performance vs Complexity', fontweight='bold')
    axes[0, 1].set_xlabel('Complexity Score')
    axes[0, 1].set_ylabel('Performance Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.5, 3.5)
    axes[0, 1].set_ylim(65, 90)
    
    # 3. Training curves simulation
    epochs = np.arange(1, 21)
    
    # Simulated training curves for different architectures
    basic_loss = 2.5 * np.exp(-0.1 * epochs) + 0.5 + 0.1 * np.random.randn(20) * 0.1
    bi_loss = 2.3 * np.exp(-0.12 * epochs) + 0.4 + 0.1 * np.random.randn(20) * 0.1
    multi_loss = 2.0 * np.exp(-0.15 * epochs) + 0.3 + 0.1 * np.random.randn(20) * 0.1
    skip_loss = 2.1 * np.exp(-0.13 * epochs) + 0.35 + 0.1 * np.random.randn(20) * 0.1
    
    axes[1, 0].plot(epochs, basic_loss, 'b-', label='Basic', linewidth=2)
    axes[1, 0].plot(epochs, bi_loss, 'g-', label='Bidirectional', linewidth=2)
    axes[1, 0].plot(epochs, multi_loss, 'orange', label='Multi-layer', linewidth=2)
    axes[1, 0].plot(epochs, skip_loss, 'r-', label='Skip Connections', linewidth=2)
    
    axes[1, 0].set_title('📉 Training Loss Curves', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Use case recommendations
    use_cases = ['Translation', 'Summarization', 'Chatbot', 'Captioning']
    basic_scores = [60, 55, 50, 45]
    bi_scores = [75, 70, 65, 70]
    multi_scores = [85, 80, 75, 80]
    skip_scores = [80, 85, 70, 75]
    
    x = np.arange(len(use_cases))
    width = 0.2
    
    axes[1, 1].bar(x - 1.5*width, basic_scores, width, label='Basic', alpha=0.7)
    axes[1, 1].bar(x - 0.5*width, bi_scores, width, label='Bidirectional', alpha=0.7)
    axes[1, 1].bar(x + 0.5*width, multi_scores, width, label='Multi-layer', alpha=0.7)
    axes[1, 1].bar(x + 1.5*width, skip_scores, width, label='Skip Connections', alpha=0.7)
    
    axes[1, 1].set_title('🎯 Use Case Performance', fontweight='bold')
    axes[1, 1].set_xlabel('Use Case')
    axes[1, 1].set_ylabel('Performance Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(use_cases)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📈 Analysis Results:")
    print("• Multi-layer: En yüksek performance ama en karmaşık")
    print("• Bidirectional: İyi balance between performance ve complexity")
    print("• Skip Connections: Özellikle summarization task'lerde etkili")
    print("• Basic: Basit task'ler ve prototype'lar için yeterli")

analyze_encoder_decoder_performance()

print_section("PRACTICAL APPLICATIONS")

def practical_encoder_decoder_applications():
    """Pratik Encoder-Decoder uygulamaları"""
    
    print("🚀 Practical applications...")
    
    # Example 1: Simple Number Sequence Translation
    print("\n1️⃣ NUMBER SEQUENCE TRANSLATION:")
    print("-" * 40)
    
    def create_number_translation_data():
        """Create simple number sequence translation data"""
        
        # Task: Convert ascending sequence to descending
        # Input: [1, 2, 3, 4] → Output: [4, 3, 2, 1]
        
        data_pairs = []
        for start in range(1, 101):  # 100 different sequences
            length = np.random.randint(3, 8)
            input_seq = list(range(start, start + length))
            output_seq = input_seq[::-1]  # Reverse
            
            data_pairs.append((input_seq, output_seq))
        
        return data_pairs
    
    number_data = create_number_translation_data()
    print(f"Created {len(number_data)} number sequence pairs")
    print(f"Example: {number_data[0][0]} → {number_data[0][1]}")
    
    # Example 2: Character-level Text Processing
    print("\n2️⃣ CHARACTER-LEVEL TEXT PROCESSING:")
    print("-" * 40)
    
    def create_character_reversal_data():
        """Create character-level text reversal data"""
        
        words = ['hello', 'world', 'python', 'deep', 'learning', 'neural', 'network',
                'encoder', 'decoder', 'attention', 'transformer', 'sequence']
        
        data_pairs = []
        for word in words:
            # Input: character sequence
            # Output: reversed character sequence
            input_chars = list(word)
            output_chars = list(word[::-1])
            
            data_pairs.append((input_chars, output_chars))
        
        return data_pairs
    
    char_data = create_character_reversal_data()
    print(f"Created {len(char_data)} character sequence pairs")
    print(f"Example: {char_data[0][0]} → {char_data[0][1]}")
    
    # Example 3: Mathematical Operation Learning
    print("\n3️⃣ MATHEMATICAL OPERATION LEARNING:")
    print("-" * 40)
    
    def create_math_operation_data():
        """Create mathematical operation data"""
        
        operations = []
        for a in range(1, 21):
            for b in range(1, 11):
                # Addition task: input two numbers, output sum
                input_seq = [a, b]
                output_seq = [a + b]
                operations.append((input_seq, output_seq))
                
                # Multiplication task
                input_seq = [a, b]
                output_seq = [a * b]
                operations.append((input_seq, output_seq))
        
        return operations[:100]  # Limit for demonstration
    
    math_data = create_math_operation_data()
    print(f"Created {len(math_data)} mathematical operations")
    print(f"Example: {math_data[0][0]} → {math_data[0][1]}")
    print(f"Example: {math_data[1][0]} → {math_data[1][1]}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('🔄 Practical Encoder-Decoder Applications', fontsize=14, fontweight='bold')
    
    # Number sequence lengths
    seq_lengths = [len(pair[0]) for pair in number_data[:50]]
    axes[0].hist(seq_lengths, bins=6, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Number Sequence Lengths', fontweight='bold')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Character sequence lengths
    char_lengths = [len(pair[0]) for pair in char_data]
    axes[1].hist(char_lengths, bins=8, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Character Sequence Lengths', fontweight='bold')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)
    
    # Math operations distribution
    results = [pair[1][0] for pair in math_data if pair[1][0] <= 50]  # Filter large results
    axes[2].hist(results, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[2].set_title('Math Operation Results', fontweight='bold')
    axes[2].set_xlabel('Result Value')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return number_data, char_data, math_data

number_data, char_data, math_data = practical_encoder_decoder_applications()

print_section("ENCODER-DECODER BEST PRACTICES")

best_practices = """
💡 ENCODER-DECODER BEST PRACTICES:

🎯 ARCHITECTURE DESIGN:
• Encoder depth: 2-4 layers genellikle yeterli
• Decoder depth: Encoder ile eşit veya 1 fazla
• Hidden units: 256-512 orta scale task'ler için
• Bidirectional encoder: Context için önemli
• Attention mechanism: Uzun sequence'ler için kritik

📊 TRAINING STRATEGIES:
1. **Teacher Forcing:**
   • Training: Gerçek target sequence kullan
   • Inference: Model'in kendi output'unu kullan
   • Scheduled sampling: Ikisini karıştır

2. **Optimization:**
   • Adam optimizer: Genellikle en iyi choice
   • Learning rate scheduling
   • Gradient clipping: Exploding gradient için
   • Early stopping: Overfitting önleme

3. **Data Preparation:**
   • Special tokens: <START>, <END>, <PAD>, <UNK>
   • Proper tokenization
   • Sequence length optimization
   • Data augmentation techniques

🔍 TROUBLESHOOTING:
• Vanishing gradients: Skip connections, better initialization
• Information bottleneck: Attention mechanism ekle
• Slow convergence: Learning rate tuning, batch size
• Poor generalization: Dropout, regularization

⚡ PERFORMANCE OPTIMIZATION:
• Batch processing
• Mixed precision training
• Model parallelism for large models
• Beam search for better inference quality
"""

print(best_practices)

print_section("MODERN ENCODER-DECODER ÉVOLUTION")

evolution_info = """
🔄 ENCODER-DECODER EVOLUTION:

📈 HISTORICAL DEVELOPMENT:
2014: 📝 Basic Seq2Seq (Sutskever et al.)
2015: 🔍 Attention Mechanism (Bahdanau et al.)
2016: 🎯 Google Neural Machine Translation
2017: 🤖 Transformer Architecture (Attention is All You Need)
2018: 🧠 BERT & GPT models
2019: 📚 T5 (Text-to-Text Transfer Transformer)
2020: 🚀 GPT-3 & Large Language Models
2021+: 🌟 Multimodal & Unified Architectures

🔄 KEY INNOVATIONS:
• Attention Mechanism → Information Bottleneck Çözümü
• Self-Attention → Intra-sequence Dependencies
• Multi-Head Attention → Parallel Information Processing
• Transformer → Eliminate Recurrence Completely
• Pre-training → Transfer Learning Revolution

🎯 CURRENT TRENDS:
• 🤖 Large Language Models (LLMs)
• 🔄 Unified Text-to-Text Models
• 🎨 Multimodal Applications
• ⚡ Efficient Architectures
• 🧠 Few-shot Learning Capabilities

📊 PERFORMANCE MILESTONES:
• WMT'14 Translation: BLEU 37 → 43
• SQuAD QA: F1 80 → 95+
• GLUE Benchmark: 80 → 90+
• Human-level performance in many tasks
"""

print(evolution_info)

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu Encoder-Decoder modülünde öğrendikleriniz:")
print("  1. 🔄 Encoder-Decoder architecture teorisi")
print("  2. 🧠 Information bottleneck problem ve çözümleri")
print("  3. 🏗️ Basic implementation ve inference models")
print("  4. 🚀 Advanced variants (bidirectional, multi-layer, skip)")
print("  5. 📊 Performance analysis ve architecture comparison")
print("  6. 🎯 Practical applications ve use cases")
print("  7. 💡 Training strategies ve best practices")
print("  8. 📈 Historical development ve modern trends")

print(f"\n🏆 MODEL COMPARISON:")
for name, model in [('Basic', basic_model)] + list(advanced_models.items()):
    param_count = model.count_params()
    print(f"   {name:20s}: {param_count:,} parameters")

print("\n💡 Ana çıkarımlar:")
print("  • Encoder-Decoder Seq2Seq learning'in temelini oluşturur")
print("  • Information bottleneck attention ile çözülür")
print("  • Architecture choice use case'e göre optimize edilmeli")
print("  • Teacher forcing training/inference gap'ini azaltır")
print("  • Modern NLP'nin foundation'ını sağlar")

print("\n🎯 Kullanım önerileri:")
print("  • Translation: Multi-layer + Attention")
print("  • Summarization: Bidirectional encoder + Skip connections")
print("  • Chatbots: Basic model + Fine-tuning")
print("  • Complex tasks: Transformer-based architectures")

print("\n📚 Bu modül ile RNN Educational Package tamamlandı!")
print("Artık RNN'lerden modern Transformer'lara kadar sequence modeling'in")
print("tüm temellerine hakim oldunuz. Başarılar! 🎉")

print("\n" + "=" * 70)
print("✅ ENCODER-DECODER MODÜLÜ TAMAMLANDI!")
print("🎓 RNN EDUCATIONAL PACKAGE COMPLETE!")
print("=" * 70)