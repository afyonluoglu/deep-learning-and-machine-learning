"""
🔍 ATTENTION MECHANISM - DİKKAT MEKANİZMASI
==========================================

Bu dosya Attention Mechanism'in detaylı teorisi ve uygulamalarını açıklar.
RNN'lerde attention ile sequence modeling'i devrim niteliğinde geliştiren teknoloji.

Attention Nedir?
- Model'in sequence'in farklı bölümlerine farklı önem verir
- Long-term dependencies'i daha iyi yakalar
- Transformer'ların temelini oluşturur
- Interpretability sağlar (hangi kısmı dinliyor?)

Kullanım Alanları:
- Machine Translation
- Text Summarization  
- Question Answering
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Bidirectional,
                                   Input, Concatenate, TimeDistributed,
                                   BatchNormalization, Embedding, Layer, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔍 ATTENTION MECHANISM - DİKKAT MEKANİZMASI")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("ATTENTION MECHANISM TEORİSİ")

print("🧠 Attention Nedir?")
print("-" * 40)
print("• Model'in input sequence'in farklı kısımlarına odaklanması")
print("• Her output için hangi input'ların önemli olduğunu öğrenir") 
print("• Context vector'ı dinamik olarak hesaplar")
print("• Bottleneck problemini çözer")

print("\n🔍 ATTENTION FORMÜL:")
print("-" * 30)
print("1. Energy: e_ij = f(s_i, h_j)")
print("2. Weights: α_ij = softmax(e_ij)")  
print("3. Context: c_i = Σ(α_ij * h_j)")
print("4. Output: y_i = g(s_i, c_i)")
print("")
print("s_i: decoder hidden state at time i")
print("h_j: encoder hidden state at time j")
print("α_ij: attention weight")
print("c_i: context vector")

print("\n🎯 ATTENTION TÜRLERİ:")
print("-" * 30)
print("1. 🔍 Additive (Bahdanau): e = v^T tanh(W₁h + W₂s)")
print("2. 🔍 Multiplicative (Luong): e = s^T W h")
print("3. 🔍 Dot-product: e = s^T h")
print("4. 🔍 Self-attention: Query=Key=Value")
print("5. 🔍 Multi-head attention: Paralel attention heads")

def visualize_attention_concept():
    """Attention mechanism kavramını görselleştirir"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔍 Attention Mechanism Concept', fontsize=16, fontweight='bold')
    
    # 1. Attention alignment visualization
    sequence_len = 8
    target_len = 6
    
    # Simulated attention weights
    np.random.seed(42)
    attention_weights = np.random.rand(target_len, sequence_len)
    
    # Normalize each row to sum to 1 (softmax-like)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    # Add some structure to make it more realistic
    for i in range(target_len):
        # Create diagonal pattern with some noise
        peak_pos = int(i * sequence_len / target_len)
        for j in range(sequence_len):
            distance = abs(j - peak_pos)
            attention_weights[i, j] *= np.exp(-distance * 0.3)
    
    # Renormalize
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    im = axes[0, 0].imshow(attention_weights, cmap='Blues', interpolation='nearest')
    axes[0, 0].set_title('🔍 Attention Alignment Matrix', fontweight='bold')
    axes[0, 0].set_xlabel('Source Position (Encoder)')
    axes[0, 0].set_ylabel('Target Position (Decoder)')
    
    # Add grid and labels
    axes[0, 0].set_xticks(range(sequence_len))
    axes[0, 0].set_yticks(range(target_len))
    axes[0, 0].set_xticklabels([f'h{i+1}' for i in range(sequence_len)])
    axes[0, 0].set_yticklabels([f's{i+1}' for i in range(target_len)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # 2. Traditional RNN vs Attention comparison
    time_steps = np.arange(1, 9)
    
    # Traditional RNN: information decay
    rnn_info = np.exp(-0.3 * (time_steps - 1))  # Exponential decay
    
    # Attention: can access any information equally
    attention_info = np.ones_like(time_steps)
    
    axes[0, 1].bar(time_steps - 0.2, rnn_info, width=0.4, 
                  alpha=0.7, color='red', label='Traditional RNN')
    axes[0, 1].bar(time_steps + 0.2, attention_info, width=0.4,
                  alpha=0.7, color='blue', label='With Attention')
    
    axes[0, 1].set_title('📊 Information Access Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Information Accessibility')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Attention mechanism flow diagram
    # Create a simple flow representation
    encoder_positions = np.arange(0, 5)
    decoder_position = 2.5
    
    # Encoder states
    for i, pos in enumerate(encoder_positions):
        axes[1, 0].scatter(pos, 2, s=200, c='lightblue', 
                          edgecolor='blue', zorder=3)
        axes[1, 0].text(pos, 1.7, f'h{i+1}', ha='center', fontweight='bold')
    
    # Decoder state
    axes[1, 0].scatter(decoder_position, 0.5, s=300, c='lightcoral', 
                      edgecolor='red', zorder=3)
    axes[1, 0].text(decoder_position, 0.2, 's_i', ha='center', fontweight='bold')
    
    # Attention connections with varying thickness
    for i, pos in enumerate(encoder_positions):
        weight = attention_weights[2, i] if i < len(attention_weights[0]) else 0.1
        axes[1, 0].plot([decoder_position, pos], [0.5, 2], 
                       'purple', linewidth=weight*10, alpha=0.7)
        # Add weight labels
        mid_x, mid_y = (decoder_position + pos)/2, 1.25
        axes[1, 0].text(mid_x, mid_y, f'{weight:.2f}', 
                       ha='center', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    axes[1, 0].set_title('🔗 Attention Connection Flow', fontweight='bold')
    axes[1, 0].set_xlim(-0.5, 4.5)
    axes[1, 0].set_ylim(0, 2.5)
    axes[1, 0].text(2, 2.3, 'Encoder States', ha='center', fontweight='bold', color='blue')
    axes[1, 0].text(2.5, 0, 'Decoder State', ha='center', fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # 4. Self-attention visualization
    # Create self-attention pattern
    self_attention = np.eye(8) * 0.4  # Diagonal dominance
    
    # Add some off-diagonal connections
    for i in range(8):
        for j in range(8):
            if abs(i - j) == 1:
                self_attention[i, j] = 0.2
            elif abs(i - j) == 2:
                self_attention[i, j] = 0.1
    
    # Add random noise and normalize
    self_attention += np.random.rand(8, 8) * 0.1
    self_attention = self_attention / self_attention.sum(axis=1, keepdims=True)
    
    im2 = axes[1, 1].imshow(self_attention, cmap='Greens', interpolation='nearest')
    axes[1, 1].set_title('🔍 Self-Attention Pattern', fontweight='bold')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')
    axes[1, 1].set_xticks(range(8))
    axes[1, 1].set_yticks(range(8))
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
    cbar2.set_label('Self-Attention Weight', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()
    
    print("🔍 Görselleştirme Açıklaması:")
    print("1. Attention Matrix: Her decoder step'i encoder'ın hangi kısımlarına dikkat ediyor")
    print("2. Information Access: Attention ile tüm bilgiye eşit erişim")
    print("3. Connection Flow: Attention weight'ler connection strength'i gösterir")
    print("4. Self-Attention: Token'ların birbirleriyle ilişkisi")

visualize_attention_concept()

print_section("ATTENTION LAYER İMPLEMENTASYONU")

class BahdanauAttention(Layer):
    """Bahdanau (Additive) Attention implementation"""
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.V = Dense(1, use_bias=False)
        
    def call(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_len, hidden_size)
        
        # Expand dims to broadcast
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Calculate energy
        # score shape: (batch_size, max_len, units)
        score = tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        )
        
        # Attention weights
        # attention_weights shape: (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Context vector
        # context_vector shape: (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class LuongAttention(Layer):
    """Luong (Multiplicative) Attention implementation"""
    
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W = Dense(units, use_bias=False)
        
    def call(self, query, values):
        # query shape: (batch_size, hidden_size)  
        # values shape: (batch_size, max_len, hidden_size)
        
        # Score calculation
        # score shape: (batch_size, max_len)
        score = tf.reduce_sum(
            tf.expand_dims(self.W(query), 1) * values, axis=2
        )
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context_vector = tf.expand_dims(attention_weights, 2) * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, tf.expand_dims(attention_weights, 2)

def demonstrate_attention_implementations():
    """Attention implementasyonlarını gösterir"""
    
    print("🔧 Custom Attention Layer implementations...")
    
    # Create synthetic sequence-to-sequence data
    np.random.seed(42)
    batch_size = 32
    max_len = 10
    hidden_size = 64
    
    # Encoder outputs (values)
    encoder_outputs = tf.random.normal((batch_size, max_len, hidden_size))
    
    # Decoder state (query)
    decoder_state = tf.random.normal((batch_size, hidden_size))
    
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Decoder state shape: {decoder_state.shape}")
    
    # Test Bahdanau Attention
    bahdanau_attention = BahdanauAttention(hidden_size)
    context_vector_b, attention_weights_b = bahdanau_attention(decoder_state, encoder_outputs)
    
    print(f"\n🔍 Bahdanau Attention:")
    print(f"Context vector shape: {context_vector_b.shape}")
    print(f"Attention weights shape: {attention_weights_b.shape}")
    print(f"Attention weights sum: {float(tf.reduce_sum(attention_weights_b, axis=1)[0]):.4f}")
    
    # Test Luong Attention
    luong_attention = LuongAttention(hidden_size)
    context_vector_l, attention_weights_l = luong_attention(decoder_state, encoder_outputs)
    
    print(f"\n🔍 Luong Attention:")
    print(f"Context vector shape: {context_vector_l.shape}")
    print(f"Attention weights shape: {attention_weights_l.shape}")
    print(f"Attention weights sum: {float(tf.reduce_sum(attention_weights_l, axis=1)[0]):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('🔍 Attention Mechanisms Comparison', fontsize=14, fontweight='bold')
    
    # Bahdanau attention weights for first sample
    bahdanau_weights = attention_weights_b[0, :, 0].numpy()
    luong_weights = attention_weights_l[0, :, 0].numpy()
    
    positions = np.arange(max_len)
    
    axes[0].bar(positions, bahdanau_weights, alpha=0.7, color='blue')
    axes[0].set_title('Bahdanau (Additive) Attention', fontweight='bold', color='blue')
    axes[0].set_xlabel('Encoder Position')
    axes[0].set_ylabel('Attention Weight')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(positions, luong_weights, alpha=0.7, color='red')
    axes[1].set_title('Luong (Multiplicative) Attention', fontweight='bold', color='red')
    axes[1].set_xlabel('Encoder Position') 
    axes[1].set_ylabel('Attention Weight')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return bahdanau_attention, luong_attention

bahdanau_attention, luong_attention = demonstrate_attention_implementations()

print_section("SEQ2SEQ WITH ATTENTION MODEL")

def create_seq2seq_with_attention():
    """Attention'lı Seq2Seq model oluşturur"""
    
    print("🏗️ Seq2Seq with Attention model oluşturuluyor...")
    
    # Hyperparameters
    vocab_size = 1000
    embedding_dim = 128
    hidden_units = 256
    max_len = 20
    
    # Encoder
    encoder_inputs = Input(shape=(max_len,), name='encoder_inputs')
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(hidden_units, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(max_len,), name='decoder_inputs')
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Attention layer
    attention_layer = BahdanauAttention(hidden_units)
    
    # Apply attention to all decoder outputs at once
    # Get the last hidden state as query for simplicity
    query = Lambda(lambda x: x[:, -1, :])(decoder_outputs)  # Use last timestep
    context_vector, attention_weights = attention_layer(query, encoder_outputs)
    
    # Expand context to match sequence length and combine with decoder outputs
    context_expanded = Lambda(lambda x: tf.expand_dims(x, 1))(context_vector)
    context_tiled = Lambda(lambda x: tf.tile(x, [1, max_len, 1]))(context_expanded)
    
    # Combine context with decoder outputs
    combined_outputs = Concatenate(axis=-1)([decoder_outputs, context_tiled])
    
    # Final dense layer
    dense = Dense(vocab_size, activation='softmax')
    outputs = TimeDistributed(dense)(combined_outputs)
    
    # Model
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    print(f"✅ Model oluşturuldu!")
    print(f"   Parameters: {model.count_params():,}")
    
    # Compile
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create the model
seq2seq_attention_model = create_seq2seq_with_attention()

print_section("SELF-ATTENTION IMPLEMENTATION")

class MultiHeadSelfAttention(Layer):
    """Multi-Head Self-Attention implementation"""
    
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim) 
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)
        
    def attention(self, query, key, value):
        """Scaled dot-product attention"""
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        """Separate heads for multi-head attention"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Linear transformations
        query = self.query_dense(inputs)
        key = self.key_dense(inputs) 
        value = self.value_dense(inputs)
        
        # Separate heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        # Attention
        attention, weights = self.attention(query, key, value)
        
        # Concatenate heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        
        # Final linear transformation
        output = self.combine_heads(concat_attention)
        
        return output

def demonstrate_self_attention():
    """Self-attention demonstration"""
    
    print("🔍 Self-Attention demonstration...")
    
    # Parameters
    batch_size = 2
    seq_len = 8
    embed_dim = 64
    num_heads = 4
    
    # Create sample input
    inputs = tf.random.normal((batch_size, seq_len, embed_dim))
    print(f"Input shape: {inputs.shape}")
    
    # Create self-attention layer
    self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
    
    # Apply self-attention
    output = self_attention(inputs)
    print(f"Output shape: {output.shape}")
    
    # Visualize attention patterns
    # For visualization, we'll create a simpler single-head version
    class SimpleAttention(Layer):
        def __init__(self):
            super(SimpleAttention, self).__init__()
            
        def call(self, inputs):
            # Simple scaled dot-product attention
            query = inputs
            key = inputs
            value = inputs
            
            score = tf.matmul(query, key, transpose_b=True)
            scaled_score = score / tf.math.sqrt(float(inputs.shape[-1]))
            weights = tf.nn.softmax(scaled_score, axis=-1)
            output = tf.matmul(weights, value)
            
            return output, weights
    
    simple_attention = SimpleAttention()
    simple_output, attention_weights = simple_attention(inputs)
    
    # Plot attention patterns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('🔍 Self-Attention Patterns', fontsize=14, fontweight='bold')
    
    # First sample attention matrix
    attention_matrix = attention_weights[0].numpy()
    
    im1 = axes[0].imshow(attention_matrix, cmap='Blues', interpolation='nearest')
    axes[0].set_title('Self-Attention Matrix (Sample 1)', fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Second sample attention matrix
    attention_matrix2 = attention_weights[1].numpy()
    
    im2 = axes[1].imshow(attention_matrix2, cmap='Reds', interpolation='nearest')
    axes[1].set_title('Self-Attention Matrix (Sample 2)', fontweight='bold')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    return self_attention, attention_weights

self_attention, attention_weights = demonstrate_self_attention()

print_section("ATTENTION-BASED RNN UYGULAMALARI")

def practical_attention_applications():
    """Pratik attention uygulamaları"""
    
    print("🚀 Practical attention applications...")
    
    # Example 1: Sentiment Analysis with Attention
    print("\n1️⃣ SENTIMENT ANALYSIS WITH ATTENTION:")
    print("-" * 40)
    
    # Synthetic text data (word indices)
    np.random.seed(42)
    vocab_size = 5000
    max_len = 50
    n_samples = 1000
    
    # Generate sequences with sentiment patterns
    X_text = []
    y_sentiment = []
    
    for i in range(n_samples):
        # Random sequence
        seq = np.random.randint(1, vocab_size, max_len)
        
        # Add sentiment-specific patterns
        if i % 2 == 0:  # Positive
            # Add positive words at random positions
            pos_positions = np.random.choice(max_len, 5, replace=False)
            seq[pos_positions] = np.random.randint(vocab_size-100, vocab_size)  # High indices = positive
            label = 1
        else:  # Negative
            # Add negative words at random positions  
            neg_positions = np.random.choice(max_len, 5, replace=False)
            seq[neg_positions] = np.random.randint(1, 100)  # Low indices = negative
            label = 0
            
        X_text.append(seq)
        y_sentiment.append(label)
    
    X_text = np.array(X_text)
    y_sentiment = np.array(y_sentiment)
    
    print(f"Text data: {X_text.shape}")
    print(f"Sentiment distribution: {np.bincount(y_sentiment)}")
    
    # Create sentiment analysis model with attention
    def create_sentiment_attention_model():
        inputs = Input(shape=(max_len,))
        embedding = Embedding(vocab_size, 128, mask_zero=True)(inputs)
        
        # Bidirectional LSTM
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embedding)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(lstm_out)
        attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention)
        
        # Weighted representation
        weighted_input = Lambda(lambda x: x[0] * x[1])([lstm_out, attention_weights])
        output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_input)
        
        # Classification
        dense1 = Dense(32, activation='relu')(output)
        dropout = Dropout(0.5)(dense1)
        predictions = Dense(1, activation='sigmoid')(dropout)
        
        model = Model(inputs=inputs, outputs=predictions)
        return model, attention_weights
    
    sentiment_model, _ = create_sentiment_attention_model()
    sentiment_model.compile(
        optimizer=Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Sentiment model: {sentiment_model.count_params():,} parameters")
    
    # Quick training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_sentiment, test_size=0.2, random_state=42
    )
    
    print("Training sentiment model...")
    history = sentiment_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=0
    )
    
    test_acc = sentiment_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"✅ Sentiment analysis accuracy: {test_acc:.4f}")
    
    return sentiment_model, X_test, y_test

sentiment_model, X_test_sent, y_test_sent = practical_attention_applications()

print_section("ATTENTION VİZUALİZASYONU")

def visualize_attention_interpretability():
    """Attention weights ile model interpretability"""
    
    print("🔍 Attention interpretability analysis...")
    
    # Create a model that outputs attention weights for analysis
    def create_interpretable_model():
        inputs = Input(shape=(50,))
        embedding = Embedding(5000, 128, mask_zero=True)(inputs)
        
        # Bidirectional LSTM
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embedding)
        
        # Attention with explicit output
        attention = Dense(1, activation='tanh')(lstm_out)
        attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention)
        
        # Weighted representation
        weighted_input = Lambda(lambda x: x[0] * x[1])([lstm_out, attention_weights])
        output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_input)
        
        # Classification
        dense1 = Dense(32, activation='relu')(output)
        predictions = Dense(1, activation='sigmoid')(dense1)
        
        # Model that outputs both predictions and attention weights
        model = Model(inputs=inputs, outputs=[predictions, attention_weights])
        return model
    
    interpretable_model = create_interpretable_model()
    
    # Use weights from trained model (transfer some layers)
    for i, layer in enumerate(sentiment_model.layers[:-3]):  # Skip last few layers
        if i < len(interpretable_model.layers) - 2:
            interpretable_model.layers[i].set_weights(layer.get_weights())
    
    # Get predictions and attention weights for test samples
    predictions, attention_weights = interpretable_model.predict(X_test_sent[:10], verbose=0)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('🔍 Attention Interpretability Analysis', fontsize=16, fontweight='bold')
    
    # Plot attention weights for different samples
    for i in range(6):
        row, col = i // 3, i % 3
        
        sample_attention = attention_weights[i, :, 0]
        sequence_positions = np.arange(len(sample_attention))
        
        axes[row, col].bar(sequence_positions, sample_attention, alpha=0.7,
                          color='blue' if y_test_sent[i] == 1 else 'red')
        
        axes[row, col].set_title(f'Sample {i+1} - {"Positive" if y_test_sent[i] == 1 else "Negative"}',
                                fontweight='bold')
        axes[row, col].set_xlabel('Sequence Position')
        axes[row, col].set_ylabel('Attention Weight')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add prediction confidence
        confidence = predictions[i, 0]
        axes[row, col].text(0.7, 0.9, f'Conf: {confidence:.3f}',
                           transform=axes[row, col].transAxes,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("🎯 Attention Analysis:")
    print("• Attention weights gösterir model hangi pozisyonlara odaklanıyor")
    print("• Yüksek attention weight = o pozisyon daha önemli")
    print("• Model interpretability için güçlü tool")
    print("• Bias ve fairness analysis için kullanılabilir")

visualize_attention_interpretability()

print_section("ATTENTION BEST PRACTICES")

best_practices = """
💡 ATTENTION MECHANISM BEST PRACTICES:

🎯 NE ZAMAN KULLANILMALI:
• Long sequence dependencies önemli olduğunda
• Interpretability gerektiğinde
• Variable length sequences ile çalışırken
• Bottleneck problem yaşandığında
• Multi-modal data fusion için

⚙️ IMPLEMENTATION İPUÇLARI:
1. **Attention Type Selection:**
   • Additive (Bahdanau): Encoder-decoder farklı boyutlarda
   • Multiplicative (Luong): Daha hızlı, aynı boyutlarda
   • Self-attention: Sequence internal relationships için

2. **Scaling & Normalization:**
   • Scaled dot-product attention kullan (√d_k ile böl)
   • Layer normalization ekle
   • Dropout attention weights'e uygula

3. **Multi-Head Attention:**
   • Farklı representation subspaces yakala
   • Head sayısını embed_dim'e göre ayarla
   • Parallel computation avantajı

4. **Memory Optimization:**
   • Attention matrix memory usage: O(n²)
   • Sparse attention patterns kullan
   • Gradient checkpointing
   • Mixed precision training

🚀 PERFORMANCE TİPS:
• Batch processing optimize et
• Mask padding tokens properly
• Use efficient attention implementations
• Consider local vs global attention trade-offs

🔍 DEBUG & ANALYSIS:
• Attention weights visualize et
• Attention patterns analyze et
• Gradient flow check et
• Ablation studies yap
"""

print(best_practices)

print_section("GELİŞMİŞ ATTENTION VARIANTS")

def advanced_attention_variants():
    """Gelişmiş attention çeşitleri"""
    
    print("🚀 Advanced attention variants...")
    
    # 1. Sparse Attention (simplified version)
    class SparseAttention(Layer):
        def __init__(self, embed_dim, sparsity_factor=4):
            super(SparseAttention, self).__init__()
            self.embed_dim = embed_dim
            self.sparsity_factor = sparsity_factor
            self.query_dense = Dense(embed_dim)
            self.key_dense = Dense(embed_dim)
            self.value_dense = Dense(embed_dim)
        
        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            
            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)
            
            # Create sparse attention pattern (local + some global)
            scores = tf.matmul(query, key, transpose_b=True)
            
            # Create mask for sparse pattern
            mask = tf.ones((seq_len, seq_len))
            
            # Local attention (diagonal band)
            for i in range(seq_len):
                for j in range(max(0, i-self.sparsity_factor), 
                              min(seq_len, i+self.sparsity_factor+1)):
                    mask = tf.tensor_scatter_nd_update(
                        mask, [[i, j]], [1.0])
            
            # Apply mask
            scores = scores + (mask - 1.0) * 1e9
            weights = tf.nn.softmax(scores, axis=-1)
            
            output = tf.matmul(weights, value)
            return output
    
    # 2. Cross Attention (for encoder-decoder)
    class CrossAttention(Layer):
        def __init__(self, embed_dim):
            super(CrossAttention, self).__init__()
            self.embed_dim = embed_dim
            self.query_dense = Dense(embed_dim)
            self.key_dense = Dense(embed_dim)
            self.value_dense = Dense(embed_dim)
        
        def call(self, decoder_input, encoder_output):
            query = self.query_dense(decoder_input)
            key = self.key_dense(encoder_output)
            value = self.value_dense(encoder_output)
            
            scores = tf.matmul(query, key, transpose_b=True)
            scaled_scores = scores / tf.math.sqrt(float(self.embed_dim))
            weights = tf.nn.softmax(scaled_scores, axis=-1)
            
            output = tf.matmul(weights, value)
            return output, weights
    
    print("✅ Advanced attention variants implemented:")
    print("1. 🎯 Sparse Attention: Reduces O(n²) to O(n√n) complexity")
    print("2. 🔄 Cross Attention: For encoder-decoder architectures")
    
    return SparseAttention, CrossAttention

SparseAttention, CrossAttention = advanced_attention_variants()

print_section("TRANSFORMER-STYLE ATTENTION")

def transformer_attention_demo():
    """Transformer-style attention demonstration"""
    
    print("🤖 Transformer-style attention demo...")
    
    class TransformerBlock(Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = MultiHeadSelfAttention(embed_dim, num_heads)
            self.ffn = tf.keras.Sequential([
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim)
            ])
            self.layernorm1 = BatchNormalization()
            self.layernorm2 = BatchNormalization()
            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)
        
        def call(self, inputs, training=None):
            attn_output = self.att(inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    
    # Create a small transformer
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    
    inputs = Input(shape=(20, embed_dim))
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    outputs = transformer_block(inputs)
    
    transformer_model = Model(inputs=inputs, outputs=outputs)
    
    print(f"✅ Transformer block created: {transformer_model.count_params():,} params")
    
    return transformer_model

transformer_model = transformer_attention_demo()

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu Attention Mechanism modülünde öğrendikleriniz:")
print("  1. 🔍 Attention mechanism teorisi ve matematik")
print("  2. ⚖️ Bahdanau vs Luong attention karşılaştırması")
print("  3. 🔧 Custom attention layer implementations")
print("  4. 🚀 Seq2Seq with attention architecture")
print("  5. 🎯 Self-attention ve multi-head attention")
print("  6. 📊 Attention interpretability ve visualization")
print("  7. 🔄 Advanced attention variants")
print("  8. 🤖 Transformer-style attention blocks")

print(f"\n🏆 ATTENTION BENEFITS:")
print("  • Long-term dependencies ✓")
print("  • Better gradient flow ✓")
print("  • Model interpretability ✓")
print("  • Variable sequence lengths ✓")
print("  • Parallel computation ✓")

print("\n💡 Ana çıkarımlar:")
print("  • Attention RNN'lerdeki bottleneck'i çözer")
print("  • Farklı attention types farklı use case'ler için optimize")
print("  • Self-attention Transformer'ların temelini oluşturur")
print("  • Interpretability için güçlü tool sağlar")
print("  • Computational cost vs performance trade-off önemli")

print("\n🎯 Kullanım önerileri:")
print("  • Long sequences: Multi-head self-attention")
print("  • Seq2seq tasks: Cross-attention")
print("  • Memory constraints: Sparse attention")
print("  • Interpretability: Attention weight analysis")

print("\n📚 Sonraki modül: 13_encoder_decoder.py")
print("Encoder-Decoder architectures ile sequence-to-sequence learning!")

print("\n" + "=" * 70)
print("✅ ATTENTION MECHANISM MODÜLÜ TAMAMLANDI!")
print("=" * 70)