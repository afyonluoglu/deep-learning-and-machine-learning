"""
🔄 BİDİRECTİONAL RNN - İLERİ VE GERİ YÖNLÜ RNN'LER
=================================================

Bu dosya Bidirectional RNN'lerin detaylı teorisi ve uygulamalarını açıklar.
İki yönlü bilgi akışı ile daha güçlü sequence modeling.

Bidirectional RNN Nedir?
- Hem forward hem backward direction'da bilgi işler
- Geçmiş ve gelecek context'i aynı anda kullanır
- Better context understanding
- Improved prediction accuracy

Kullanım Alanları:
- Natural Language Processing
- Speech recognition
- Sentiment analysis
- Machine translation
- Time series with full sequence available
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
                                   SimpleRNN, Input, Concatenate, 
                                   BatchNormalization, Conv1D, MaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔄 BİDİRECTİONAL RNN - İLERİ VE GERİ YÖNLÜ RNN'LER")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("BİDİRECTİONAL RNN TEORİSİ")

print("🧠 Bidirectional RNN Nedir?")
print("-" * 40)
print("• Sequence'i hem ileri hem geri yönde işler")
print("• İki ayrı RNN: forward RNN + backward RNN")
print("• Her time step'te geçmiş VE gelecek bilgisini kullanır")
print("• Output'lar birleştirilir (concatenate veya sum)")

print("\n🔄 BİDİRECTİONAL RNN YAPISI:")
print("-" * 30)
print("Forward RNN:  x₁ → x₂ → x₃ → x₄ → x₅")
print("              h₁ → h₂ → h₃ → h₄ → h₅")
print("")
print("Backward RNN: x₅ → x₄ → x₃ → x₂ → x₁")
print("              h₅ ← h₄ ← h₃ ← h₂ ← h₁")
print("")
print("Combined:     [h₁,h₁'] [h₂,h₂'] [h₃,h₃'] [h₄,h₄'] [h₅,h₅']")

print("\n⚖️ AVANTAJ VE DEZAVANTAJLAR:")
print("-" * 30)
print("✅ AVANTAJLAR:")
print("  • Tam context bilgisi (past + future)")
print("  • Daha iyi accuracy genellikle")
print("  • Belirsizlikleri çözme kabiliyeti")
print("  • NLP task'lerinde çok etkili")

print("\n❌ DEZAVANTAJLAR:")
print("  • 2x daha fazla parametre")
print("  • 2x daha yavaş training")
print("  • Real-time prediction için uygun değil")
print("  • Tam sequence gerekir")

def visualize_bidirectional_concept():
    """Bidirectional RNN kavramını görselleştirir"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bidirectional RNN Concept', fontsize=16, fontweight='bold')
    
    # 1. Forward vs Backward information flow
    x_pos = np.arange(5)
    
    # Forward flow
    axes[0, 0].arrow(0, 0.5, 4, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    for i in range(5):
        axes[0, 0].scatter(i, 0.5, s=200, c='lightblue', edgecolor='blue')
        axes[0, 0].text(i, 0.3, f'x{i+1}', ha='center', fontweight='bold')
        axes[0, 0].text(i, 0.7, f'h{i+1}→', ha='center', fontweight='bold', color='blue')
    
    axes[0, 0].set_title('Forward RNN', fontweight='bold', color='blue')
    axes[0, 0].set_xlim(-0.5, 4.5)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')
    
    # Backward flow
    axes[0, 1].arrow(4, 0.5, -4, 0, head_width=0.1, head_length=0.2, fc='red', ec='red')
    for i in range(5):
        axes[0, 1].scatter(i, 0.5, s=200, c='lightcoral', edgecolor='red')
        axes[0, 1].text(i, 0.3, f'x{i+1}', ha='center', fontweight='bold')
        axes[0, 1].text(i, 0.7, f'←h{i+1}', ha='center', fontweight='bold', color='red')
    
    axes[0, 1].set_title('Backward RNN', fontweight='bold', color='red')
    axes[0, 1].set_xlim(-0.5, 4.5)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')
    
    # Combined bidirectional
    for i in range(5):
        axes[1, 0].scatter(i, 0.7, s=150, c='lightblue', edgecolor='blue', label='Forward' if i==0 else "")
        axes[1, 0].scatter(i, 0.3, s=150, c='lightcoral', edgecolor='red', label='Backward' if i==0 else "")
        axes[1, 0].text(i, 0.1, f'x{i+1}', ha='center', fontweight='bold')
        
        # Combined output
        axes[1, 0].scatter(i, 0.5, s=100, c='purple', edgecolor='darkviolet')
        axes[1, 0].text(i, 0.05, f'[h{i+1}→,←h{i+1}]', ha='center', fontweight='bold', 
                       color='purple', fontsize=8, rotation=0)
    
    axes[1, 0].arrow(-0.2, 0.7, 4.4, 0, head_width=0.05, head_length=0.1, 
                    fc='blue', ec='blue', alpha=0.7)
    axes[1, 0].arrow(4.2, 0.3, -4.4, 0, head_width=0.05, head_length=0.1, 
                    fc='red', ec='red', alpha=0.7)
    
    axes[1, 0].set_title('Bidirectional RNN (Combined)', fontweight='bold', color='purple')
    axes[1, 0].set_xlim(-0.5, 4.5)
    axes[1, 0].set_ylim(-0.1, 0.9)
    axes[1, 0].legend()
    axes[1, 0].axis('off')
    
    # Comparison of context awareness
    time_steps = np.arange(1, 6)
    
    # Unidirectional context (only past)
    uni_context = np.cumsum(np.ones(5))  # [1, 2, 3, 4, 5]
    axes[1, 1].bar(time_steps - 0.2, uni_context, width=0.4, 
                  alpha=0.7, color='blue', label='Unidirectional (Forward)')
    
    # Bidirectional context (past + future)
    bi_context = [5, 5, 5, 5, 5]  # Full context at each step
    axes[1, 1].bar(time_steps + 0.2, bi_context, width=0.4,
                  alpha=0.7, color='purple', label='Bidirectional')
    
    axes[1, 1].set_title('Context Awareness Comparison', fontweight='bold')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Available Context')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(time_steps)
    
    plt.tight_layout()
    plt.show()
    
    print("🔍 Görselleştirme Açıklaması:")
    print("1. Forward RNN: Sadece geçmiş bilgiyi kullanır")
    print("2. Backward RNN: Gelecek bilgisini geriye doğru işler")
    print("3. Bidirectional: İkisini birleştirerek tam context sağlar")
    print("4. Context Comparison: Bidirectional her adımda tam bilgiye sahip")

visualize_bidirectional_concept()

print_section("BİDİRECTİONAL RNN İMPLEMENTASYONU")

def demonstrate_bidirectional_implementation():
    """Bidirectional RNN implementasyonunu gösterir"""
    
    print("🔧 Manual Bidirectional RNN Implementation...")
    
    # Basit sequence data
    np.random.seed(42)
    sequence_length = 10
    n_features = 3
    n_samples = 100
    
    # Synthetic sequence data (patterns that benefit from bidirectional processing)
    X = np.random.randn(n_samples, sequence_length, n_features)
    
    # Target: sequence ortasındaki maksimum değerin pozisyonu
    y = []
    for i in range(n_samples):
        seq_sum = np.sum(X[i], axis=1)  # Her time step'in toplamı
        max_pos = np.argmax(seq_sum)
        y.append(max_pos)
    
    y = np.array(y)
    
    print(f"Örnek veri: {X.shape}, Target: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Models to compare
    models = {}
    
    # 1. Forward-only LSTM
    forward_lstm = Sequential([
        LSTM(32, input_shape=(sequence_length, n_features)),
        Dense(16, activation='relu'),
        Dense(sequence_length, activation='softmax')  # Classification over positions
    ], name='Forward_LSTM')
    
    # 2. Backward-only LSTM (simulate by reversing input)
    backward_lstm = Sequential([
        LSTM(32, input_shape=(sequence_length, n_features), go_backwards=True),
        Dense(16, activation='relu'),
        Dense(sequence_length, activation='softmax')
    ], name='Backward_LSTM')
    
    # 3. Bidirectional LSTM
    bidirectional_lstm = Sequential([
        Bidirectional(LSTM(32), input_shape=(sequence_length, n_features)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(sequence_length, activation='softmax')
    ], name='Bidirectional_LSTM')
    
    # 4. Manual Bidirectional (for educational purposes)
    def create_manual_bidirectional():
        # Input
        inputs = Input(shape=(sequence_length, n_features))
        
        # Forward LSTM
        forward_lstm_layer = LSTM(32)(inputs)
        
        # Backward LSTM  
        backward_lstm_layer = LSTM(32, go_backwards=True)(inputs)
        
        # Concatenate
        merged = Concatenate()([forward_lstm_layer, backward_lstm_layer])
        
        # Dense layers
        dense1 = Dense(32, activation='relu')(merged)
        dense2 = Dense(16, activation='relu')(dense1)
        output = Dense(sequence_length, activation='softmax')(dense2)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    manual_bidirectional = create_manual_bidirectional()
    manual_bidirectional._name = 'Manual_Bidirectional'
    
    models = {
        'Forward Only': forward_lstm,
        'Backward Only': backward_lstm,
        'Bidirectional': bidirectional_lstm,
        'Manual Bidirectional': manual_bidirectional
    }
    
    # Compile models
    for name, model in models.items():
        model.compile(
            optimizer=Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ {name}: {model.count_params():,} parameters")
    
    # Train and compare
    print("\n🚀 Model eğitimleri başlıyor...")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    histories = {}
    
    for name, model in models.items():
        print(f"\n📊 {name} eğitiliyor...")
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, 
                                     restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0
        )
        
        histories[name] = history
        
        # Test evaluation
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'parameters': model.count_params(),
            'epochs': len(history.history['loss'])
        }
        
        print(f"   ✅ Test Accuracy: {test_acc:.4f}")
        print(f"   📊 Parameters: {model.count_params():,}")
    
    return models, histories, results, X_test, y_test

models, histories, results, X_test, y_test = demonstrate_bidirectional_implementation()

print_section("BİDİRECTİONAL RNN KARŞILAŞTIRMA ANALİZİ")

def comprehensive_bidirectional_analysis():
    """Bidirectional RNN'lerin kapsamlı analizi"""
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bidirectional RNN Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training curves
    colors = ['blue', 'red', 'purple', 'green']
    for i, (name, history) in enumerate(histories.items()):
        axes[0, 0].plot(history.history['accuracy'], color=colors[i], 
                       label=f'{name}', linewidth=2, alpha=0.8)
        axes[0, 0].plot(history.history['val_accuracy'], color=colors[i], 
                       linestyle='--', linewidth=2, alpha=0.8)
    
    axes[0, 0].set_title('Training & Validation Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in model_names]
    
    bars = axes[0, 1].bar(range(len(model_names)), accuracies, 
                         color=colors, alpha=0.7)
    axes[0, 1].set_title('Test Accuracy Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Model Type')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Parameter count vs accuracy
    param_counts = [results[name]['parameters'] for name in model_names]
    
    axes[0, 2].scatter(param_counts, accuracies, s=150, c=colors, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[0, 2].annotate(name.replace(' ', '\n'), 
                           (param_counts[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[0, 2].set_title('Parameters vs Accuracy', fontweight='bold')
    axes[0, 2].set_xlabel('Parameter Count')
    axes[0, 2].set_ylabel('Test Accuracy')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training efficiency (accuracy/epoch)
    training_epochs = [results[name]['epochs'] for name in model_names]
    efficiency = [acc/epochs for acc, epochs in zip(accuracies, training_epochs)]
    
    bars = axes[1, 0].bar(range(len(model_names)), efficiency,
                         color=colors, alpha=0.7)
    axes[1, 0].set_title('Training Efficiency (Acc/Epoch)', fontweight='bold')
    axes[1, 0].set_xlabel('Model Type')
    axes[1, 0].set_ylabel('Accuracy per Epoch')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Loss comparison
    test_losses = [results[name]['test_loss'] for name in model_names]
    
    bars = axes[1, 1].bar(range(len(model_names)), test_losses,
                         color=colors, alpha=0.7)
    axes[1, 1].set_title('Test Loss Comparison', fontweight='bold')
    axes[1, 1].set_xlabel('Model Type')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Model complexity analysis
    complexity_metrics = ['Parameters', 'Training Time', 'Memory Usage']
    
    # Normalize metrics for comparison (0-1 scale)
    normalized_params = np.array(param_counts) / max(param_counts)
    normalized_epochs = np.array(training_epochs) / max(training_epochs)
    normalized_memory = normalized_params * 1.2  # Approximate memory usage
    
    x = np.arange(len(model_names))
    width = 0.25
    
    axes[1, 2].bar(x - width, normalized_params, width, label='Parameters', alpha=0.7)
    axes[1, 2].bar(x, normalized_epochs, width, label='Training Time', alpha=0.7)
    axes[1, 2].bar(x + width, normalized_memory, width, label='Memory Usage', alpha=0.7)
    
    axes[1, 2].set_title('Model Complexity Comparison', fontweight='bold')
    axes[1, 2].set_xlabel('Model Type')
    axes[1, 2].set_ylabel('Normalized Complexity')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 DETAYLANDIRILMIŞ KARŞILAŞTIRMA:")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n🎯 {name}:")
        print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"   Test Loss: {result['test_loss']:.4f}")
        print(f"   Parameters: {result['parameters']:,}")
        print(f"   Training Epochs: {result['epochs']}")
        print(f"   Efficiency: {result['test_accuracy']/result['epochs']:.6f} acc/epoch")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    print(f"\n🏆 EN İYİ MODEL: {best_model}")
    print(f"   Accuracy: {results[best_model]['test_accuracy']:.4f}")

comprehensive_bidirectional_analysis()

print_section("BİDİRECTİONAL RNN UYGULAMA ÖRNEKLERİ")

def practical_bidirectional_examples():
    """Pratik bidirectional RNN örnekleri"""
    
    print("🚀 Pratik Bidirectional RNN uygulamaları...")
    
    # Example 1: Sequence classification with different patterns
    print("\n1️⃣ SEQUENCE CLASSIFICATION ÖRNEĞİ:")
    print("-" * 40)
    
    # Create synthetic sequence data with patterns
    np.random.seed(42)
    n_samples = 1000
    seq_len = 20
    n_features = 1
    
    # Different sequence patterns
    X_sequences = []
    y_labels = []
    
    for i in range(n_samples):
        if i % 4 == 0:
            # Pattern 1: Increasing then decreasing
            seq = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
            label = 0
        elif i % 4 == 1:
            # Pattern 2: Decreasing then increasing  
            seq = np.concatenate([np.linspace(1, 0, 10), np.linspace(0, 1, 10)])
            label = 1
        elif i % 4 == 2:
            # Pattern 3: High in middle
            seq = np.concatenate([np.linspace(0, 0.2, 5), np.linspace(0.2, 1, 10), np.linspace(1, 0.2, 5)])
            label = 2
        else:
            # Pattern 4: Low in middle
            seq = np.concatenate([np.linspace(0.8, 1, 5), np.linspace(1, 0, 10), np.linspace(0, 1, 5)])
            label = 3
        
        # Add noise
        seq += np.random.normal(0, 0.1, seq_len)
        X_sequences.append(seq.reshape(-1, 1))
        y_labels.append(label)
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_labels)
    
    print(f"Sequence data: {X_seq.shape}")
    print(f"Label distribution: {np.bincount(y_seq)}")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    # Compare unidirectional vs bidirectional
    uni_model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ], name='Unidirectional')
    
    bi_model = Sequential([
        Bidirectional(LSTM(32), input_shape=(seq_len, n_features)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ], name='Bidirectional')
    
    # Compile
    for model in [uni_model, bi_model]:
        model.compile(optimizer=Adam(0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    print(f"Unidirectional parameters: {uni_model.count_params():,}")
    print(f"Bidirectional parameters: {bi_model.count_params():,}")
    
    # Quick training
    print("Training models...")
    
    uni_history = uni_model.fit(X_train_seq, y_train_seq,
                               validation_data=(X_test_seq, y_test_seq),
                               epochs=20, batch_size=32, verbose=0)
    
    bi_history = bi_model.fit(X_train_seq, y_train_seq,
                             validation_data=(X_test_seq, y_test_seq),
                             epochs=20, batch_size=32, verbose=0)
    
    # Evaluate
    uni_acc = uni_model.evaluate(X_test_seq, y_test_seq, verbose=0)[1]
    bi_acc = bi_model.evaluate(X_test_seq, y_test_seq, verbose=0)[1]
    
    print(f"✅ Unidirectional accuracy: {uni_acc:.4f}")
    print(f"✅ Bidirectional accuracy: {bi_acc:.4f}")
    print(f"📈 Improvement: {((bi_acc - uni_acc) / uni_acc) * 100:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔄 Practical Bidirectional RNN Examples', fontsize=16, fontweight='bold')
    
    # Sample patterns
    for i, pattern_id in enumerate([0, 1, 2, 3]):
        sample_idx = np.where(y_test_seq == pattern_id)[0][0]
        sample_seq = X_test_seq[sample_idx].flatten()
        
        row, col = i // 2, i % 2
        axes[row, col].plot(sample_seq, 'b-', linewidth=2, marker='o', markersize=4)
        axes[row, col].set_title(f'Pattern {pattern_id} Example', fontweight='bold')
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Value')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add pattern description
        descriptions = [
            'Increasing → Decreasing',
            'Decreasing → Increasing', 
            'High in Middle',
            'Low in Middle'
        ]
        axes[row, col].text(0.05, 0.95, descriptions[pattern_id], 
                           transform=axes[row, col].transAxes,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return uni_model, bi_model, X_test_seq, y_test_seq

uni_model, bi_model, X_test_seq, y_test_seq = practical_bidirectional_examples()

print_section("BİDİRECTİONAL RNN BEST PRACTICES")

best_practices = """
💡 BİDİRECTİONAL RNN BEST PRACTICES:

🎯 NE ZAMAN KULLANILMALI:
• Tüm sequence önceden mevcut olduğunda
• Context'in her iki yönde de önemli olduğu durumlarda
• NLP tasks (sentiment analysis, named entity recognition)
• Speech recognition ve audio processing
• Biological sequence analysis
• Time series analysis (offline processing)

❌ NE ZAMAN KULLANILMAMALI:
• Real-time prediction gerektiğinde
• Streaming data işlerken
• Memory kısıtlı ortamlarda
• Çok uzun sequence'ler için (memory limitations)

🔧 OPTIMIZATION İPUÇLARI:
1. **Layer Design:**
   • Forward ve backward layer'ları dengele
   • Dropout kullanarak overfitting'i önle
   • BatchNormalization ekle

2. **Memory Management:**
   • return_sequences=False son layer için
   • Gradient clipping uygula
   • Mixed precision training kullan

3. **Training Strategy:**
   • Learning rate scheduling
   • Early stopping ile overfitting önle
   • Regularization techniques

4. **Architecture Choices:**
   • LSTM vs GRU trade-offs
   • Single vs multiple bidirectional layers
   • Dense layer combinations

🎛️ HİPERPARAMETRE REHBERİ:
• Hidden Units: 32-256 (data complexity'e göre)
• Dropout Rate: 0.2-0.5
• Learning Rate: 0.001-0.01
• Batch Size: 16-128
• Sequence Length: Task-specific optimize edin
"""

print(best_practices)

print_section("GELIŞMIŞ BİDİRECTİONAL UYGULAMALAR")

def advanced_bidirectional_applications():
    """Gelişmiş bidirectional RNN uygulamaları"""
    
    print("🚀 Gelişmiş bidirectional applications...")
    
    # Example: Multi-layer bidirectional with attention-like mechanism
    def create_advanced_bidirectional():
        inputs = Input(shape=(20, 10))  # sequence_length, features
        
        # First bidirectional layer
        bi_1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(inputs)
        bi_1 = BatchNormalization()(bi_1)
        
        # Second bidirectional layer
        bi_2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(bi_1)
        bi_2 = BatchNormalization()(bi_2)
        
        # Global pooling to handle variable length sequences
        pooled = tf.keras.layers.GlobalAveragePooling1D()(bi_2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(pooled)
        dense1 = Dropout(0.3)(dense1)
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        # Output
        output = Dense(1, activation='sigmoid')(dense2)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    advanced_model = create_advanced_bidirectional()
    
    print(f"✅ Advanced Bidirectional Model:")
    print(f"   Parameters: {advanced_model.count_params():,}")
    
    # Architecture summary
    print("\n🏗️ Architecture Details:")
    print("1. Input Layer: (sequence_length, features)")
    print("2. Bidirectional LSTM 1: 64 units, return_sequences=True")
    print("3. BatchNormalization")
    print("4. Bidirectional LSTM 2: 32 units, return_sequences=True") 
    print("5. BatchNormalization")
    print("6. GlobalAveragePooling1D")
    print("7. Dense layers with dropout")
    print("8. Output layer")
    
    return advanced_model

advanced_model = advanced_bidirectional_applications()

print_section("PERFORMANS VE BELLEKLESTİRME")

performance_tips = """
⚡ PERFORMANS OPTİMİZASYON İPUÇLARI:

💾 BELLEK YÖNETİMİ:
• CuDNN optimizations kullan (GPU)
• Mixed precision training (FP16)
• Gradient accumulation for large batches
• Dynamic padding for variable length sequences

🏃 HIZLANTRMA TEKNİKLERİ:
• Vectorized operations
• Batch processing
• Parallel computing
• Model distillation

📊 MONİTORİNG:
• Memory usage tracking
• Training speed metrics
• GPU utilization
• Loss curve analysis

🔄 MODEL COMPRESSİON:
• Pruning less important connections
• Quantization (8-bit, 16-bit)
• Knowledge distillation
• Architecture search
"""

print(performance_tips)

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu Bidirectional RNN modülünde öğrendikleriniz:")
print("  1. 🔄 Bidirectional RNN teorisi ve yapısı")
print("  2. 🧠 Forward vs backward information flow")
print("  3. ⚖️ Avantaj ve dezavantajlar")
print("  4. 🔧 Manual implementation techniques")
print("  5. 📊 Performance comparison ve analysis")
print("  6. 🚀 Practical application examples")
print("  7. 💡 Best practices ve optimization")
print("  8. 🏗️ Advanced architectures")

print(f"\n🏆 PERFORMANS ÖZETİ:")
for name, result in results.items():
    print(f"   {name:20s}: Acc={result['test_accuracy']:.4f}, Params={result['parameters']:,}")

print("\n💡 Ana çıkarımlar:")
print("  • Bidirectional RNN'ler genellikle daha iyi accuracy sağlar")
print("  • 2x parametre artışı performance gain'e değer")
print("  • Context-dependent task'lerde çok etkili")
print("  • Real-time applications için uygun değil")
print("  • Memory ve computational cost dikkate alınmalı")

print("\n🎯 Kullanım önerileri:")
print("  • NLP tasks için ideal")
print("  • Offline sequence analysis")
print("  • Pattern recognition applications")
print("  • Full sequence available scenarios")

print("\n📚 Sonraki modül: 12_attention_mechanism.py")
print("Attention mechanism ile sequence modeling'i bir üst seviyeye taşıyacağız!")

print("\n" + "=" * 70)
print("✅ BİDİRECTİONAL RNN MODÜLÜ TAMAMLANDI!")
print("=" * 70)