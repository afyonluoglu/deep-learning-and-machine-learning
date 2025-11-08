"""
🔤 VANILLA RNN - TEMEL RNN DETAYILI AÇIKLAMA
===========================================

Bu dosya temel Vanilla RNN'lerin detaylı implementasyonunu ve
limitasyonlarını açıklar. LSTM ve GRU ile karşılaştırma yapar.

Vanilla RNN Özellikleri:
1. En basit RNN türü
2. Hidden state sadece tanh aktivasyonu kullanır
3. Vanishing gradient problemi yaşar
4. Kısa vadeli bağımlılıklar için uygun

Öğreneceğiniz konular:
- Vanilla RNN mathematiksel formülasyonu
- Manual implementation
- Limitasyonlar ve problemler
- LSTM/GRU ile performans karşılaştırması
"""

from calendar import EPOCH
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from datetime import datetime



def print_section(title, char="=", single_line:bool=False, width=55):
    title = title
    if not single_line:
        print(f"{char*width}")
    if char == "=":
        title = "📋 "+ title
    print(title)
    print(f"{char*width}")

print_section("🔤 VANILLA RNN - TEMEL RNN DETAYILI AÇIKLAMA", char="#", width=80)

print_section("VANILLA RNN MATEMATIKSEL TEMELLER")

print_section("📐 VANILLA RNN FORMÜLLERI:", char="-", single_line=True, width=35)

print("h_t = tanh(W_hh * h_t-1 + W_xh * x_t + b_h)")
print("y_t = W_hy * h_t + b_y")
print("")
print("Burada:")
print("• h_t     : t anındaki hidden state")
print("• x_t     : t anındaki input")
print("• W_hh    : hidden-to-hidden weight matrix")
print("• W_xh    : input-to-hidden weight matrix") 
print("• W_hy    : hidden-to-output weight matrix")
print("• b_h, b_y: bias vektörleri")

def manual_vanilla_rnn_step(x_t, h_prev, W_hh, W_xh, b_h):
    """
    Vanilla RNN adımını manuel olarak uygular
    """
    h_t = np.tanh(np.dot(W_hh, h_prev) + np.dot(W_xh, x_t) + b_h)
    return h_t

def demonstrate_manual_rnn():
    """Manuel RNN implementasyonunu gösterir"""
    
    print_section("MANUEL RNN IMPLEMENTASYONU")
    
    # Parametreler
    hidden_size = 4
    input_size = 2
    sequence_length = 8
    
    # Rastgele ağırlıklar
    np.random.seed(42)
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hy = np.random.randn(1, hidden_size) * 0.1
    b_h = np.zeros((hidden_size, 1))
    b_y = np.zeros((1, 1))
    
    # Örnek sequence
    sequence = []
    for i in range(sequence_length):
        x = np.array([[np.sin(i * 0.5)], [np.cos(i * 0.5)]])
        sequence.append(x)
    
    print(f"🔧 Parametreler:")
    print(f"   Hidden size     : {hidden_size}")
    print(f"   Input size      : {input_size}")
    print(f"   Sequence length : {sequence_length}")
    
    # RNN forward pass
    hidden_states = []
    outputs = []
    h = np.zeros((hidden_size, 1))
    
    print(f"\n🔄 RNN Forward Pass:")
    for t, x_t in enumerate(sequence):
        h = manual_vanilla_rnn_step(x_t, h, W_hh, W_xh, b_h)
        y_t = np.dot(W_hy, h) + b_y
        
        hidden_states.append(h.copy())
        outputs.append(y_t.copy())
        
        x_t_formatted = ", ".join([f"{x:11.8f}" for x in x_t.flatten()])
        h_t_formatted = ", ".join([f"{h:11.8f}" for h in h.flatten()])
        y_t_formatted = f"{y_t[0,0]:.3f}"
        print(f"   t={t}: x_t=[{x_t_formatted}]   h_t=[{h_t_formatted}]    y_t=[{y_t_formatted}]")

    return np.array(hidden_states), np.array(outputs)

hidden_states, outputs = demonstrate_manual_rnn()

print_section("VANILLA RNN PROBLEMLERI")

def demonstrate_vanishing_gradient():
    """Vanishing gradient problemini gösterir"""

    print_section("⚠️  VANISHING GRADIENT PROBLEM:", char="-", single_line=True, width=35)

    # Farklı sequence uzunlukları test et
    sequence_lengths = [5, 10, 20, 50, 100]
    final_gradients = []
    
    for seq_len in sequence_lengths:
        # Basit toy problem
        np.random.seed(42)
        X = np.random.randn(1, seq_len, 1)
        y = np.array([[1.0]])  # Hedef
        
        # Vanilla RNN modeli
        model = Sequential([
            Input(shape=(seq_len, 1)),  # Giriş katmanı
            SimpleRNN(10, activation='tanh'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print(f"Model compiled... {seq_len} sequence length")

        # İlk ağırlıkları kaydet
        initial_weights = model.get_weights()
        
        # Bir adım gradient hesapla
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = tf.keras.losses.mse(y, pred)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # İlk katmanın gradientini al
        if gradients[0] is not None:
            grad_norm = tf.norm(gradients[0]).numpy()
            final_gradients.append(grad_norm)
        else:
            final_gradients.append(0.0)
    
    # Görselleştir
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(sequence_lengths, final_gradients, 'ro-', linewidth=2, markersize=8)
    plt.title('Gradient Norm vs Sequence Length', fontweight='bold')
    plt.xlabel('Sequence Length')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Hidden state evrimi
    plt.subplot(2, 2, 2)
    for i in range(min(4, hidden_states.shape[2])):
        plt.plot(hidden_states[:, 0, i], label=f'Hidden {i+1}', linewidth=2)
    plt.title('Hidden State Evolution', fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('Hidden State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Aktivasyon saturation analizi
    plt.subplot(2, 2, 3)
    tanh_input = np.linspace(-5, 5, 1000)
    tanh_output = np.tanh(tanh_input)
    tanh_derivative = 1 - tanh_output**2
    
    plt.plot(tanh_input, tanh_output, 'b-', label='tanh(x)', linewidth=2)
    plt.plot(tanh_input, tanh_derivative, 'r-', label="tanh'(x)", linewidth=2)
    plt.title('Tanh Activation & Derivative', fontweight='bold')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Problem illustration
    plt.subplot(2, 2, 4)
    problems = ['Vanishing\nGradient', 'Limited\nMemory', 'Slow\nTraining', 'Poor Long\nDependencies']
    severity = [0.9, 0.8, 0.6, 0.95]
    colors = ['red', 'orange', 'yellow', 'darkred']
    
    bars = plt.bar(problems, severity, color=colors, alpha=0.7)
    plt.title('Vanilla RNN Problems', fontweight='bold')
    plt.ylabel('Severity (0-1)')
    plt.ylim(0, 1)
    
    for bar, sev in zip(bars, severity):
        plt.text(bar.get_x() + bar.get_width()/2, sev + 0.02, 
                f'{sev:.1f}', ha='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("📊 Sonuçlar:")
    print(f"   • Kısa sequence (5): Gradient norm = {final_gradients[0]:.6f}")
    print(f"   • Uzun sequence (100): Gradient norm = {final_gradients[-1]:.6f}")
    print(f"   • Gradient azalma oranı: {final_gradients[-1]/final_gradients[0]:.2e}")

demonstrate_vanishing_gradient()

print_section("VANILLA RNN vs LSTM vs GRU KARŞILAŞTIRMA")

def compare_rnn_types():
    """Farklı RNN türlerini karşılaştırır"""
    
    # Veri oluştur
    print("📊 Karşılaştırma verisi oluşturuluyor...")
    
    np.random.seed(42)
    seq_length = 50
    n_features = 1
    n_samples = 1000
    
    # Uzun vadeli bağımlılık gerektiren veri
    X = []
    y = []
    
    for i in range(n_samples):
        # İlk 10 değer önemli sinyal içeriyor
        seq = np.random.randn(seq_length, n_features)
        important_signal = np.random.choice([1, -1]) * 2
        seq[:5] += important_signal  # İlk 5 değere sinyal ekle
        
        # Hedef: İlk 5 değerin ortalamasının işareti
        target = 1 if np.mean(seq[:5]) > 0 else 0
        
        X.append(seq)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ Veri hazırlandı: {X.shape}, Hedef: {len(np.unique(y))} sınıf")
    
    # Modeller
    models = {}
    histories = {}
    
    print("\n🏗️ Modeller oluşturuluyor ve eğitiliyor...")
    EPOCHS_FOR_MODELS = 30
    LAYER_1_NEURONS = 32
    LAYER_2_NEURONS = 16
    ADAM_PARAMETER = 0.001
    BATCH_SIZE = 32
    ACTIVATION = "relu"
    OUT_ACTIVATION = "sigmoid"
    LOSS_ALGORITHM = "binary_crossentropy"
    METRICS = ["accuracy"]

    print(f"   • Epochs         : {EPOCHS_FOR_MODELS}")
    print(f"   • Layer 1 Neurons: {LAYER_1_NEURONS}")
    print(f"   • Layer 2 Neurons: {LAYER_2_NEURONS}")
    print(f"   • Adam Parameter : {ADAM_PARAMETER}")
    print(f"   • Batch Size     : {BATCH_SIZE}")
    print(f"   • Sequence Length: {seq_length}")
    print(f"   • Features       : {n_features}")
    print(f"   • Loss Algorithm : {LOSS_ALGORITHM}")
    print(f"   • Metrics        : {METRICS}")

    # Vanilla RNN
    print("   📊 Vanilla RNN...")
    start = tf.timestamp()
    vanilla_rnn = Sequential([
        Input(shape=(seq_length, n_features)),         
        SimpleRNN(LAYER_1_NEURONS),
        Dense(LAYER_2_NEURONS, activation=ACTIVATION),
        Dense(1, activation=OUT_ACTIVATION)
    ])
    vanilla_rnn.compile(optimizer=Adam(ADAM_PARAMETER), loss=LOSS_ALGORITHM, metrics=METRICS)
    
    history_vanilla = vanilla_rnn.fit(X_train, y_train, 
                                     validation_data=(X_test, y_test),
                                     epochs=EPOCHS_FOR_MODELS, batch_size=BATCH_SIZE, verbose=0)
    
    models['Vanilla RNN'] = vanilla_rnn
    histories['Vanilla RNN'] = history_vanilla

    end_time = tf.timestamp()
    training_time = end_time - start
    print(f"   • Eğitim süresi: {training_time:.2f} saniye")
    

    # LSTM
    print("   📊 LSTM...")
    start = tf.timestamp()
    lstm = Sequential([
        Input(shape=(seq_length, n_features)),         
        LSTM(LAYER_1_NEURONS),
        Dense(LAYER_2_NEURONS, activation=ACTIVATION),
        Dense(1, activation=OUT_ACTIVATION)
    ])
    lstm.compile(optimizer=Adam(ADAM_PARAMETER), loss=LOSS_ALGORITHM, metrics=METRICS)
    
    history_lstm = lstm.fit(X_train, y_train,
                           validation_data=(X_test, y_test),
                           epochs=EPOCHS_FOR_MODELS, batch_size=BATCH_SIZE, verbose=0)
    
    models['LSTM'] = lstm
    histories['LSTM'] = history_lstm

    end_time = tf.timestamp()
    training_time = end_time - start
    print(f"   • Eğitim süresi: {training_time:.2f} saniye")


    # GRU
    print("   📊 GRU...")
    start = tf.timestamp()
    gru = Sequential([
        Input(shape=(seq_length, n_features)),         
        GRU(LAYER_1_NEURONS),
        Dense(LAYER_2_NEURONS, activation=ACTIVATION),
        Dense(1, activation=OUT_ACTIVATION)
    ])
    gru.compile(optimizer=Adam(ADAM_PARAMETER), loss=LOSS_ALGORITHM, metrics=METRICS)
    
    history_gru = gru.fit(X_train, y_train,
                         validation_data=(X_test, y_test),
                         epochs=EPOCHS_FOR_MODELS, batch_size=BATCH_SIZE, verbose=0)
    
    models['GRU'] = gru
    histories['GRU'] = history_gru

    end_time = tf.timestamp()
    training_time = end_time - start
    print(f"   • Eğitim süresi: {training_time:.2f} saniye")


    print("✅ Tüm modeller eğitildi!")
    
    # Sonuçları karşılaştır
    plt.figure(figsize=(15, 10))
    
    # Loss karşılaştırması
    plt.subplot(2, 3, 1)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train', linewidth=2)
        plt.plot(history.history['val_loss'], '--', label=f'{name} Val', linewidth=2)
    plt.title('Training Loss Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy karşılaştırması
    plt.subplot(2, 3, 2)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train', linewidth=2)
        plt.plot(history.history['val_accuracy'], '--', label=f'{name} Val', linewidth=2)
    plt.title('Accuracy Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performans
    plt.subplot(2, 3, 3)
    final_accuracies = []
    model_names = []
    
    print("   • Final Validation Accuracy:")
    for name, history in histories.items():
        final_acc = history.history['val_accuracy'][-1]
        print(f"{name:>15} final validation accuracy: {final_acc:.4f}")
        final_accuracies.append(final_acc)
        model_names.append(name)
    
    bars = plt.bar(model_names, final_accuracies, 
                   color=['red', 'blue', 'green'], alpha=0.7)
    plt.title('Final Validation Accuracy', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.01, 
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)

    print("   • Model Complexity:")
    # Model karmaşıklığı
    plt.subplot(2, 3, 4)
    param_counts = []
    for name, model in models.items():
        param_counts.append(model.count_params())
        print(f"{name:>20} parameter count: {model.count_params():,}")
    
    bars = plt.bar(model_names, param_counts, 
                   color=['red', 'blue', 'green'], alpha=0.7)
    plt.title('Model Parameters', fontweight='bold')
    plt.ylabel('Parameter Count')
    
    for bar, count in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2, count + max(param_counts)*0.01, 
                f'{count:,}', ha='center', fontweight='bold', rotation=45)
    
    plt.grid(True, alpha=0.3)
    
    # Eğitim süresi (simulated)
    plt.subplot(2, 3, 5)
    training_times = [1.0, 3.2, 2.1]  # Relative times
    bars = plt.bar(model_names, training_times, 
                   color=['red', 'blue', 'green'], alpha=0.7)
    plt.title('Relative Training Time', fontweight='bold')
    plt.ylabel('Relative Time')
    
    for bar, time in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, time + 0.05, 
                f'{time:.1f}x', ha='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Özellik karşılaştırması
    plt.subplot(2, 3, 6)
    features = ['Simple', 'Fast', 'Memory', 'Long Dep.', 'Stable']
    vanilla_scores = [1.0, 1.0, 0.3, 0.2, 0.4]
    lstm_scores = [0.6, 0.5, 1.0, 1.0, 0.9]
    gru_scores = [0.7, 0.7, 0.9, 0.9, 0.8]
    
    x = np.arange(len(features))
    width = 0.25
    
    plt.bar(x - width, vanilla_scores, width, label='Vanilla RNN', alpha=0.7, color='red')
    plt.bar(x, lstm_scores, width, label='LSTM', alpha=0.7, color='blue')
    plt.bar(x + width, gru_scores, width, label='GRU', alpha=0.7, color='green')
    
    plt.title('Feature Comparison', fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Score (0-1)')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performans raporu
    print_section(f"📊 PERFORMANS RAPORU:", char="-", single_line=True, width=35)
    
    for name, history in histories.items():
        final_loss = history.history['val_loss'][-1]
        final_acc = history.history['val_accuracy'][-1]
        param_count = models[name].count_params()
        
        print(f"\n{name}:")
        print(f"   Final Validation Loss    : {final_loss:.4f}")
        print(f"   Final Validation Accuracy: {final_acc:.4f}")
        print(f"   Parameters               : {param_count:,}")
    
    # En iyi modeli belirle
    best_model = max(histories.items(), key=lambda x: x[1].history['val_accuracy'][-1])
    print(f"\n🏆 EN İYİ MODEL: {best_model[0]}")
    print(f"   Accuracy: {best_model[1].history['val_accuracy'][-1]:.4f}")

compare_rnn_types()

print_section("VANILLA RNN KULLANIM ÖNERİLERİ")

def print_usage_guidelines():
    """Vanilla RNN kullanım önerilerini yazdırır"""
    print("💡 VANILLA RNN NE ZAMAN KULLANILMALI?")
    print("-" * 40)
    print("✅ UYGUN DURUMLAR:")
    print("   • Kısa sequence'ler (< 10-20 adım)")
    print("   • Basit temporal pattern'ler")
    print("   • Hızlı prototyping")
    print("   • Sınırlı hesaplama kaynakları")
    print("   • Eğitici/öğrenme amaçlı")

    print("\n❌ UYGUN OLMAYAN DURUMLAR:")
    print("   • Uzun sequence'ler (> 20-30 adım)")
    print("   • Karmaşık long-term dependencies")
    print("   • Üretim seviyesi uygulamalar")
    print("   • Yüksek accuracy gereken durumlar")

    print("\n🔧 VANILLA RNN İYİLEŞTİRME YÖNTEMLERİ:")
    print("   1. Gradient Clipping: Exploding gradient önler")
    print("   2. Smaller Learning Rate: Daha stabil eğitim")
    print("   3. Regularization: Dropout, L2 regularization")
    print("   4. Proper Weight Initialization: Xavier/He init")
    print("   5. Batch Normalization: Internal covariate shift")

print_usage_guidelines()

def demonstrate_improvements():
    """Vanilla RNN iyileştirmelerini gösterir"""
    
    print_section("VANILLA RNN İYİLEŞTİRME ÖRNEĞİ")
    
    MAX_DATA_POINTS = 1000
    TIME_SERIES_LENGTH = 35  # Time series büyüdükçe improved versiyon çok öne geçiyor
    LAYER_1_NEURONS = 20 
    LAYER_2_NEURONS = 10
    LOSS_ALGORITHM = "binary_crossentropy"
    OUTPUT_LAYER_ACTIVATION = "sigmoid"
    MODEL2_DROPOUT = 0.2
    MODEL_2_LEARNING_RATE = 0.005
    EPOCHS = 40

    print(f"   • Data Points          : {MAX_DATA_POINTS}")
    print(f"   • Time Series Length   : {TIME_SERIES_LENGTH}")
    print(f"   • Layer 1 Neurons      : {LAYER_1_NEURONS}")
    print(f"   • Layer 2 Neurons      : {LAYER_2_NEURONS}")
    print(f"   • Loss Algorithm       : {LOSS_ALGORITHM}")
    print(f"   • Output Layer Activation: {OUTPUT_LAYER_ACTIVATION}")
    print(f"   • Model 2 Dropout      : {MODEL2_DROPOUT}")
    print(f"   • Model 2 Learning Rate: {MODEL_2_LEARNING_RATE}")
    print(f"   • Epochs               : {EPOCHS}")

    # Veri hazırla
    np.random.seed(42)
    X = np.random.randn(MAX_DATA_POINTS, TIME_SERIES_LENGTH, 1)

    # Örnek set Set
    y = np.sum(X[:, :5, 0], axis=1) > 0  # İlk 5 adımın toplamı = tüm data_pointlerin ilk 5 zaman verisinin 0. elemanlarının toplamı
    y = y.astype(int)   # Binary sınıflandırma: TRUE ise 1, FALSE ise 0 döndürür
    # Burada y pozitif ise sonuç 1 negatif ise 0 olur

    # Daha karmaşık bir örnek: İlk 3 ve son 3 değerin kombinasyonu
    y1 = np.sum(X[:, :3, 0], axis=1) > 0
    y2 = np.sum(X[:, -3:, 0], axis=1) > 0
    y = (y1 & y2).astype(int)  # Daha zor pattern
    
    print(f"✅ Veri hazırlandı: {X.shape}, Hedef: {len(np.unique(y))} sınıf")

    # Basit Vanilla RNN
    basic_model = Sequential([
        Input(shape=(TIME_SERIES_LENGTH, 1)),
        SimpleRNN(LAYER_1_NEURONS),
        Dense(1, activation=OUTPUT_LAYER_ACTIVATION)
    ])
    basic_model.compile(optimizer='adam', loss=LOSS_ALGORITHM, metrics=['accuracy'])
            
    # İyileştirilmiş Vanilla RNN
    improved_model = Sequential([
        Input(shape=(TIME_SERIES_LENGTH, 1)),
        SimpleRNN(LAYER_1_NEURONS, dropout=MODEL2_DROPOUT),
        Dense(LAYER_2_NEURONS, activation='relu'),
        Dropout(MODEL2_DROPOUT),
        Dense(1, activation=OUTPUT_LAYER_ACTIVATION)
    ])
    improved_model.compile(
        optimizer=Adam(learning_rate=MODEL_2_LEARNING_RATE, clipnorm=1.0),  # clipnorm: Gradient clipping
        loss=LOSS_ALGORITHM, 
        metrics=['accuracy']
    )
    
    print(f"Basic Model   : Input ({TIME_SERIES_LENGTH},1) -> SimpleRNN({LAYER_1_NEURONS}) -> Dense(1)")
    print(f"Improved Model: Input ({TIME_SERIES_LENGTH},1) -> SimpleRNN({LAYER_1_NEURONS}, dropout={MODEL2_DROPOUT}) -> Dense({LAYER_2_NEURONS}, relu) -> Dropout({MODEL2_DROPOUT}) -> Dense(1)")
    print("📊 Modeller eğitiliyor...")
    
    # Eğitim
    history_basic = basic_model.fit(X, y, epochs=EPOCHS, validation_split=0.2, verbose=0)
    history_improved = improved_model.fit(X, y, epochs=EPOCHS, validation_split=0.2, verbose=0)

    # Karşılaştırma
    plt.figure(figsize=(13, 6))

    plt.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.16)
    
    current_time = datetime.now()
    params_text = (
        f"{current_time.strftime('%d-%m-%Y  (%H:%M)')} "
        f"  Data Points: {MAX_DATA_POINTS}  |"
        f"  Time Series Length: {TIME_SERIES_LENGTH}  |"
        f"  Layer 1 Neurons: {LAYER_1_NEURONS}  |"
        f"  Layer 2 Neurons: {LAYER_2_NEURONS}  |"
        f"  Loss Algorithm: {LOSS_ALGORITHM}\n"
        f"  Output Layer Activation: {OUTPUT_LAYER_ACTIVATION}  |"
        f"  Model 2 Dropout: {MODEL2_DROPOUT}  |"
        f"  Model 2 Learning Rate: {MODEL_2_LEARNING_RATE}  |"
        f"  Epochs: {EPOCHS}"
    )
    plt.figtext(0.01, 0.02, params_text, ha='left', va='bottom', 
                fontsize=11, color='black', 
                bbox=dict(facecolor="#A9E5FF", alpha=0.9))
        
    plt.subplot(1, 2, 1)
    plt.plot(history_basic.history['loss'], label='Basic - Train', linewidth=2)
    plt.plot(history_basic.history['val_loss'], label='Basic - Val', linewidth=2)
    plt.plot(history_improved.history['loss'], '--', label='Improved - Train', linewidth=2)
    plt.plot(history_improved.history['val_loss'], '--', label='Improved - Val', linewidth=2)
    plt.title('Loss Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_basic.history['accuracy'], label='Basic - Train', linewidth=2)
    plt.plot(history_basic.history['val_accuracy'], label='Basic - Val', linewidth=2)
    plt.plot(history_improved.history['accuracy'], '--', label='Improved - Train', linewidth=2)
    plt.plot(history_improved.history['val_accuracy'], '--', label='Improved - Val', linewidth=2)
    plt.title('Accuracy Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    plt.show()
    
    # Sonuçları yazdır
    basic_final = history_basic.history['val_accuracy'][-1]
    improved_final = history_improved.history['val_accuracy'][-1]
    
    print(f"📊 SONUÇLAR:")
    print(f"   Basit Model         : {basic_final:.4f}")
    print(f"   İyileştirilmiş Model: {improved_final:.4f}")
    print(f"   İyileştirme         : {(improved_final - basic_final)*100:+.2f}%")

demonstrate_improvements()

def summary_and_results():
    """Özet ve sonuçları yazdırır"""
    print_section("ÖZET VE SONUÇLAR")

    print("✅ Bu Vanilla RNN modülünde öğrendikleriniz:")
    print("  1. 🧮  Vanilla RNN matematiksel formülasyonu")
    print("  2. 🔧  Manuel RNN implementasyonu")
    print("  3. ⚠️  Vanishing gradient problemi")
    print("  4. 📊  LSTM/GRU ile performans karşılaştırması")
    print("  5. 💡  Vanilla RNN kullanım alanları")
    print("  6. 🔧  İyileştirme teknikleri")

    print("\n💡 Ana çıkarımlar:")
    print("  • Vanilla RNN basit ama sınırlı")
    print("  • Kısa sequence'ler için yeterli")
    print("  • LSTM/GRU uzun sequence'ler için daha iyi")
    print("  • Doğru tekniklerle iyileştirilebilir")

    print("\n📚 Sonraki modül: 06_gru_example.py")
    print("GRU'nun LSTM'e göre avantajlarını göreceğiz!")

summary_and_results()
