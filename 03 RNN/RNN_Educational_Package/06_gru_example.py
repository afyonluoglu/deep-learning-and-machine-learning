"""
🌀 GRU ÖRNEĞİ - GATED RECURRENT UNIT
====================================

Bu dosya GRU (Gated Recurrent Unit) ağlarını detaylı şekilde açıklar.
LSTM'e göre daha basit ama benzer performanslı olan GRU'yu öğrenin.

GRU Özellikleri:
1. LSTM'den daha basit (2 gate vs 3 gate)
2. Daha az parametre
3. Daha hızlı eğitim
4. Benzer performans

Gate'ler:
- Reset Gate (r_t): Geçmiş bilgiyi ne kadar unutacağını kontrol eder
- Update Gate (z_t): Ne kadar yeni bilgi alacağını kontrol eder

Kullanım Alanları:
- LSTM alternatifi olarak
- Sınırlı hesaplama kaynakları
- Hızlı prototyping
- Mobil/edge deployment
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import time

print("=" * 70)
print("🌀 GRU ÖRNEĞİ - GATED RECURRENT UNIT")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("GRU TEORİSİ VE MATEMATİK")

print("🧠 GRU (Gated Recurrent Unit) Nedir?")
print("-" * 40)
print("• LSTM'e alternatif olarak geliştirilmiş")
print("• Daha basit yapı (2 gate vs LSTM'in 3 gate'i)")
print("• Daha az parametre, daha hızlı eğitim")
print("• Çoğu durumda LSTM ile benzer performans")

print("\n🚪 GRU GATE MEKANİZMALARI:")
print("-" * 30)
print("1️⃣ RESET GATE (r_t):")
print("   r_t = σ(W_r · [h_t-1, x_t] + b_r)")
print("   → Geçmiş bilgiyi ne kadar unutacağını kontrol eder")
print("")
print("2️⃣ UPDATE GATE (z_t):")
print("   z_t = σ(W_z · [h_t-1, x_t] + b_z)")
print("   → Yeni bilgiyi ne kadar alacağını kontrol eder")
print("")
print("3️⃣ CANDIDATE HIDDEN STATE (h̃_t):")
print("   h̃_t = tanh(W_h · [r_t ⊙ h_t-1, x_t] + b_h)")
print("   → Yeni bilgi adayı")
print("")
print("4️⃣ FINAL HIDDEN STATE (h_t):")
print("   h_t = (1 - z_t) ⊙ h_t-1 + z_t ⊙ h̃_t")
print("   → Eski ve yeni bilgiyi karıştır")

def visualize_gru_gates():
    """GRU gate mekanizmalarını görselleştirir"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GRU Gate Mekanizmaları', fontsize=16, fontweight='bold')
    
    x = np.linspace(-5, 5, 1000)
    
    # Reset Gate
    reset_gate = 1 / (1 + np.exp(-x))  # Sigmoid
    axes[0, 0].plot(x, reset_gate, 'r-', linewidth=3, label='Reset Gate')
    axes[0, 0].set_title('Reset Gate (r_t)', fontweight='bold')
    axes[0, 0].set_xlabel('Input')
    axes[0, 0].set_ylabel('Gate Value (0-1)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    axes[0, 0].legend()
    
    # Update Gate
    update_gate = 1 / (1 + np.exp(-x))
    axes[0, 1].plot(x, update_gate, 'b-', linewidth=3, label='Update Gate')
    axes[0, 1].set_title('Update Gate (z_t)', fontweight='bold')
    axes[0, 1].set_xlabel('Input')
    axes[0, 1].set_ylabel('Gate Value (0-1)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    axes[0, 1].legend()
    
    # Candidate Hidden State
    candidate = np.tanh(x)
    axes[1, 0].plot(x, candidate, 'g-', linewidth=3, label='tanh(candidate)')
    axes[1, 0].set_title('Candidate Hidden State', fontweight='bold')
    axes[1, 0].set_xlabel('Input')
    axes[1, 0].set_ylabel('Candidate Value (-1 to 1)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].legend()
    
    # GRU vs LSTM karmaşıklık
    components = ['Gates', 'Parameters', 'Memory\nStates', 'Computation']
    gru_complexity = [2, 100, 1, 80]  # Relative values
    lstm_complexity = [3, 133, 2, 100]  # Relative values
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x_pos - width/2, gru_complexity, width, 
                          label='GRU', alpha=0.7, color='blue')
    bars2 = axes[1, 1].bar(x_pos + width/2, lstm_complexity, width,
                          label='LSTM', alpha=0.7, color='red')
    
    axes[1, 1].set_title('🔧 GRU vs LSTM Karmaşıklık', fontweight='bold')
    axes[1, 1].set_xlabel('Components')
    axes[1, 1].set_ylabel('Relative Complexity')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("🚪 Gate Açıklamaları:")
    print("-" * 20)
    print("1. 🔄 Reset Gate: Geçmiş bilgiyi ne kadar görmezden geleceğini kontrol eder")
    print("2. 🔄 Update Gate: Yeni bilgiyi ne kadar güncelleyeceğini kontrol eder")
    print("3. 🧠 Candidate: Reset edilmiş geçmiş + mevcut giriş")
    print("4. 🎯 Final: Eski ve yeni bilgiyi karıştırır")

visualize_gru_gates()

print_section("GRU vs LSTM DETAYILI KARŞILAŞTIRMA")

def comprehensive_gru_lstm_comparison():
    """GRU ve LSTM'i kapsamlı şekilde karşılaştırır"""
    
    # Farklı zorluk seviyelerinde veri setleri oluştur
    datasets = {}
    
    print("📊 Farklı zorluk seviyelerinde veri setleri oluşturuluyor...")
    
    np.random.seed(42)
    
    # Dataset 1: Kısa vadeli bağımlılık (kolay)
    seq_len_short = 10
    X_short = np.random.randn(1000, seq_len_short, 1)
    y_short = np.sum(X_short[:, -3:, 0], axis=1) > 0  # Son 3 değerin toplamı
    datasets['Short Term'] = (X_short, y_short.astype(int), seq_len_short)
    
    # Dataset 2: Orta vadeli bağımlılık (orta)
    seq_len_med = 25
    X_med = np.random.randn(1000, seq_len_med, 1)
    y_med = np.sum(X_med[:, :5, 0], axis=1) > 0  # İlk 5 değerin toplamı (20 adım sonra)
    datasets['Medium Term'] = (X_med, y_med.astype(int), seq_len_med)
    
    # Dataset 3: Uzun vadeli bağımlılık (zor)
    seq_len_long = 50
    X_long = np.random.randn(1000, seq_len_long, 1)
    # İlk ve son 3 değerin çarpımının işareti
    first_part = np.sum(X_long[:, :3, 0], axis=1)
    last_part = np.sum(X_long[:, -3:, 0], axis=1)
    y_long = (first_part * last_part) > 0
    datasets['Long Term'] = (X_long, y_long.astype(int), seq_len_long)
    
    results = {}
    training_times = {}
    
    print("🏗️ Modelleri eğitiyoruz ve karşılaştırıyoruz...")
    
    for dataset_name, (X, y, seq_len) in datasets.items():
        print(f"\n📊 {dataset_name} Dataset ({seq_len} adım):")
        
        # Train/test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        dataset_results = {}
        dataset_times = {}
        
        for model_type in ['GRU', 'LSTM']:
            print(f"   🔧 {model_type} eğitiliyor...")
            
            # Model oluştur
            if model_type == 'GRU':
                model = Sequential([
                    GRU(32, input_shape=(seq_len, 1)),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
            else:  # LSTM
                model = Sequential([
                    LSTM(32, input_shape=(seq_len, 1)),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
            
            model.compile(optimizer=Adam(0.001), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            # Eğitim süresi ölç
            start_time = time.time()
            
            history = model.fit(X_train, y_train,
                              validation_data=(X_test, y_test),
                              epochs=20, batch_size=32, verbose=0)
            
            training_time = time.time() - start_time
            
            # Sonuçları kaydet
            final_acc = history.history['val_accuracy'][-1]
            final_loss = history.history['val_loss'][-1]
            param_count = model.count_params()
            
            dataset_results[model_type] = {
                'accuracy': final_acc,
                'loss': final_loss,
                'parameters': param_count,
                'history': history
            }
            
            dataset_times[model_type] = training_time
            
            print(f"      ✅ Accuracy: {final_acc:.4f}, Time: {training_time:.1f}s")
        
        results[dataset_name] = dataset_results
        training_times[dataset_name] = dataset_times
    
    # Sonuçları görselleştir
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('GRU vs LSTM Kapsamlı Karşılaştırma', fontsize=16, fontweight='bold')
    
    # Accuracy karşılaştırması
    dataset_names = list(results.keys())
    gru_accs = [results[name]['GRU']['accuracy'] for name in dataset_names]
    lstm_accs = [results[name]['LSTM']['accuracy'] for name in dataset_names]
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, gru_accs, width, 
                          label='GRU', alpha=0.7, color='blue')
    bars2 = axes[0, 0].bar(x + width/2, lstm_accs, width,
                          label='LSTM', alpha=0.7, color='red')
    
    axes[0, 0].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dataset_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parametre sayısı karşılaştırması
    gru_params = [results[name]['GRU']['parameters'] for name in dataset_names]
    lstm_params = [results[name]['LSTM']['parameters'] for name in dataset_names]
    
    bars3 = axes[0, 1].bar(x - width/2, gru_params, width, 
                          label='GRU', alpha=0.7, color='blue')
    bars4 = axes[0, 1].bar(x + width/2, lstm_params, width,
                          label='LSTM', alpha=0.7, color='red')
    
    axes[0, 1].set_title('Parameter Count', fontweight='bold')
    axes[0, 1].set_xlabel('Dataset')
    axes[0, 1].set_ylabel('Parameters')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(dataset_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Eğitim süresi
    gru_times = [training_times[name]['GRU'] for name in dataset_names]
    lstm_times = [training_times[name]['LSTM'] for name in dataset_names]
    
    bars5 = axes[0, 2].bar(x - width/2, gru_times, width, 
                          label='GRU', alpha=0.7, color='blue')
    bars6 = axes[0, 2].bar(x + width/2, lstm_times, width,
                          label='LSTM', alpha=0.7, color='red')
    
    axes[0, 2].set_title('Training Time (seconds)', fontweight='bold')
    axes[0, 2].set_xlabel('Dataset')
    axes[0, 2].set_ylabel('Time (s)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(dataset_names)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Her dataset için training curve'ler
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[1, i]
        
        gru_history = results[dataset_name]['GRU']['history']
        lstm_history = results[dataset_name]['LSTM']['history']
        
        ax.plot(gru_history.history['val_accuracy'], 'b-', 
               label='GRU', linewidth=2)
        ax.plot(lstm_history.history['val_accuracy'], 'r-', 
               label='LSTM', linewidth=2)
        
        ax.set_title(f'{dataset_name} - Validation Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Model karmaşıklığı analizi
    complexities = ['Gates', 'Cell States', 'Computations', 'Memory Usage']
    gru_complexity = [2, 1, 3, 4]  # Relative values
    lstm_complexity = [3, 2, 4, 5]  # Relative values
    
    x_comp = np.arange(len(complexities))
    
    bars7 = axes[2, 0].bar(x_comp - width/2, gru_complexity, width, 
                          label='GRU', alpha=0.7, color='blue')
    bars8 = axes[2, 0].bar(x_comp + width/2, lstm_complexity, width,
                          label='LSTM', alpha=0.7, color='red')
    
    axes[2, 0].set_title('Architecture Complexity', fontweight='bold')
    axes[2, 0].set_xlabel('Component')
    axes[2, 0].set_ylabel('Relative Complexity')
    axes[2, 0].set_xticks(x_comp)
    axes[2, 0].set_xticklabels(complexities, rotation=45)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Performans vs Karmaşıklık scatter
    all_gru_acc = [results[name]['GRU']['accuracy'] for name in dataset_names]
    all_lstm_acc = [results[name]['LSTM']['accuracy'] for name in dataset_names]
    all_gru_params = [results[name]['GRU']['parameters'] for name in dataset_names]
    all_lstm_params = [results[name]['LSTM']['parameters'] for name in dataset_names]
    
    axes[2, 1].scatter(all_gru_params, all_gru_acc, s=100, alpha=0.7, 
                      color='blue', label='GRU')
    axes[2, 1].scatter(all_lstm_params, all_lstm_acc, s=100, alpha=0.7, 
                      color='red', label='LSTM')
    
    axes[2, 1].set_title('Accuracy vs Parameters', fontweight='bold')
    axes[2, 1].set_xlabel('Parameter Count')
    axes[2, 1].set_ylabel('Accuracy')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Avantaj/dezavantaj tablosu
    axes[2, 2].axis('off')
    table_text = """
    GRU AVANTAJLARI:
    ✅ Daha basit mimari
    ✅ Daha az parametre
    ✅ Daha hızlı eğitim
    ✅ Daha az memory
    ✅ Overfitting riski düşük
    
    LSTM AVANTAJLARI:
    ✅ Daha güçlü hafıza
    ✅ Kompleks pattern'ler
    ✅ Çok uzun sequence'ler
    ✅ Daha detaylı kontrol
    ✅ Geniş research desteği
    """
    
    axes[2, 2].text(0.05, 0.95, table_text, transform=axes[2, 2].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[2, 2].set_title('Avantaj/Dezavantaj', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return results, training_times

results, training_times = comprehensive_gru_lstm_comparison()

print_section("GRU PRAKTİK UYGULAMA: ZAMAN SERİSİ TAHMİNİ")

def gru_time_series_example():
    """GRU ile kapsamlı zaman serisi tahmini örneği"""
    
    print("📊 Karmaşık zaman serisi verisi oluşturuluyor...")
    
    # Çok bileşenli zaman serisi
    np.random.seed(42)
    n_points = 2000
    time_steps = np.arange(n_points)
    
    # Multiple components
    trend = 0.02 * time_steps + 100
    seasonal_yearly = 20 * np.sin(2 * np.pi * time_steps / 365)
    seasonal_monthly = 8 * np.sin(2 * np.pi * time_steps / 30)
    seasonal_weekly = 4 * np.sin(2 * np.pi * time_steps / 7)
    
    # ARCH-like volatility
    volatility = np.zeros(n_points)
    volatility[0] = 1
    for i in range(1, n_points):
        volatility[i] = 0.05 + 0.9 * volatility[i-1] + 0.05 * np.random.randn()**2
    
    noise = np.random.randn(n_points) * np.sqrt(volatility)
    
    # Combine all components
    ts_data = trend + seasonal_yearly + seasonal_monthly + seasonal_weekly + noise * 5
    
    # Add some structural breaks
    break_points = [500, 1000, 1500]
    for bp in break_points:
        ts_data[bp:] += np.random.normal(0, 10)
    
    print(f"✅ {n_points} noktalık zaman serisi oluşturuldu")
    
    # Veriyi görselleştir
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(time_steps[:500], ts_data[:500], 'b-', linewidth=1, alpha=0.8)
    plt.title('Zaman Serisi - İlk 500 Nokta', fontweight='bold')
    plt.xlabel('Zaman')
    plt.ylabel('Değer')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.hist(ts_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Veri Dağılımı', fontweight='bold')
    plt.xlabel('Değer')
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    
    # Seasonal decomposition (basit)
    plt.subplot(3, 2, 3)
    plt.plot(time_steps[:365], seasonal_yearly[:365], label='Yıllık', linewidth=2)
    plt.plot(time_steps[:365], seasonal_monthly[:365], label='Aylık', linewidth=2)
    plt.plot(time_steps[:365], seasonal_weekly[:365], label='Haftalık', linewidth=2)
    plt.title('Mevsimsel Bileşenler', fontweight='bold')
    plt.xlabel('Zaman')
    plt.ylabel('Değer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ACF approximation
    plt.subplot(3, 2, 4)
    lags = range(1, 51)
    acf_values = []
    for lag in lags:
        if lag < len(ts_data):
            corr = np.corrcoef(ts_data[:-lag], ts_data[lag:])[0, 1]
            acf_values.append(corr)
        else:
            acf_values.append(0)
    
    plt.plot(lags, acf_values, 'o-', linewidth=2, markersize=4)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
    plt.title('Otokorelasyon Fonksiyonu', fontweight='bold')
    plt.xlabel('Lag')
    plt.ylabel('Korelasyon')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    plt.plot(time_steps, volatility, 'r-', linewidth=1, alpha=0.7)
    plt.title('Volatilite Evrimi', fontweight='bold')
    plt.xlabel('Zaman')
    plt.ylabel('Volatilite')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    returns = np.diff(ts_data)
    plt.plot(returns, 'g-', linewidth=0.5, alpha=0.7)
    plt.title('Getiriler (First Differences)', fontweight='bold')
    plt.xlabel('Zaman')
    plt.ylabel('Getiri')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Veri hazırlığı
    print("\n🔧 GRU için veri hazırlığı...")
    
    # Normalizasyon
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ts_data.reshape(-1, 1)).flatten()
    
    # Sequence oluşturma - çok adımlı tahmin
    def create_multi_step_sequences(data, lookback, forecast_horizon):
        X, y = [], []
        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i-lookback:i])
            y.append(data[i:i+forecast_horizon])
        return np.array(X), np.array(y)
    
    LOOKBACK = 60  # 60 adım geriye bak
    FORECAST = 10  # 10 adım ileriye tahmin et
    
    X, y = create_multi_step_sequences(scaled_data, LOOKBACK, FORECAST)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"📊 Sequence'ler oluşturuldu:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Train/validation/test split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"📊 Veri bölümleme:")
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # GRU modeli tasarımı
    print(f"\n🏗️ Gelişmiş GRU modeli oluşturuluyor...")
    
    # Model alternatifleri
    models = {}
    
    # 1. Basic GRU
    basic_gru = Sequential([
        GRU(64, input_shape=(LOOKBACK, 1)),
        Dense(32, activation='relu'),
        Dense(FORECAST)
    ], name='Basic_GRU')
    
    # 2. Stacked GRU
    stacked_gru = Sequential([
        GRU(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dense(32, activation='relu'),
        Dense(FORECAST)
    ], name='Stacked_GRU')
    
    # 3. Bidirectional GRU
    bidirectional_gru = Sequential([
        Bidirectional(GRU(32, return_sequences=True), input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        Bidirectional(GRU(32)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(FORECAST)
    ], name='Bidirectional_GRU')
    
    models['Basic GRU'] = basic_gru
    models['Stacked GRU'] = stacked_gru
    models['Bidirectional GRU'] = bidirectional_gru
    
    # Modelleri compile et
    for name, model in models.items():
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        print(f"✅ {name}: {model.count_params():,} parameters")
    
    # Modelleri eğit
    print(f"\n🚀 Modeller eğitiliyor...")
    
    histories = {}
    training_times = {}
    
    for name, model in models.items():
        print(f"\n📊 {name} eğitiliyor...")
        
        start_time = time.time()
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=8, min_lr=0.0001
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        histories[name] = history
        training_times[name] = training_time
        
        print(f"   ✅ Tamamlandı ({training_time:.1f}s)")
        print(f"   📊 Final val_loss: {history.history['val_loss'][-1]:.6f}")
        print(f"   📊 Final val_mae: {history.history['val_mae'][-1]:.6f}")
    
    # Sonuçları karşılaştır
    print(f"\n📊 MODEL PERFORMANS KARŞILAŞTIRMASI:")
    print("="*60)
    
    test_results = {}
    
    for name, model in models.items():
        # Test predictions
        test_pred = model.predict(X_test, verbose=0)
        
        # Metrics
        mse = mean_squared_error(y_test.flatten(), test_pred.flatten())
        mae = mean_absolute_error(y_test.flatten(), test_pred.flatten())
        rmse = np.sqrt(mse)
        
        test_results[name] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': test_pred,
            'training_time': training_times[name]
        }
        
        print(f"\n{name}:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Training Time: {training_times[name]:.1f}s")
        print(f"   Parameters: {model.count_params():,}")
    
    # En iyi modeli seç
    best_model_name = min(test_results.keys(), 
                         key=lambda x: test_results[x]['mae'])
    
    print(f"\n🏆 EN İYİ MODEL: {best_model_name}")
    print(f"   MAE: {test_results[best_model_name]['mae']:.6f}")
    
    # Görselleştirme
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('GRU Model Karşılaştırması', fontsize=16, fontweight='bold')
    
    # Training histories
    colors = ['blue', 'red', 'green']
    for i, (name, history) in enumerate(histories.items()):
        axes[0, 0].plot(history.history['loss'], color=colors[i], 
                       label=f'{name} Train', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], color=colors[i], 
                       linestyle='--', label=f'{name} Val', linewidth=2)
    
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # MAE comparison
    for i, (name, history) in enumerate(histories.items()):
        axes[0, 1].plot(history.history['mae'], color=colors[i], 
                       label=f'{name} Train', linewidth=2)
        axes[0, 1].plot(history.history['val_mae'], color=colors[i], 
                       linestyle='--', label=f'{name} Val', linewidth=2)
    
    axes[0, 1].set_title('Training MAE', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test performance metrics
    model_names = list(test_results.keys())
    mae_values = [test_results[name]['mae'] for name in model_names]
    rmse_values = [test_results[name]['rmse'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, mae_values, width, 
                          label='MAE', alpha=0.7, color='blue')
    bars2 = axes[1, 0].bar(x + width/2, rmse_values, width,
                          label='RMSE', alpha=0.7, color='red')
    
    axes[1, 0].set_title('Test Performance', fontweight='bold')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training time vs Performance
    times = [test_results[name]['training_time'] for name in model_names]
    
    axes[1, 1].scatter(times, mae_values, s=[models[name].count_params()/1000 
                      for name in model_names], 
                      alpha=0.7, c=colors[:len(model_names)])
    
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name, (times[i], mae_values[i]),
                           xytext=(5, 5), textcoords='offset points')
    
    axes[1, 1].set_title('Training Time vs Performance', fontweight='bold')
    axes[1, 1].set_xlabel('Training Time (s)')
    axes[1, 1].set_ylabel('Test MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Best model predictions
    best_model = models[best_model_name]
    best_pred = test_results[best_model_name]['predictions']
    
    # Son 100 test örneğinden 5 tanesini göster
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices[:2]):
        ax = axes[2, i]
        
        # Actual vs predicted for multi-step
        actual = y_test[idx]
        predicted = best_pred[idx]
        steps = range(1, FORECAST + 1)
        
        ax.plot(steps, actual, 'bo-', label='Actual', linewidth=2, markersize=6)
        ax.plot(steps, predicted, 'ro-', label='Predicted', linewidth=2, markersize=6)
        ax.fill_between(steps, actual - 0.02, actual + 0.02, 
                       alpha=0.2, color='blue', label='Confidence')
        
        ax.set_title(f'Sample {i+1}: {FORECAST}-Step Forecast', fontweight='bold')
        ax.set_xlabel('Forecast Step')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models, test_results, scaler, best_model_name

models, test_results, scaler, best_model_name = gru_time_series_example()

print_section("GRU PRAKTİK İPUÇLARI VE ÖNERİLER")

print("💡 GRU KULLANIM ÖNERİLERİ:")
print("-" * 30)

print("\n✅ GRU NE ZAMAN KULLANILMALI?")
print("• LSTM'e göre daha az parametre gerektiğinde")
print("• Hızlı prototipleme ve iterasyon")
print("• Sınırlı hesaplama kaynakları")
print("• Mobil/edge deployment")
print("• Orta uzunlukta sequence'ler (10-100 adım)")
print("• LSTM ile benzer performans, daha basit model")

print("\n❌ GRU NE ZAMAN KULLANILMAMALI?")
print("• Çok karmaşık long-term dependencies")
print("• Çok uzun sequence'ler (>200 adım)")
print("• Hassas memory kontrolü gerektiğinde")
print("• Research odaklı projeler (LSTM daha yaygın)")

print("\n🔧 GRU OPTİMİZASYON İPUÇLARI:")
print("1. **Dropout kullanın**: Özellikle recurrent_dropout")
print("2. **Gradient clipping**: Exploding gradients için")
print("3. **Learning rate scheduling**: Adaptive öğrenme")
print("4. **Batch normalization**: Stabil eğitim")
print("5. **Bidirectional**: Tam context için")
print("6. **Stacking**: Daha derin representation")

print("\n📊 HİPERPARAMETRE REHBERİ:")

hyperparams_guide = """
🎛️ GRU HİPERPARAMETRE REHBERİ:

📈 UNITS (Hidden Size):
   • Kısa seq.: 16-64
   • Orta seq.: 64-128  
   • Uzun seq.: 128-512
   
⏱️ SEQUENCE LENGTH:
   • Min: 10-20 adım
   • Optimal: 30-100 adım
   • Max: 200+ (dikkatli)
   
📚 BATCH SIZE:
   • Küçük data: 16-32
   • Büyük data: 64-128
   • Memory limit: 256+
   
🧠 LEARNING RATE:
   • Başlangıç: 0.001
   • Fine-tuning: 0.0001
   • Schedule: ReduceLROnPlateau
   
🎯 DROPOUT:
   • Standard: 0.1-0.3
   • Recurrent: 0.1-0.2
   • Dense layers: 0.2-0.5
"""

print(hyperparams_guide)

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu GRU modülünde öğrendikleriniz:")
print("  1. 🌀 GRU mimarisi ve gate mekanizmaları")
print("  2. 📊 LSTM ile detaylı karşılaştırma")
print("  3. ⚖️ Performans vs karmaşıklık trade-off'u")
print("  4. 🏗️ Farklı GRU varyantları (Stacked, Bidirectional)")
print("  5. 📈 Zaman serisi tahmininde pratik uygulama")
print("  6. 🔧 Hiperparametre optimizasyon teknikleri")
print("  7. 💡 Kullanım alanları ve sınırları")

print(f"\n🏆 PERFORMANS ÖZETİ:")
print(f"   En iyi model: {best_model_name}")
print(f"   Test MAE: {test_results[best_model_name]['mae']:.6f}")
print(f"   Training time: {test_results[best_model_name]['training_time']:.1f}s")

print("\n💡 Ana çıkarımlar:")
print("  • GRU genellikle LSTM kadar iyi performans gösterir")
print("  • %25-30 daha az parametre kullanır")
print("  • Daha hızlı eğitir ve deploy eder")
print("  • Çoğu zaman series problemi için yeterli")
print("  • LSTM'e göre daha basit ve anlaşılır")

print("\n🚀 İyileştirme önerileri:")
print("  1. Attention mechanism eklemek")
print("  2. Ensemble modeller kullanmak")
print("  3. External features dahil etmek")
print("  4. Advanced regularization (DropConnect)")
print("  5. Custom loss functions")

print("\n📚 Sonraki modül: 08_sentiment_analysis.py")
print("RNN ile doğal dil işleme öğreneceğiz!")

print("\n" + "=" * 70)
print("✅ GRU ÖRNEĞİ TAMAMLANDI!")
print("=" * 70)