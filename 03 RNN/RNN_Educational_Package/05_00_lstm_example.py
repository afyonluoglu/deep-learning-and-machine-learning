"""
🔤 LSTM ÖRNEĞİ - UZUN KISA SÜRELİ HAFIZA
======================================

Bu dosya LSTM (Long Short-Term Memory) ağlarını detaylı şekilde açıklar.
LSTM'lerin Vanilla RNN'lere göre avantajlarını praktik örneklerle gösterir.

LSTM'nin Temel Özellikleri:
1. Cell State (Hücre Durumu) - Uzun vadeli hafıza
2. Forget Gate - Hangi bilgilerin unutulacağını karar verir
3. Input Gate - Hangi yeni bilgilerin saklanacağını karar verir
4. Output Gate - Hangi bilgilerin çıktı olacağını karar verir

Kullanım Alanları:
- Uzun metinlerin analizi
- Makine çevirisi
- Konuşma tanıma
- Zaman serisi tahmini (uzun dönemli bağımlılıklar)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

def print_section(title, char="=", single_line:bool=False, width=55):
    title = title
    if not single_line:
        print(f"{char*width}")
    if char == "=":
        title = "📋 "+ title
    print(title)
    print(f"{char*width}")

print_section("🔤 LSTM ÖRNEĞİ - UZUN KISA SÜRELİ HAFIZA", char="#", width=80)

print_section("LSTM TEORİSİ VE VANILLA RNN KARŞILAŞTIRMASI")

print_section("🧠 LSTM vs Vanilla RNN:", char="-", single_line=True, width=35)

print("Vanilla RNN Problemleri:")
print("  ❌ Vanishing Gradient Problem")
print("  ❌ Uzun vadeli bağımlılıkları öğrenemez")
print("  ❌ Gradyanlar kaybolur/patlar")
print("")
print("LSTM Çözümleri:")
print("  ✅ Cell State ile uzun vadeli hafıza")
print("  ✅ Gate mekanizmaları ile kontrollü bilgi akışı")
print("  ✅ Gradyan akışını korur")
print("  ✅ Selektif unutma ve hatırlama")

print_section("LSTM GATE MEKANİZMALARI")

def visualize_lstm_gates():
    """LSTM gate mekanizmalarını görselleştirir"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Gate Mekanizmaları', fontsize=16, fontweight='bold')
    
    # Forget Gate
    x = np.linspace(0, 10, 100)
    forget_gate = 1 / (1 + np.exp(-(x - 5)))  # Sigmoid
    
    axes[0, 0].plot(x, forget_gate, 'r-', linewidth=3, label='Forget Gate')
    axes[0, 0].set_title('Forget Gate (Unutma Kapısı)', fontweight='bold')
    axes[0, 0].set_xlabel('Giriş')
    axes[0, 0].set_ylabel('Çıktı (0-1)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    axes[0, 0].text(5, 0.3, 'Unutma Eşiği', ha='center', fontsize=10)
    axes[0, 0].legend()
    
    # Input Gate
    input_gate = 1 / (1 + np.exp(-(x - 3)))
    candidate = np.tanh(x - 5)
    
    axes[0, 1].plot(x, input_gate, 'b-', linewidth=3, label='Input Gate')
    axes[0, 1].plot(x, candidate, 'g--', linewidth=2, label='Candidate Values')
    axes[0, 1].set_title('Input Gate (Giriş Kapısı)', fontweight='bold')
    axes[0, 1].set_xlabel('Giriş')
    axes[0, 1].set_ylabel('Değer')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Cell State
    time = np.arange(20)
    cell_state = np.cumsum(np.random.randn(20) * 0.1) + np.sin(time * 0.5)
    
    axes[1, 0].plot(time, cell_state, 'purple', linewidth=3, marker='o', markersize=6)
    axes[1, 0].set_title('Cell State (Hücre Durumu)', fontweight='bold')
    axes[1, 0].set_xlabel('Zaman')
    axes[1, 0].set_ylabel('Cell State Değeri')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(time, cell_state, alpha=0.3, color='purple')
    
    # Output Gate
    output_gate = 1 / (1 + np.exp(-(x - 4)))
    tanh_cell = np.tanh(x - 5)
    
    axes[1, 1].plot(x, output_gate, 'orange', linewidth=3, label='Output Gate')
    axes[1, 1].plot(x, tanh_cell, 'brown', linestyle='--', linewidth=2, label='tanh(Cell State)')
    axes[1, 1].set_title('Output Gate (Çıktı Kapısı)', fontweight='bold')
    axes[1, 1].set_xlabel('Giriş')
    axes[1, 1].set_ylabel('Değer')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("🚪 Gate Açıklamaları:")
    print("-" * 20)
    print("1. 🗑️ Forget Gate: Hangi bilgilerin Cell State'den silineceğini karar verir")
    print("2. 📥 Input Gate: Hangi yeni bilgilerin Cell State'e ekleneceğini karar verir")
    print("3. 🧠 Cell State: Uzun vadeli hafızayı saklar")
    print("4. 📤 Output Gate: Cell State'in hangi kısmının çıktı olacağını karar verir")

visualize_lstm_gates()

print_section("PRATIK ÖRNEK: HİSSE SENEDİ FİYAT TAHMİNİ")

def create_complex_stock_data():
    """Karmaşık hisse senedi verisi oluşturur"""
    
    np.random.seed(42)
    days = 1000
    
    # Trend bileşeni
    trend = np.linspace(100, 200, days)
    
    # Mevsimsel bileşen (aylık döngü)
    seasonal = 20 * np.sin(np.arange(days) * 2 * np.pi / 30)
    
    # Uzun vadeli döngü (yıllık)
    long_cycle = 15 * np.sin(np.arange(days) * 2 * np.pi / 365)
    
    # Volatilite (GARCH benzeri)
    volatility = np.zeros(days)
    volatility[0] = 1
    for i in range(1, days):
        volatility[i] = 0.1 + 0.8 * volatility[i-1] + 0.1 * np.random.randn()**2
    
    # Rastgele şoklar
    shocks = np.random.randn(days) * np.sqrt(volatility)
    
    # Final fiyat
    price = trend + seasonal + long_cycle + shocks * 10
    
    # Pozitif değerler için clamp
    price = np.maximum(price, 50)
    
    return price

# Veri oluştur
print("📊 Karmaşık hisse senedi verisi oluşturuluyor...")
stock_price = create_complex_stock_data()

# Veriyi görselleştir
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(stock_price[:500], 'b-', linewidth=1.5, alpha=0.8)
plt.title('Sentetik Hisse Senedi Fiyatı (İlk 500 Gün)', fontsize=14, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Fiyat ($)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(stock_price[500:], 'r-', linewidth=1.5, alpha=0.8)
plt.title('Son 500 Gün', fontsize=14, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Fiyat ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"📊 Veri istatistikleri:")
print(f"   Toplam gün sayısı: {len(stock_price)}")
print(f"   Ortalama fiyat: ${np.mean(stock_price):.2f}")
print(f"   Min fiyat: ${np.min(stock_price):.2f}")
print(f"   Max fiyat: ${np.max(stock_price):.2f}")
print(f"   Standart sapma: ${np.std(stock_price):.2f}")

print_section("VERİ ÖN İŞLEME VE HAZIRLAMA")

# Veriyi normalize et
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_price.reshape(-1, 1))

def create_lstm_sequences(data, lookback_days, prediction_days=1):
    """LSTM için sequence'ler oluşturur"""
    
    X, y = [], []
    for i in range(lookback_days, len(data) - prediction_days + 1):
        X.append(data[i-lookback_days:i, 0])
        y.append(data[i:i+prediction_days, 0])
    
    return np.array(X), np.array(y)

# Parametre ayarları
LOOKBACK_DAYS = 60  # Son 60 günü kullan
PREDICTION_DAYS = 5  # 5 gün ileriye tahmin

print(f"⚙️ Parametre ayarları:")
print(f"   Geriye bakış günleri: {LOOKBACK_DAYS}")
print(f"   Tahmin günleri: {PREDICTION_DAYS}")

# Sequence'ler oluştur
X, y = create_lstm_sequences(scaled_data, LOOKBACK_DAYS, PREDICTION_DAYS)

# Reshape for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"📏 Veri şekilleri:")
print(f"   X: {X.shape} (örnekler, zaman_adımları, özellikler)")
print(f"   y: {y.shape} (örnekler, tahmin_günleri)")

# Train/validation/test split
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"📊 Veri bölümlemesi:")
print(f"   Eğitim: {len(X_train)} örnek ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validasyon: {len(X_val)} örnek ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test)} örnek ({len(X_test)/len(X)*100:.1f}%)")

print_section("LSTM MODELİ TASARIMI")

print("🏗️ Gelişmiş LSTM modeli oluşturuluyor...")

# Model mimarisi
model = Sequential([
    # İlk LSTM katmanı
    Input(shape=(LOOKBACK_DAYS, 1)),
    LSTM(units=100, return_sequences=True),
    Dropout(0.23),
    
    # İkinci LSTM katmanı
    LSTM(units=100, return_sequences=True),
    Dropout(0.23),
    
    # Üçüncü LSTM katmanı
    LSTM(units=50, return_sequences=False),
    Dropout(0.23),
    
    # Dense katmanları
    Dense(50, activation='relu'),
    Dropout(0.15),
    Dense(25, activation='relu'),
    Dense(PREDICTION_DAYS)  # Multi-step prediction
])

print(f"Model: LSTM (units=100) x2, Dropout: 0.23-> LSTM (units=50), Dropout: 0.23 -> Dense (50), Dropout: 0.15 -> Dense (25) -> Dense ({PREDICTION_DAYS})")

# Model derle
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("✅ Model hazırlandı!")
print("\n📋 MODEL ÖZETİ:")
model.summary()

print_section("MODEL EĞİTİMİ")

print("🚀 LSTM modeli eğitiliyor...")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.0001,
    verbose=1
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
keras_file = os.path.join(CURRENT_DIR, 'best_lstm_model.keras')
print(f"🔖 En iyi model '{keras_file}' dosyasına kaydedilecek.")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    keras_file,
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Eğitim
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

print("✅ Eğitim tamamlandı!")

print_section("MODEL DEĞERLENDİRME")

# Tahminleri yap
print("🔮 Tahminler hesaplanıyor...")
train_pred = model.predict(X_train, verbose=0)
val_pred = model.predict(X_val, verbose=0)
test_pred = model.predict(X_test, verbose=0)

# Inverse transform (normalizasyonu geri al)
def inverse_transform_predictions(predictions, scaler):
    """Tahminleri orijinal scale'e dönüştürür"""
    predictions_reshaped = predictions.reshape(-1, 1)
    return scaler.inverse_transform(predictions_reshaped).reshape(predictions.shape)

# Sadece ilk günün tahminini değerlendir
train_pred_inv = inverse_transform_predictions(train_pred[:, 0:1], scaler)
val_pred_inv = inverse_transform_predictions(val_pred[:, 0:1], scaler)
test_pred_inv = inverse_transform_predictions(test_pred[:, 0:1], scaler)

y_train_inv = inverse_transform_predictions(y_train[:, 0:1], scaler)
y_val_inv = inverse_transform_predictions(y_val[:, 0:1], scaler)
y_test_inv = inverse_transform_predictions(y_test[:, 0:1], scaler)

# Metrikleri hesapla
def calculate_detailed_metrics(y_true, y_pred, set_name):
    """Detaylı metrikleri hesaplar"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n📊 {set_name} Metrikleri:")
    print(f"   MSE :  {mse:.6f}")
    print(f"   MAE :  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}

train_metrics = calculate_detailed_metrics(y_train_inv.flatten(), train_pred_inv.flatten(), "Eğitim")
val_metrics = calculate_detailed_metrics(y_val_inv.flatten(), val_pred_inv.flatten(), "Validasyon")
test_metrics = calculate_detailed_metrics(y_test_inv.flatten(), test_pred_inv.flatten(), "Test")

print_section("SONUÇLARI GÖRSELLEŞTİRME")

# Kapsamlı görselleştirmeler
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

# Eğitim geçmişi
axes[0, 0].plot(history.history['loss'], 'b-', label='Eğitim Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], 'r-', label='Validasyon Loss', linewidth=2)
axes[0, 0].set_title('Model Loss Geçmişi', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# MAE geçmişi
axes[0, 1].plot(history.history['mae'], 'b-', label='Eğitim MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], 'r-', label='Validasyon MAE', linewidth=2)
axes[0, 1].set_title('Mean Absolute Error', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Test seti tahminleri (scatter plot)
axes[1, 0].scatter(y_test_inv.flatten(), test_pred_inv.flatten(), alpha=0.6, s=30, color='black')
min_val = min(y_test_inv.min(), test_pred_inv.min())
max_val = max(y_test_inv.max(), test_pred_inv.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[1, 0].set_title('Gerçek vs Tahmin (Test)', fontweight='bold')
axes[1, 0].set_xlabel('Gerçek Değerler ($)')
axes[1, 0].set_ylabel('Tahmin Edilen Değerler ($)')
axes[1, 0].grid(True, alpha=0.3)

# Zaman serisi tahminleri (son 200 test örneği)
last_n = min(200, len(y_test_inv))
test_time = range(last_n)
axes[1, 1].plot(test_time, y_test_inv[-last_n:].flatten(), 'b-', 
               label='Gerçek', linewidth=2, alpha=0.8)
axes[1, 1].plot(test_time, test_pred_inv[-last_n:].flatten(), 'r-', 
               label='LSTM Tahmini', linewidth=2, alpha=0.8)
axes[1, 1].set_title('Son 200 Test Günü - Zaman Serisi', fontweight='bold')
axes[1, 1].set_xlabel('Test Günleri')
axes[1, 1].set_ylabel('Hisse Fiyatı ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Hata dağılımı
errors = y_test_inv.flatten() - test_pred_inv.flatten()
axes[2, 0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[2, 0].set_title('Hata Dağılımı (Test Seti)', fontweight='bold')
axes[2, 0].set_xlabel('Tahmin Hatası ($)')
axes[2, 0].set_ylabel('Frekans')
axes[2, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[2, 0].grid(True, alpha=0.3)

# Metrik karşılaştırması
metrics_names = ['MSE', 'MAE', 'RMSE', 'MAPE (%)']
train_values = [train_metrics['mse'], train_metrics['mae'], 
                train_metrics['rmse'], train_metrics['mape']]
val_values = [val_metrics['mse'], val_metrics['mae'], 
              val_metrics['rmse'], val_metrics['mape']]
test_values = [test_metrics['mse'], test_metrics['mae'], 
               test_metrics['rmse'], test_metrics['mape']]

x = np.arange(len(metrics_names))
width = 0.25

bars1 = axes[2, 1].bar(x - width, train_values, width, label='Eğitim', alpha=0.8)
bars2 = axes[2, 1].bar(x, val_values, width, label='Validasyon', alpha=0.8)
bars3 = axes[2, 1].bar(x + width, test_values, width, label='Test', alpha=0.8)

axes[2, 1].set_title('Metrik Karşılaştırması', fontweight='bold')
axes[2, 1].set_xlabel('Metrikler')
axes[2, 1].set_ylabel('Değer')
axes[2, 1].set_xticks(x)
axes[2, 1].set_xticklabels(metrics_names)
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_yscale('log')

plt.tight_layout()
plt.show()

print_section("MULTI-STEP PREDICTION ANALİZİ")

print("🔮 Çok adımlı tahmin analizi...")

# 5 günlük tahminleri analiz et
test_pred_multi = model.predict(X_test[-50:], verbose=0)  # Son 50 örnek
test_pred_multi_inv = inverse_transform_predictions(test_pred_multi, scaler)
y_test_multi_inv = inverse_transform_predictions(y_test[-50:], scaler)

# Her gün için ayrı metrikler
daily_metrics = []
for day in range(PREDICTION_DAYS):
    day_pred = test_pred_multi_inv[:, day]
    day_true = y_test_multi_inv[:, day]
    
    mae = mean_absolute_error(day_true, day_pred)
    rmse = np.sqrt(mean_squared_error(day_true, day_pred))
    mape = np.mean(np.abs((day_true - day_pred) / day_true)) * 100
    
    daily_metrics.append({'day': day+1, 'mae': mae, 'rmse': rmse, 'mape': mape})
    print(f"📅 Gün {day+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# Multi-step tahmin görselleştirmesi
plt.figure(figsize=(15, 10))

# İlk 10 örnek için 5 günlük tahminleri göster
for i in range(min(5, len(test_pred_multi_inv))):
    plt.subplot(2, 3, i+1)
    days = range(1, PREDICTION_DAYS + 1)
    plt.plot(days, y_test_multi_inv[i], 'bo-', label='Gerçek', linewidth=2, markersize=8)
    plt.plot(days, test_pred_multi_inv[i], 'ro-', label='Tahmin', linewidth=2, markersize=8)
    plt.title(f'Örnek {i+1} - 5 Günlük Tahmin', fontweight='bold')
    plt.xlabel('Gün')
    plt.ylabel('Fiyat ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Günlük hata analizi
plt.subplot(2, 3, 6)
days = [m['day'] for m in daily_metrics]
maes = [m['mae'] for m in daily_metrics]
rmses = [m['rmse'] for m in daily_metrics]

plt.plot(days, maes, 'bo-', label='MAE', linewidth=2, markersize=8)
plt.plot(days, rmses, 'ro-', label='RMSE', linewidth=2, markersize=8)
plt.title('Günlük Hata Analizi', fontweight='bold')
plt.xlabel('Tahmin Günü')
plt.ylabel('Hata')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_section("GERÇEK ZAMAN TAHMİNİ ÖRNEĞİ")

print("🔮 Son verileri kullanarak gelecek 5 gün tahmini...")

# En son 60 günlük veriyi al
last_sequence = scaled_data[-LOOKBACK_DAYS:].reshape(1, LOOKBACK_DAYS, 1)

# Tahmin yap
future_prediction = model.predict(last_sequence, verbose=0)
future_prediction_inv = inverse_transform_predictions(future_prediction, scaler)

# Son gerçek fiyatları göster
last_prices = stock_price[-10:]
print("\n📊 Son 10 günün gerçek fiyatları:")
for i, price in enumerate(last_prices, 1):
    print(f"   Gün -{10-i+1}: ${price:.2f}")

print(f"\n🔮 Gelecek 5 günün tahminleri:")
for i, pred_price in enumerate(future_prediction_inv[0], 1):
    print(f"   Gün +{i}: ${pred_price:.2f}")

# Tahminleri görselleştir
plt.figure(figsize=(12, 6))

# Geçmiş verileri göster
past_days = range(-len(last_prices), 0)
future_days = range(1, PREDICTION_DAYS + 1)

plt.plot(past_days, last_prices, 'bo-', label='Gerçek Fiyatlar', 
         linewidth=3, markersize=10, alpha=0.8)
plt.plot(future_days, future_prediction_inv[0], 'ro-', label='LSTM Tahminleri', 
         linewidth=3, markersize=10, alpha=0.8)

# Bugünü işaretle
plt.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
plt.text(0.1, plt.ylim()[1]*0.9, 'Bugün', fontsize=12, fontweight='bold')

plt.title('Hisse Senedi Fiyat Tahmini - Son 10 Gün + Gelecek 5 Gün', 
          fontsize=14, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Fiyat ($)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu LSTM örneğinde öğrendikleriniz:")
print("  1. 🧠 LSTM'in Vanilla RNN'den farkları")
print("  2. 🚪 Gate mekanizmalarının çalışma prensibi")
print("  3. 📊 Karmaşık zaman serisi verisiyle çalışma")
print("  4. 🏗️ Çok katmanlı LSTM mimarisi tasarımı")
print("  5. 🔮 Multi-step prediction (çok adımlı tahmin)")
print("  6. 📈 Gerçek zamanlı tahmin uygulaması")
print("")
print("💡 LSTM'in avantajları bu örnekte görüldü:")
print("  ✅ Uzun vadeli bağımlılıkları öğrenebilir")
print("  ✅ Gradient vanishing problemi olmaz")
print("  ✅ Kompleks zaman serilerinde başarılı")
print("  ✅ Multi-step prediction yapabilir")
print("")
print("📈 Model performansı:")
print(f"  • Test MAE: {test_metrics['mae']:.2f}$")
print(f"  • Test MAPE: {test_metrics['mape']:.2f}%")
print(f"  • Model, ortalama {test_metrics['mape']:.1f}% hata ile tahmin yapıyor")
print("")
print("🔄 Model iyileştirme önerileri:")
print("  1. Daha fazla feature (teknik göstergeler)")
print("  2. Attention mekanizması ekleme")
print("  3. Ensemble modeller kullanma")
print("  4. Hiperparametre optimizasyonu")
print("")
print("📚 Sonraki dosya: 06_gru_example.py")
print("GRU (Gated Recurrent Unit) ile LSTM'i karşılaştıracağız!")

print_section("✅ LSTM ÖRNEĞİ TAMAMLANDI!", char="-", single_line=True, width=35)
