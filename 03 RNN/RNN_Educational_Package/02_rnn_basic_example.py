"""
🌡️ BASIT RNN ÖRNEĞİ - SICAKLIK TAHMİNİ
=====================================

Bu dosya TensorFlow/Keras kullanarak basit bir RNN modeli oluşturur.
Sıcaklık verilerini kullanarak zaman serisi tahmini yapar.

Öğreneceğiniz konular:
1. TensorFlow/Keras ile RNN oluşturma
2. Zaman serisi verisi hazırlama
3. Model eğitimi ve değerlendirme
4. Sonuçları görselleştirme
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=" * 60)
print("🌡️ BASIT RNN ÖRNEĞİ - SICAKLIK TAHMİNİ")
print("=" * 60)

# Model özetini güzel göstermek için
def print_section(title):
    print(f"\n{'='*50}")
    print(f"📋 {title}")
    print(f"{'='*50}")

print_section("ADIM 1: VERİ HAZIRLIĞI")

# Sentetik sıcaklık verisi oluştur
print("🔧 Sentetik sıcaklık verisi oluşturuluyor...")
np.random.seed(42)
days = 365  # 1 yıllık veri
time = np.arange(days)

# Gerçekçi sıcaklık verisi:
# - Yıllık trend (sinüs dalgası)
# - Haftalık küçük dalgalanmalar
# - Rastgele gürültü
annual_trend = 15 + 10 * np.sin(time * 2 * np.pi / 365)  # Yıllık mevsimlik değişim
weekly_variation = 3 * np.sin(time * 2 * np.pi / 7)      # Haftalık değişim
noise = np.random.normal(0, 2, size=days)                 # Rastgele gürültü

temperature = annual_trend + weekly_variation + noise

print(f"✅ {days} günlük sıcaklık verisi oluşturuldu")
print(f"📊 Ortalama sıcaklık: {np.mean(temperature):.2f}°C")
print(f"📊 Min: {np.min(temperature):.2f}°C, Max: {np.max(temperature):.2f}°C")

# Veriyi görselleştir
plt.figure(figsize=(15, 6))
plt.plot(time[:100], temperature[:100], 'b-', linewidth=2, alpha=0.8)
plt.title('İlk 100 Gün - Sentetik Sıcaklık Verisi', fontsize=14, fontweight='bold')
plt.xlabel('Gün')
plt.ylabel('Sıcaklık (°C)')
plt.grid(True, alpha=0.3)
plt.show()

print_section("ADIM 2: RNN İÇİN VERİ HAZIRLAMA")

def create_sequences(data, window_size):
    """
    Zaman serisi verisini RNN için hazırlar
    
    Args:
        data: Ham zaman serisi verisi
        window_size: Geçmiş kaç günü kullanacağız
    
    Returns:
        X: Giriş dizileri (geçmiş veriler)
        y: Hedef değerler (tahmin edilecek gün)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Parametre ayarları
WINDOW_SIZE = 7  # Geçmiş 7 günü kullan
print(f"🎛️ Pencere boyutu: {WINDOW_SIZE} gün")

# Dizileri oluştur
X, y = create_sequences(temperature, WINDOW_SIZE)

# RNN için şekil düzenleme: (örnekler, zaman_adımları, özellikler)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"📏 X şekli: {X.shape} (örnekler, zaman_adımları, özellikler)")
print(f"📏 y şekli: {y.shape}")

# İlk birkaç örneği göster
print("\n📋 İLK 3 ÖRNEK:")
for i in range(3):
    print(f"Örnek {i+1}:")
    print(f"  Giriş (son 7 gün): {X[i].flatten()}")
    print(f"  Hedef (8. gün):    {y[i]:.2f}")

print_section("ADIM 3: TRAIN/VALIDASYON/TEST AYIRMA")

# Veriyi böl
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"📊 Eğitim seti:    {X_train.shape[0]} örnek ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"📊 Validasyon seti: {X_val.shape[0]} örnek ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"📊 Test seti:      {X_test.shape[0]} örnek ({X_test.shape[0]/len(X)*100:.1f}%)")

print_section("ADIM 4: RNN MODELİ OLUŞTURMA")

print("🏗️ RNN modeli oluşturuluyor...")

# Model mimarisi
model = Sequential([
    SimpleRNN(
        units=32,                    # 32 gizli nöron
        activation='tanh',           # Tanh aktivasyon
        input_shape=(WINDOW_SIZE, 1), # Giriş şekli
        dropout=0.1,                 # Dropout (overfitting önleme)
        recurrent_dropout=0.1,       # Recurrent dropout
        return_sequences=False       # Sadece son çıktıyı döndür
    ),
    Dense(16, activation='relu'),    # Tam bağlantılı katman
    Dense(1)                         # Çıktı katmanı (tek değer)
])

# Model derle
model.compile(
    optimizer='adam',
    loss='mse',                      # Mean Squared Error
    metrics=['mae']                  # Mean Absolute Error
)

print("✅ Model hazırlandı!")
print("\n📋 MODEL ÖZETİ:")
model.summary()

print_section("ADIM 5: MODEL EĞİTİMİ")

print("🚀 Model eğitimi başlıyor...")

# Callback'ler (eğitimi iyileştirmek için)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001
)

# Model eğitimi
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("✅ Eğitim tamamlandı!")

print_section("ADIM 6: MODEL DEĞERLENDİRME")

# Tahminleri yap
print("🔮 Tahminler hesaplanıyor...")
train_pred = model.predict(X_train, verbose=0)
val_pred = model.predict(X_val, verbose=0)
test_pred = model.predict(X_test, verbose=0)

# Metrikleri hesapla
def calculate_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"📊 {set_name} Metrikleri:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    return mse, mae, rmse

calculate_metrics(y_train, train_pred.flatten(), "Eğitim")
calculate_metrics(y_val, val_pred.flatten(), "Validasyon")
calculate_metrics(y_test, test_pred.flatten(), "Test")

print_section("ADIM 7: SONUÇLARI GÖRSELLEŞTİRME")

# Eğitim geçmişini görselleştir
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss grafiği
axes[0, 0].plot(history.history['loss'], 'b-', label='Eğitim Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], 'r-', label='Validasyon Loss', linewidth=2)
axes[0, 0].set_title('Model Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE grafiği
axes[0, 1].plot(history.history['mae'], 'b-', label='Eğitim MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], 'r-', label='Validasyon MAE', linewidth=2)
axes[0, 1].set_title('Mean Absolute Error', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Test seti tahminleri
axes[1, 0].scatter(y_test, test_pred, alpha=0.6, color='blue')
min_val, max_val = min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[1, 0].set_title('Gerçek vs Tahmin (Test Seti)', fontweight='bold')
axes[1, 0].set_xlabel('Gerçek Değerler')
axes[1, 0].set_ylabel('Tahmin Edilen Değerler')
axes[1, 0].grid(True, alpha=0.3)

# Zaman serisi tahminleri
test_time = range(len(y_test))
axes[1, 1].plot(test_time, y_test, 'b-', label='Gerçek', linewidth=2, alpha=0.8)
axes[1, 1].plot(test_time, test_pred.flatten(), 'r-', label='Tahmin', linewidth=2, alpha=0.8)
axes[1, 1].set_title('Zaman Serisi Tahminleri', fontweight='bold')
axes[1, 1].set_xlabel('Test Günleri')
axes[1, 1].set_ylabel('Sıcaklık (°C)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_section("ADIM 8: YENİ VERİ İLE TAHMİN")

print("🔮 Yeni verilerle tahmin örneği...")

# Son 7 günlük veriyi al
last_week = temperature[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
next_day_prediction = model.predict(last_week, verbose=0)[0, 0]

print(f"📅 Son 7 gün sıcaklıkları: {temperature[-WINDOW_SIZE:]}")
print(f"🌡️ Yarınki tahmin edilen sıcaklık: {next_day_prediction:.2f}°C")

print_section("ÖZETİ ve SONUÇLAR")

print("✅ Bu örnekte öğrendiklerimiz:")
print("  1. Zaman serisi verisini RNN için hazırlama")
print("  2. SimpleRNN katmanını kullanma")
print("  3. Model eğitimi ve değerlendirme")
print("  4. Sonuçları görselleştirme")
print("")
print("💡 İyileştirme önerileri:")
print("  1. Daha fazla veri kullanın")
print("  2. Hiperparametrelerle oynayın")
print("  3. LSTM veya GRU deneyin")
print("  4. Feature engineering yapın")
print("")
print("📚 Sonraki dosya: 03_rnn_visualization.py")
print("RNN mimarisini görselleştirmeyi öğreneceksiniz!")

print("\n" + "=" * 60)
print("✅ BASIT RNN ÖRNEĞİ TAMAMLANDI!")
print("=" * 60)