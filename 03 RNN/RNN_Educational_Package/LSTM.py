def praktik_lstm_ornegi():
    """Gerçek LSTM modelini gösterir"""
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.layers import Input
    import numpy as np
    
    print("🚀 PRATİK LSTM ÖRNEĞİ:")
    print("="*40)
    
    # 1. VERİ HAZIRLA
    print("📊 1. Veri Hazırlama...")
    
    # Örnek: Sayı dizisi tahmini
    # [1,2,3,4,5] → 6 tahmin et
    sequences = []
    targets = []
    
    for start in range(100):  # 100 farklı dizi
        seq = list(range(start, start + 5))  # 5 elemanlı dizi
        target = start + 5  # Sonraki sayı
        sequences.append(seq)
        targets.append(target)
    
    # Numpy array'e çevir
    X = np.array(sequences).reshape(-1, 5, 1)  # (sample, timestep, feature)
    y = np.array(targets)
    
    print(f"   Veri boyutu: {X.shape}")
    print(f"   Örnek giriş: {X[0].flatten()}")
    print(f"   Örnek hedef: {y[0]}")
    
    # 2. MODEL OLUŞTUR
    print("\n🏗️ 2. LSTM Model Oluşturma...")
    
    model = Sequential([
        Input(shape=(5, 1)), 
        LSTM(50, return_sequences=True),  # İlk LSTM katmanı
        Dropout(0.2),  # Overfitting önle
        LSTM(50, return_sequences=False),  # İkinci LSTM katmanı
        Dropout(0.2),
        Dense(25, activation='relu'),  # Dense katman
        Dense(1)  # Çıkış katmanı
    ])
    
    # Modeli derle
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("   ✅ Model oluşturuldu!")
    print(f"   Toplam parametre: {model.count_params():,}")
    
    # Model özetini göster
    print("\n📋 Model Özeti:")
    model.summary()
    
    # 3. EĞİTİM
    print("\n🎓 3. Model Eğitimi...")
    
    # Eğitim
    history = model.fit(X, y, epochs=50, batch_size=32, 
                       validation_split=0.2, verbose=0)
    
    print("   ✅ Eğitim tamamlandı!")
    
    # 4. SONUÇLARI GÖRSELLEŞTIR
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Loss grafiği
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Eğitim Süreci')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Tahmin testi
    plt.subplot(1, 3, 2)
    test_sequences = [
        [10, 11, 12, 13, 14],  # Beklenen: 15
        [20, 21, 22, 23, 24],  # Beklenen: 25
        [50, 51, 52, 53, 54],  # Beklenen: 55
    ]
    
    predictions = []
    actuals = []
    
    for seq in test_sequences:
        X_test = np.array(seq).reshape(1, 5, 1)
        pred = model.predict(X_test, verbose=0)[0][0]
        actual = seq[-1] + 1
        predictions.append(pred)
        actuals.append(actual)
        print(f"   Giriş: {seq} → Tahmin: {pred:.1f}, Gerçek: {actual}")
    
    plt.scatter(actuals, predictions, s=100, alpha=0.7)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', alpha=0.5)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.title('Tahmin Performansı')
    plt.grid(True, alpha=0.3)
    
    # Model karmaşıklığı
    plt.subplot(1, 3, 3)
    layers = ['LSTM1', 'LSTM2', 'Dense1', 'Dense2']
    params = [model.layers[i].count_params() for i in [0, 2, 4, 5]]
    
    plt.bar(layers, params, alpha=0.7, color=['blue', 'blue', 'green', 'green'])
    plt.title('Katman Başına Parametre')
    plt.ylabel('Parametre Sayısı')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for i, p in enumerate(params):
        plt.text(i, p + max(params)*0.01, f'{p:,}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 5. PERFORMANS DEĞERLENDİR
    print(f"\n📊 PERFORMANS:")
    final_loss = history.history['val_loss'][-1]
    final_mae = history.history['val_mae'][-1]
    
    print(f"   Final Validation Loss: {final_loss:.4f}")
    print(f"   Final MAE: {final_mae:.4f}")
    
    # Hata analizi
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    avg_error = np.mean(errors)
    print(f"   Test Ortalama Hatası: {avg_error:.2f}")
    
    if avg_error < 1.0:
        print("   ✅ Mükemmel performans!")
    elif avg_error < 2.0:
        print("   ✅ İyi performans!")
    else:
        print("   ⚠️  Daha fazla eğitim gerekebilir")

# Pratik örneği çalıştır
praktik_lstm_ornegi()