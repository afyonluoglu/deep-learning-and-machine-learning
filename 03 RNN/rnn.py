"""
🧠 RNN (Recurrent Neural Network) - GELİŞTİRİLMİŞ EĞİTİCİ VERSİYON
================================================================

Bu dosya artık geliştirilmiş RNN Eğitim Paketine yönlendiriyor.
Daha kapsamlı, eğitici ve profesyonel bir RNN öğrenme deneyimi için
***RNN_Educational_Package*** klasörüne bakın.

YENİ ÖZELLİKLER:
- İnteraktif adım adım öğretim
- Detaylı görselleştirmeler  
- Farklı RNN türleri (LSTM, GRU)
- Metin üretimi örnekleri
- Duygu analizi uygulaması
- Kapsamlı model değerlendirme
- Profesyonel kod organizasyonu

HIZLI BAŞLANGIÇ:
cd RNN_Educational_Package
python quick_start.py

TAM EĞİTİM:
python main_educational_rnn.py
"""

import os
import sys
import subprocess

def main():
    print("🎓 RNN EĞİTİM PAKETİNE HOŞGELDİNİZ!")
    print("="*60)
    print("Bu program şimdi daha kapsamlı bir eğitim paketi olarak")
    print("yeniden tasarlandı. RNN'leri profesyonel şekilde öğrenmek")
    print("için yeni eğitim paketini kullanın!")
    print("="*60)
    
    print("\n📁 Eğitim Paketi Konumu:")
    package_path = os.path.join(os.path.dirname(__file__), "RNN_Educational_Package")
    print(f"   {package_path}")
    
    if os.path.exists(package_path):
        print("✅ Eğitim paketi bulundu!")
        
        print("\n🚀 BAŞLATMA SEÇENEKLERİ:")
        print("1. Hızlı Başlangıç (Kolay)")
        print("2. Tam Eğitim Tutorial (Detaylı)")
        print("3. Manueel gezinme")
        
        while True:
            try:
                choice = input("\nSeçiminizi yapın (1-3): ")
                
                if choice == "1":
                    quick_start_path = os.path.join(package_path, "quick_start.py")
                    if os.path.exists(quick_start_path):
                        print("\n🚀 Hızlı başlangıç başlatılıyor...")
                        subprocess.run([sys.executable, quick_start_path])
                    else:
                        print("❌ quick_start.py bulunamadı!")
                    break
                    
                elif choice == "2":
                    main_path = os.path.join(package_path, "main_educational_rnn.py")
                    if os.path.exists(main_path):
                        print("\n📚 Tam eğitim tutorial başlatılıyor...")
                        subprocess.run([sys.executable, main_path])
                    else:
                        print("❌ main_educational_rnn.py bulunamadı!")
                    break
                    
                elif choice == "3":
                    print(f"\n📁 Eğitim paketi klasörüne gidin:")
                    print(f"   cd \"{package_path}\"")
                    print(f"\n🎯 Önerilen başlangıç dosyaları:")
                    print("   python 01_rnn_theory.py      # RNN teorisi")
                    print("   python 02_rnn_basic_example.py # Basit örnek")
                    print("   python quick_start.py         # Hızlı menü")
                    break
                    
                else:
                    print("❌ Lütfen 1, 2 veya 3 girin!")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Çıkılıyor...")
                break
    else:
        print("❌ Eğitim paketi bulunamadı!")
        print("\n🔧 Çözüm önerileri:")
        print("1. RNN_Educational_Package klasörünün mevcut olduğundan emin olun")
        print("2. Dosyaları tekrar indirin")
        print("3. Yol kontrolü yapın")

# Eski basit RNN kodu (referans için korundu)
def run_simple_rnn_demo():
    """Basit RNN demo - eski kod"""
    
    print("\n📚 ESKİ BASİT RNN KODU (REFERANS İÇİN):")
    print("-"*40)
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    import numpy as np
    import matplotlib.pyplot as plt
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, Dense
        
        # 1. Örnek sıcaklık verisi üretelim (sinüs + biraz rastgele gürültü)
        np.random.seed(42)
        days = 200
        time = np.arange(days)
        temperature = 10 + 5 * np.sin(time * 0.1) + np.random.normal(0, 0.5, size=days)
        
        # 2. Veriyi RNN için hazırla (geçmiş 5 günü giriş, 6. günü hedef)
        window_size = 5
        X, y = [], []
        for i in range(len(temperature) - window_size):
            X.append(temperature[i:i+window_size])
            y.append(temperature[i+window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        # RNN giriş şekli: (örnek sayısı, zaman adımı, özellik sayısı)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # 3. Veriyi train/test ayır
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        print("✨ Model eğitimi başlıyor...")
        # 4. Modeli kur
        model = Sequential([
            SimpleRNN(10, activation='tanh', input_shape=(window_size, 1)),
            Dense(1)
        ])
        
        print("  ➡️ Model derleniyor...")
        model.compile(optimizer='adam', loss='mse')
        print("🎉 Model hazırlandı:")
        model.summary()
        
        # 5. Modeli eğit
        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
        
        # 6. Tahmin yap
        predictions = model.predict(X_test)
        
        # 7. Sonuçları görselleştir
        plt.figure(figsize=(10,5))
        plt.plot(range(len(y_test)), y_test, label="Gerçek")
        plt.plot(range(len(predictions)), predictions, label="Tahmin")
        plt.legend()
        plt.title("RNN ile sıcaklık tahmini")
        plt.show()
        
        print("✅ Basit RNN demo tamamlandı!")
        
    except ImportError:
        print("❌ TensorFlow yüklü değil!")
        print("🔧 Kurulum: pip install tensorflow")

if __name__ == "__main__":
    main()
    
    # Kullanıcı eski kodu görmek isterse
    show_old = input("\n❓ Eski basit RNN kodunu da görmek ister misiniz? (e/h): ")
    if show_old.lower() in ['e', 'evet', 'yes', 'y']:
        run_simple_rnn_demo()
