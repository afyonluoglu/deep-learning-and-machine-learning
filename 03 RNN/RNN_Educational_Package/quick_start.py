"""
🚀 HIZLI BAŞLANGIÇ REHBERİ
========================

Bu dosya RNN Eğitim Paketini hızla başlatmanız için hazırlanmıştır.
"""

import subprocess
import sys
import os

def check_and_install_requirements():
    """Gerekli paketleri kontrol eder ve kurar"""
    
    print("🔧 Gerekli paketler kontrol ediliyor...")
    
    required_packages = [
        'numpy',
        'matplotlib', 
        'tensorflow',
        'scikit-learn',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} kurulu")
        except ImportError:
            print(f"❌ {package} eksik")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Eksik paketler kuruluyor: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Tüm paketler kuruldu!")
        except subprocess.CalledProcessError:
            print("❌ Paket kurulumunda hata!")
            print("Manuel kurulum: pip install " + " ".join(missing_packages))
            return False
    
    return True

def run_quick_demo():
    """Hızlı demo çalıştırır"""
    
    print("\n🎯 HIZLI RNN DEMO")
    print("="*30)
    
    # Basit imports
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, Dense
        
        print("✅ Tüm kütüphaneler başarıyla yüklendi!")
        
        # Basit veri oluştur
        print("\n📊 Örnek veri oluşturuluyor...")
        np.random.seed(42)
        data = 10 + 5 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 1, 100)
        
        # Basit sequence oluştur
        X, y = [], []
        for i in range(10, len(data)):
            X.append(data[i-10:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"✅ {len(X)} örnek hazırlandı!")
        
        # Basit model oluştur
        print("\n🏗️ RNN modeli oluşturuluyor...")
        model = Sequential([
            SimpleRNN(10, input_shape=(10, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        print("✅ Model hazırlandı!")
        print(f"📊 Parametre sayısı: {model.count_params():,}")
        
        # Kısa eğitim
        print("\n🚀 Model eğitiliyor (10 epoch)...")
        history = model.fit(X, y, epochs=10, verbose=0)
        
        # Tahmin
        pred = model.predict(X[-10:], verbose=0)
        
        print("✅ Eğitim tamamlandı!")
        print(f"📉 Final loss: {history.history['loss'][-1]:.6f}")
        
        # Basit görselleştirme
        plt.figure(figsize=(10, 6))
        plt.plot(data, 'b-', label='Orijinal Veri', alpha=0.8)
        plt.plot(range(len(data)-10, len(data)), pred.flatten(), 
                'ro-', label='RNN Tahminleri', markersize=8)
        plt.title('🧠 RNN Demo - Basit Zaman Serisi Tahmini', fontweight='bold')
        plt.xlabel('Zaman')
        plt.ylabel('Değer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("\n🎉 Demo başarıyla tamamlandı!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo sırasında hata: {str(e)}")
        return False

def show_menu():
    """Ana menüyü gösterir"""
    
    print("\n" + "="*50)
    print("🎓 RNN EĞİTİM PAKETİ - ANA MENÜ")
    print("="*50)
    
    print("\n📚 Seçenekler:")
    print("1️⃣ Hızlı Demo (5 dakika)")
    print("2️⃣ Tam Eğitim Tutorial (30-45 dakika)")
    print("3️⃣ Belirli Konu (İsteğe bağlı)")
    print("4️⃣ Kurulum Kontrolü")
    print("5️⃣ Çıkış")
    
    while True:
        try:
            choice = int(input("\n🎯 Seçiminizi yapın (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                return choice
            print("❌ Lütfen 1-5 arası bir sayı girin!")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin!")

def run_specific_topic():
    """Belirli konuları çalıştırır"""
    
    print("\n📋 KONU SEÇİMİ")
    print("-"*20)
    
    topics = {
        1: ("01_rnn_theory.py", "RNN Teorisi"),
        2: ("02_rnn_basic_example.py", "Basit RNN Örneği"), 
        3: ("03_rnn_visualization.py", "RNN Görselleştirme"),
        4: ("05_lstm_example.py", "LSTM Örneği"),
        5: ("07_text_generation.py", "Metin Üretimi")
    }
    
    print("Mevcut konular:")
    for num, (file, desc) in topics.items():
        print(f"{num}. {desc}")
    
    while True:
        try:
            choice = int(input("\nKonu seçin (1-5): "))
            if choice in topics:
                file, desc = topics[choice]
                print(f"\n🚀 {desc} çalıştırılıyor...")
                
                if os.path.exists(file):
                    subprocess.run([sys.executable, file])
                else:
                    print(f"❌ {file} bulunamadı!")
                break
            print("❌ Lütfen 1-5 arası bir sayı girin!")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin!")

def main():
    """Ana program"""
    
    print("🎓 RNN EĞİTİM PAKETİ BAŞLATILIYOR...")
    
    # Banner
    banner = """
    ╔══════════════════════════════════════════╗
    ║        🧠 RNN EĞİTİM PAKETİ 🧠           ║
    ║                                          ║
    ║   Recurrent Neural Networks öğrenin!     ║
    ║   Adım adım, pratikle, görsellerle!      ║
    ╚══════════════════════════════════════════╝
    """
    print(banner)
    
    while True:
        choice = show_menu()
        
        if choice == 1:
            print("\n🚀 Hızlı demo başlatılıyor...")
            if check_and_install_requirements():
                run_quick_demo()
                
        elif choice == 2:
            print("\n📚 Tam tutorial başlatılıyor...")
            if check_and_install_requirements():
                try:
                    subprocess.run([sys.executable, "main_educational_rnn.py"])
                except FileNotFoundError:
                    print("❌ main_educational_rnn.py bulunamadı!")
                    
        elif choice == 3:
            run_specific_topic()
            
        elif choice == 4:
            check_and_install_requirements()
            
        elif choice == 5:
            print("\n👋 Görüşmek üzere!")
            print("🎓 RNN öğrenmeye devam edin!")
            break
        
        input("\n⏭️ Menüye dönmek için Enter tuşuna basın...")

if __name__ == "__main__":
    main()