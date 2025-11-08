"""
🔧 KURULUM VE BAŞLANGIÇ REHBERİ
============================

Bu dosya RNN Eğitim Paketini kurmak ve çalıştırmak için gerekli
tüm adımları içerir.
"""

import os
import sys
import subprocess
import importlib

print("="*60)
print("🔧 RNN EĞİTİM PAKETİ KURULUM REHBERİ")
print("="*60)

def check_python_version():
    """Python versiyonunu kontrol eder"""
    version = sys.version_info
    print(f"🐍 Python Versiyonu: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 veya üzeri gereklidir!")
        return False
    else:
        print("✅ Python versiyonu uygun")
        return True

def check_package(package_name, import_name=None):
    """Paket kurulu mu kontrol eder"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} kurulu")
        return True
    except ImportError:
        print(f"❌ {package_name} kurulu değil")
        return False

def install_requirements():
    """Gerekli paketleri kurar"""
    print("\n📦 Gerekli paketler kuruluyor...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Tüm paketler başarıyla kuruldu!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Paket kurulumunda hata oluştu!")
        return False

def create_test_data():
    """Test verilerini oluşturur"""
    print("\n📊 Test verileri oluşturuluyor...")
    
    # data klasörü oluştur
    os.makedirs("data", exist_ok=True)
    
    # Basit test verisi oluştur
    import numpy as np
    
    # Sıcaklık verisi
    np.random.seed(42)
    days = 365
    time = np.arange(days)
    temperature = 15 + 8 * np.sin(time * 2 * np.pi / 365) + np.random.normal(0, 2, size=days)
    
    with open("data/temperature_data.txt", "w") as f:
        for temp in temperature:
            f.write(f"{temp:.2f}\n")
    
    # Hisse senedi verisi
    price = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    price = np.maximum(price, 50)  # Minimum 50
    
    with open("data/stock_data.txt", "w") as f:
        for p in price:
            f.write(f"{p:.2f}\n")
    
    # Metin verisi
    sample_texts = [
        "Bu bir örnek metin.",
        "RNN öğreniyoruz.",
        "Makine öğrenmesi çok ilginç.",
        "Python harika bir dil.",
        "Yapay zeka geleceği şekillendirecek."
    ]
    
    with open("data/sample_text.txt", "w", encoding="utf-8") as f:
        for text in sample_texts:
            f.write(text + "\n")
    
    print("✅ Test verileri oluşturuldu!")

def run_system_check():
    """Sistem gereksinimlerini kontrol eder"""
    print("\n🔍 SİSTEM KONTROLLARI")
    print("-"*30)
    
    # Python versiyonu
    if not check_python_version():
        return False
    
    print("\n📦 PAKET KONTROLLARI")
    print("-"*20)
    
    # Temel paketler
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"), 
        ("matplotlib", "matplotlib.pyplot"),
        ("seaborn", "seaborn"),
        ("sklearn", "sklearn"),
        ("tensorflow", "tensorflow")
    ]
    
    missing_packages = []
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Eksik paketler: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ Tüm temel paketler kurulu!")
        return True

def show_usage_guide():
    """Kullanım rehberini gösterir"""
    print("\n"+"="*60)
    print("📚 KULLANIM REHBERİ")
    print("="*60)
    
    print("\n🚀 Başlamak için:")
    print("1. 01_rnn_theory.py dosyasını çalıştırın")
    print("2. Sırasıyla diğer dosyaları takip edin")
    print("3. Her dosya kendi konusunu detaylı anlatır")
    
    print("\n📋 Dosya sırası:")
    files = [
        "01_rnn_theory.py - RNN teorisi ve temel kavramlar",
        "02_rnn_basic_example.py - Basit RNN örneği",
        "03_rnn_visualization.py - RNN görselleştirme", 
        "04_vanilla_rnn.py - Vanilla RNN detayları",
        "05_lstm_example.py - LSTM örneği",
        "06_gru_example.py - GRU örneği",
        "07_text_generation.py - Metin üretimi",
        "08_sentiment_analysis.py - Duygu analizi",
        "09_time_series_prediction.py - Zaman serisi tahmini",
        "10_stock_price_prediction.py - Hisse senedi tahmini"
    ]
    
    for i, file_desc in enumerate(files, 1):
        print(f"{i:2d}. {file_desc}")
    
    print("\n💡 İpuçları:")
    print("• Her dosyayı ayrı ayrı çalıştırın")
    print("• Kodu okuyarak anlayın")
    print("• Parametrelerle oynayın")
    print("• Kendi verilerinizi deneyin")
    
    print("\n🆘 Yardım:")
    print("• Hata alırsanız önce requirements.txt kontrol edin")
    print("• Python 3.8+ kullandığınızdan emin olun")
    print("• GPU varsa CUDA kurulumu yapabilirsiniz")

def main():
    """Ana kurulum fonksiyonu"""
    print("🎯 RNN Eğitim Paketi kurulum başlatılıyor...\n")
    
    # Sistem kontrolü
    if not run_system_check():
        print("\n❌ Sistem gereksinimleri karşılanmıyor!")
        print("Lütfen önce gerekli paketleri kurun:")
        print("pip install -r requirements.txt")
        return
    
    # Test verileri oluştur
    create_test_data()
    
    # Kullanım rehberi göster
    show_usage_guide()
    
    print("\n🎉 Kurulum tamamlandı!")
    print("Artık RNN öğrenmeye başlayabilirsiniz!")
    print("\nİlk dosyayı çalıştırmak için:")
    print("python 01_rnn_theory.py")

if __name__ == "__main__":
    main()