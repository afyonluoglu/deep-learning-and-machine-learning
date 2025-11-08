"""
🚀 RNN EĞİTİM PAKETİ - ANA UYGULAMA
==================================

Bu dosya RNN öğrenim paketinin ana menüsüdür.
Tüm tamamlanmış modüllere erişim sağlar.

Özellikler:
1. Modül seçim menüsü
2. İnteraktif öğrenme
3. Tamamlanmış modüller
4. Detaylı açıklamalar
5. Pratik örnekler
"""

import os
import sys
import subprocess
import importlib.util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ana modül listesi - tamamlanmış modüller
AVAILABLE_MODULES = {
    "01": {
        "file": "01_rnn_theory.py",
        "title": "📚 RNN Teorisi ve Temel Kavramlar",
        "description": "RNN'lerin temel prensipleri, matematiksel formüller",
        "difficulty": "Başlangıç",
        "status": "Mevcut"
    },
    "02": {
        "file": "02_rnn_basic_example.py", 
        "title": "🌡️ Basit RNN Örneği (Sıcaklık Tahmini)",
        "description": "Adım adım RNN implementasyonu ve eğitimi",
        "difficulty": "Başlangıç",
        "status": "Mevcut"
    },
    "03": {
        "file": "03_rnn_visualization.py",
        "title": "🎨 RNN Mimarisi Görselleştirme", 
        "description": "RNN yapısının görsel analizi",
        "difficulty": "Başlangıç",
        "status": "Mevcut"
    },
    "04": {
        "file": "04_vanilla_rnn.py",
        "title": "🔄 Vanilla RNN - Temel RNN Yapısı",
        "description": "Manual RNN implementasyonu, vanishing gradient problemi",
        "difficulty": "Orta",
        "status": "✅ Tamamlandı"
    },
    "05": {
        "file": "05_lstm_example.py",
        "title": "🧠 LSTM - Long Short-Term Memory",
        "description": "LSTM mimarisi ve gate mekanizmaları",
        "difficulty": "Orta", 
        "status": "Mevcut"
    },
    "06": {
        "file": "06_gru_example.py",
        "title": "⚡ GRU - Gated Recurrent Unit",
        "description": "GRU vs LSTM karşılaştırması, hiperparametre optimizasyonu",
        "difficulty": "Orta",
        "status": "✅ Tamamlandı"
    },
    "07": {
        "file": "07_text_generation.py",
        "title": "📝 Metin Üretimi",
        "description": "RNN ile karakter/kelime bazlı metin üretimi",
        "difficulty": "Orta",
        "status": "Mevcut"
    },
    "08": {
        "file": "08_sentiment_analysis.py", 
        "title": "😊 Duygu Analizi",
        "description": "Text preprocessing, RNN ile sentiment classification",
        "difficulty": "Orta",
        "status": "✅ Tamamlandı"
    },
    "09": {
        "file": "09_time_series_prediction.py",
        "title": "📈 Zaman Serisi Tahmini",
        "description": "İleri düzey time series forecasting teknikleri",
        "difficulty": "İleri",
        "status": "✅ Tamamlandı" 
    },
    "10": {
        "file": "10_stock_price_prediction.py",
        "title": "💰 Hisse Senedi Fiyat Tahmini",
        "description": "Technical indicators, finansal analiz, trading simülasyonu",
        "difficulty": "İleri",
        "status": "✅ Tamamlandı"
    },
    "11": {
        "file": "11_bidirectional_rnn.py",
        "title": "🔄 Bidirectional RNN",
        "description": "İleri-geri yönlü RNN, context awareness",
        "difficulty": "İleri",
        "status": "✅ Tamamlandı"
    },
    "12": {
        "file": "12_attention_mechanism.py",
        "title": "🔍 Attention Mechanism",
        "description": "Dikkat mekanizması, self-attention, Transformer temelleri",
        "difficulty": "İleri",
        "status": "✅ Tamamlandı"
    },
    "13": {
        "file": "13_encoder_decoder.py", 
        "title": "🔄 Encoder-Decoder Architecture",
        "description": "Seq2seq learning, advanced architectures",
        "difficulty": "İleri",
        "status": "✅ Tamamlandı"
    }
}


class RNNEducationalModule:
    """Ana modül seçim ve yönetim sınıfı"""
    
    def __init__(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        print("🚀 RNN EĞİTİM PAKETİ BAŞLATILIYOR...")
        print("="*60)
        
    def display_welcome(self):
        """Karşılama mesajı ve genel bilgiler"""
        print("\n🎓 RNN EĞİTİM PAKETİNE HOŞGELDİNİZ!")
        print("="*50)
        print("\n📘 Bu paket RNN'leri kapsamlı öğrenmeniz için tasarlanmıştır.")
        print("🎯 Hedefimiz: Temel kavramlardan ileri seviye uygulamalara kadar")
        print("💡 Her modül interaktif örnekler ve detaylı açıklamalar içerir")
        print("\n📊 Paket İçeriği:")
        print("   • Temel RNN teorisi ve matematiksel temeller")
        print("   • LSTM ve GRU mimarileri")
        print("   • Bidirectional RNN'ler")  
        print("   • Attention mekanizmaları")
        print("   • Praktik uygulamalar (sentiment analizi, zaman serisi tahmini)")
        print("   • İleri seviye konular (Encoder-Decoder, finansal analiz)")
        
    def display_modules_menu(self):
        """Modül seçim menüsünü gösterir"""
        print("\n" + "="*80)
        print("📚 MEVCUT MODÜLLER")
        print("="*80)
        
        # Zorluk seviyesine göre gruplayalım
        difficulty_groups = {
            "Başlangıç": [],
            "Orta": [],
            "İleri": []
        }
        
        for module_id, module_info in AVAILABLE_MODULES.items():
            difficulty_groups[module_info["difficulty"]].append((module_id, module_info))
            
        # Her grup için modülleri göster
        for difficulty, modules in difficulty_groups.items():
            if modules:
                print(f"\n🎯 {difficulty.upper()} SEVİYE:")
                print("-" * 40)
                for module_id, module_info in modules:
                    status_icon = "✅" if "✅" in module_info["status"] else "📄"
                    print(f"{status_icon} {module_id}: {module_info['title']}")
                    print(f"     📝 {module_info['description']}")
                    if self.module_exists(module_info['file']):
                        print(f"     ✅ Dosya mevcut")
                    else:
                        print(f"     ⚠️  Dosya bulunamadı")
                    print()
        
        print("="*80)
        
    def module_exists(self, filename):
        """Modül dosyasının var olup olmadığını kontrol eder"""
        file_path = os.path.join(self.current_path, filename)
        return os.path.exists(file_path)
        
    def run_module(self, module_id):
        """Seçilen modülü çalıştırır"""
        if module_id not in AVAILABLE_MODULES:
            print(f"❌ Hata: '{module_id}' modülü bulunamadı!")
            return False
            
        module_info = AVAILABLE_MODULES[module_id]
        filename = module_info['file']
        
        if not self.module_exists(filename):
            print(f"❌ Hata: '{filename}' dosyası bulunamadı!")
            return False
            
        print(f"\n🚀 {module_info['title']} BAŞLATILIYOR...")
        print("="*60)
        print(f"📝 {module_info['description']}")
        print(f"🎯 Zorluk: {module_info['difficulty']}")
        print("="*60)
        
        try:
            # Modülü çalıştır
            file_path = os.path.join(self.current_path, filename)
            
            # Python script'ini çalıştır
            result = subprocess.run([sys.executable, file_path], 
                                  capture_output=False, 
                                  text=True)
            
            if result.returncode == 0:
                print(f"\n✅ {module_info['title']} başarıyla tamamlandı!")
            else:
                print(f"\n❌ Modül çalıştırılırken hata oluştu!")
                
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            return False
            
        return True
        
    def get_user_choice(self):
        """Kullanıcıdan modül seçimi alır"""
        while True:
            print("\n" + "="*50)
            print("🎮 MENÜ SEÇENEKLERİ:")
            print("="*50)
            print("📚 Modül numarası (örn: '01', '02', '13')")
            print("🔍 'l' veya 'liste' - Tüm modülleri tekrar göster")
            print("ℹ️  'i' veya 'bilgi' - Paket hakkında bilgi")
            print("🚪 'q' veya 'çıkış' - Programdan çık")
            print("="*50)
            
            choice = input("\n👤 Seçiminiz: ").strip().lower()
            
            if choice == 'çıkış' or choice == 'q':
                return 'exit'
            elif choice == 'liste' or choice == 'l':
                return 'list'
            elif choice == 'bilgi' or choice == 'i':
                return 'info'
            elif choice in AVAILABLE_MODULES:
                return choice
            else:
                # Sayı formatını kontrol et
                if len(choice) == 1 and choice.isdigit():
                    choice = '0' + choice  # '1' -> '01'
                    
                if choice in AVAILABLE_MODULES:
                    return choice
                else:
                    print(f"\n❌ Geçersiz seçim: '{choice}'")
                    print("💡 Lütfen geçerli bir modül numarası girin (01-13)")
                    
    def show_package_info(self):
        """Paket hakkında detaylı bilgi gösterir"""
        print("\n" + "="*70)
        print("📘 RNN EĞİTİM PAKETİ - DETAYLAR")
        print("="*70)
        
        total_modules = len(AVAILABLE_MODULES)
        completed_modules = sum(1 for m in AVAILABLE_MODULES.values() if "✅" in m["status"])
        available_modules = sum(1 for m in AVAILABLE_MODULES.values() if self.module_exists(m["file"]))
        
        print(f"\n📊 İstatistikler:")
        print(f"   • Toplam modül sayısı: {total_modules}")
        print(f"   • Tamamlanmış modüller: {completed_modules}")
        print(f"   • Mevcut dosyalar: {available_modules}")
        print(f"   • Tamamlanma oranı: {(completed_modules/total_modules)*100:.1f}%")
        
        print(f"\n🛠️ Teknoloji Stack:")
        print(f"   • TensorFlow/Keras: Deep Learning framework")
        print(f"   • NumPy: Numerik hesaplamalar")
        print(f"   • Matplotlib/Seaborn: Görselleştirme")
        print(f"   • Scikit-learn: Machine learning utilities")
        
        print(f"\n🎯 Öğrenim Hedefleri:")
        print(f"   • RNN temellerini kavrama")
        print(f"   • LSTM ve GRU mimarilerini anlama")
        print(f"   • Praktik problemlerde RNN kullanımı")
        print(f"   • Attention ve advanced konular")
        
    def run(self):
        """Ana program döngüsü"""
        self.display_welcome()
        
        while True:
            self.display_modules_menu()
            choice = self.get_user_choice()
            
            if choice == 'exit':
                print("\n👋 RNN Eğitim Paketi kapatılıyor...")
                print("🎓 Öğrenmeye devam edin!")
                break
            elif choice == 'list':
                continue  # Menüyü tekrar göster
            elif choice == 'info':
                self.show_package_info()
            else:
                # Modül çalıştır
                success = self.run_module(choice)
                if success:
                    input("\n⏸️  Devam etmek için Enter'a basın...")


def main():
    """Ana fonksiyon"""
    try:
        print("🔧 Sistem kontrolü yapılıyor...")
        
        # Gerekli kütüphanelerin kontrolü
        required_packages = ['tensorflow', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("❌ Eksik Python paketleri tespit edildi:")
            for package in missing_packages:
                print(f"   • {package}")
            print("\n💡 Bu paketleri yüklemek için:")
            print(f"   pip install {' '.join(missing_packages)}")
            return
            
        print("✅ Tüm gereksinimler karşılanıyor\n")
        
        # Ana modül başlat
        educational_module = RNNEducationalModule()
        educational_module.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Program kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        print("🔧 Lütfen sistem gereksinimlerini kontrol edin.")


if __name__ == "__main__":
    main()
