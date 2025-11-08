"""
Model Performans Takip Sistemi
Eğitim sonuçlarını kaydeder ve geçmiş performansları takip eder
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class PerformanceTracker:
    def __init__(self, log_file="training_log.json"):
        self.log_file = log_file
        self.training_history = []
        self.load_history()
    
    def load_history(self):
        """Geçmiş eğitim kayıtlarını yükle"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.training_history = json.load(f)
                print(f"📊 {len(self.training_history)} geçmiş eğitim kaydı yüklendi.")
        except Exception as e:
            print(f"⚠️  Geçmiş kayıtlar yüklenemedi: {e}")
            self.training_history = []
    
    def log_training_result(self, model_config, results):
        """Eğitim sonucunu kaydet"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_config": model_config,
            "results": results,
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        self.training_history.append(log_entry)
        self.save_history()
        
        print(f"✅ Eğitim sonucu kaydedildi (Session: {log_entry['session_id']})")
    
    def save_history(self):
        """Geçmişi dosyaya kaydet"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Kayıt dosyasına yazma hatası: {e}")
    
    def generate_progress_report(self):
        """İlerleme raporu oluştur"""
        if not self.training_history:
            print("❌ Henüz eğitim kaydı bulunmuyor.")
            return
        
        print("📈 İLERLEME RAPORU")
        print("="*60)
        
        # Son 10 eğitimi analiz et
        recent_trainings = self.training_history[-10:]
        
        # En iyi performansları bul
        best_accuracy = max(recent_trainings, 
                           key=lambda x: x['results'].get('test_accuracy', 0))
        fastest_training = min(recent_trainings, 
                             key=lambda x: x['results'].get('training_time', float('inf')))
        
        print(f"🏆 EN İYİ DOĞRULUK:")
        print(f"   Tarih: {best_accuracy['timestamp'][:19]}")
        print(f"   Doğruluk: {best_accuracy['results'].get('test_accuracy', 0):.4f}")
        print(f"   Ayarlar: {self._format_config(best_accuracy['model_config'])}")
        
        print(f"\n⚡ EN HIZLI EĞİTİM:")
        print(f"   Tarih: {fastest_training['timestamp'][:19]}")
        print(f"   Süre: {fastest_training['results'].get('training_time', 0):.1f} saniye")
        print(f"   Doğruluk: {fastest_training['results'].get('test_accuracy', 0):.4f}")
        
        # Trend analizi
        if len(recent_trainings) >= 3:
            accuracies = [t['results'].get('test_accuracy', 0) for t in recent_trainings]
            recent_trend = accuracies[-3:]  # Son 3 eğitim
            
            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                trend = "📈 Yükseliş trendi"
            elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                trend = "📉 Düşüş trendi"
            else:
                trend = "📊 Karışık trend"
            
            print(f"\n{trend} (Son 3 eğitim)")
            print(f"   Doğruluk değerleri: {[f'{acc:.3f}' for acc in recent_trend]}")
    
    def _format_config(self, config):
        """Konfigürasyonu kısa formatta göster"""
        key_params = ['hidden_layers', 'neuron_count', 'activation', 'optimizer']
        formatted = []
        for key in key_params:
            if key in config:
                formatted.append(f"{key}:{config[key]}")
        return ", ".join(formatted[:3]) + "..."
    
    def create_performance_charts(self):
        """Performans grafikleri oluştur"""
        if len(self.training_history) < 2:
            print("❌ Grafik için en az 2 eğitim kaydı gerekli.")
            return
        
        # Veriyi hazırla
        dates = []
        accuracies = []
        training_times = []
        
        for entry in self.training_history:
            try:
                date = datetime.fromisoformat(entry['timestamp'])
                accuracy = entry['results'].get('test_accuracy', 0)
                time = entry['results'].get('training_time', 0)
                
                dates.append(date)
                accuracies.append(accuracy)
                training_times.append(time)
            except:
                continue
        
        if len(dates) < 2:
            print("❌ Grafik için yeterli geçerli veri yok.")
            return
        
        # Grafikleri oluştur
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Doğruluk trendi
        ax1.plot(dates, accuracies, 'bo-', linewidth=2, markersize=6)
        ax1.set_title('📈 Test Doğruluğu Trendi', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Doğruluk')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # En iyi performansı vurgula
        best_idx = accuracies.index(max(accuracies))
        ax1.plot(dates[best_idx], accuracies[best_idx], 'r*', markersize=15, 
                label=f'En İyi: {accuracies[best_idx]:.3f}')
        ax1.legend()
        
        # Eğitim süresi trendi
        ax2.plot(dates, training_times, 'go-', linewidth=2, markersize=6)
        ax2.set_title('⏱️ Eğitim Süresi Trendi', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Süre (saniye)')
        ax2.set_xlabel('Tarih')
        ax2.grid(True, alpha=0.3)
        
        # En hızlı eğitimi vurgula  
        fastest_idx = training_times.index(min(training_times))
        ax2.plot(dates[fastest_idx], training_times[fastest_idx], 'r*', markersize=15,
                label=f'En Hızlı: {training_times[fastest_idx]:.1f}s')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_to_excel(self, filename="training_results.xlsx"):
        """Sonuçları Excel'e aktar"""
        if not self.training_history:
            print("❌ Excel'e aktarılacak veri yok.")
            return
        
        try:
            # DataFrame oluştur
            rows = []
            for entry in self.training_history:
                row = {
                    'Tarih': entry['timestamp'][:19],
                    'Session_ID': entry['session_id'],
                    'Test_Accuracy': entry['results'].get('test_accuracy', 0),
                    'Training_Time': entry['results'].get('training_time', 0),
                    'Final_Loss': entry['results'].get('final_loss', 0),
                    'Hidden_Layers': entry['model_config'].get('hidden_layers', 0),
                    'Neuron_Count': entry['model_config'].get('neuron_count', 0),
                    'Activation': entry['model_config'].get('activation', ''),
                    'Optimizer': entry['model_config'].get('optimizer', ''),
                    'Learning_Rate': entry['model_config'].get('learning_rate', 0),
                    'Epochs': entry['model_config'].get('epochs', 0)
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_excel(filename, index=False)
            print(f"✅ Sonuçlar {filename} dosyasına aktarıldı.")
            
        except Exception as e:
            print(f"❌ Excel aktarımı başarısız: {e}")
    
    def get_parameter_recommendations(self):
        """Geçmiş performansa göre parametre önerileri"""
        if len(self.training_history) < 3:
            print("❌ Öneri için en az 3 eğitim kaydı gerekli.")
            return
        
        print("💡 GEÇMİŞ PERFORMANSA DAYALI ÖNERİLER")
        print("="*60)
        
        # En iyi sonuçları analiz et
        top_3 = sorted(self.training_history, 
                      key=lambda x: x['results'].get('test_accuracy', 0), 
                      reverse=True)[:3]
        
        # Ortak özellikleri bul
        common_features = {}
        for key in ['hidden_layers', 'neuron_count', 'activation', 'optimizer']:
            values = [t['model_config'].get(key) for t in top_3 if key in t['model_config']]
            if values and len(set(values)) <= 2:  # En fazla 2 farklı değer
                common_features[key] = max(set(values), key=values.count)  # En sık kullanılan
        
        if common_features:
            print("🎯 En iyi performans gösteren ortak özellikler:")
            for feature, value in common_features.items():
                print(f"   {feature}: {value}")
        else:
            print("⚠️  Ortak pattern bulunamadı. Daha fazla deneyim gerekli.")
        
        # Ortalama değerleri ver
        print(f"\n📊 En iyi 3 eğitimin ortalamaları:")
        avg_layers = sum(t['model_config'].get('hidden_layers', 0) for t in top_3) / 3
        avg_neurons = sum(t['model_config'].get('neuron_count', 0) for t in top_3) / 3
        avg_lr = sum(t['model_config'].get('learning_rate', 0) for t in top_3) / 3
        
        print(f"   Ortalama gizli katman: {avg_layers:.1f}")
        print(f"   Ortalama nöron sayısı: {avg_neurons:.1f}")
        print(f"   Ortalama öğrenme oranı: {avg_lr:.4f}")

# Ana fonksiyonlar
def demo_performance_tracking():
    """Demo performans takibi"""
    tracker = PerformanceTracker("demo_training_log.json")
    
    # Sahte eğitim kayıtları ekle
    import random
    from datetime import timedelta
    
    base_time = datetime.now() - timedelta(days=10)
    
    for i in range(8):
        config = {
            'hidden_layers': random.choice([2, 3, 4]),
            'neuron_count': random.choice([16, 32, 64, 128]),
            'activation': random.choice(['relu', 'tanh', 'sigmoid']),
            'optimizer': random.choice(['adam', 'rmsprop', 'sgd']),
            'learning_rate': random.choice([0.001, 0.01, 0.1]),
            'epochs': random.randint(50, 150)
        }
        
        results = {
            'test_accuracy': random.uniform(0.7, 0.95),
            'training_time': random.uniform(20, 120),
            'final_loss': random.uniform(0.1, 0.5)
        }
        
        # Zaman damgası ayarla
        tracker.training_history.append({
            'timestamp': (base_time + timedelta(days=i)).isoformat(),
            'model_config': config,
            'results': results,
            'session_id': f"demo_{i:03d}"
        })
    
    tracker.save_history()
    
    print("🎮 DEMO: Performans Takip Sistemi")
    tracker.generate_progress_report()
    
    choice = input("\nGrafikleri görmek istiyor musunuz? (y/n): ")
    if choice.lower() == 'y':
        tracker.create_performance_charts()
    
    choice = input("Excel raporu oluşturmak istiyor musunuz? (y/n): ")
    if choice.lower() == 'y':
        tracker.export_to_excel("demo_results.xlsx")
    
    tracker.get_parameter_recommendations()

if __name__ == "__main__":
    demo_performance_tracking()