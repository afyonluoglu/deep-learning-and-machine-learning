"""
Model Karşılaştırma ve Analiz Aracı
Farklı model mimarilerini karşılaştırır ve görselleştirir
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import json

class ModelComparison:
    def __init__(self):
        self.models_tested = []
        self.comparison_results = pd.DataFrame()
        
    def add_model_result(self, model_name, architecture, history, test_accuracy, test_loss, training_time):
        """Model sonuçlarını kaydet"""
        result = {
            'model_name': model_name,
            'architecture': architecture,
            'final_train_accuracy': history.history['accuracy'][-1] if history and 'accuracy' in history.history else 0,
            'final_val_accuracy': history.history['val_accuracy'][-1] if history and 'val_accuracy' in history.history else 0,
            'test_accuracy': test_accuracy,
            'final_train_loss': history.history['loss'][-1] if history and 'loss' in history.history else 0,
            'final_val_loss': history.history['val_loss'][-1] if history and 'val_loss' in history.history else 0,
            'test_loss': test_loss,
            'training_time': training_time,
            'overfitting_score': self._calculate_overfitting(history),
            'convergence_epoch': self._find_convergence_epoch(history),
            'stability_score': self._calculate_stability(history),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.models_tested.append(result)
        self.comparison_results = pd.DataFrame(self.models_tested)
        
    def _calculate_overfitting(self, history):
        """Overfitting skorunu hesapla"""
        if not history or 'val_accuracy' not in history.history or 'accuracy' not in history.history:
            return 0
            
        train_acc = np.array(history.history['accuracy'])
        val_acc = np.array(history.history['val_accuracy'])
        
        # Son 5 epoch'un ortalama farkı
        if len(train_acc) >= 5:
            train_avg = np.mean(train_acc[-5:])
            val_avg = np.mean(val_acc[-5:])
            return max(0, train_avg - val_avg)  # Pozitif değer overfitting gösterir
        return 0
    
    def _find_convergence_epoch(self, history):
        """Modelin yakınsadığı epoch'u bul"""
        if not history or 'loss' not in history.history:
            return 0
            
        losses = history.history['loss']
        if len(losses) < 3:
            return len(losses)
            
        # Loss değişiminin %1'in altına düştüğü noktayı bul
        for i in range(2, len(losses)):
            recent_change = abs(losses[i] - losses[i-1]) / losses[i-1]
            if recent_change < 0.01:  # %1'den az değişim
                return i + 1
        return len(losses)
    
    def _calculate_stability(self, history):
        """Eğitim kararlılığını hesapla (loss variance)"""
        if not history or 'loss' not in history.history:
            return 0
            
        losses = np.array(history.history['loss'])
        if len(losses) < 2:
            return 0
            
        # Loss değerlerindeki varyans (düşük = daha kararlı)
        return 1 / (1 + np.var(losses))  # 0-1 arası normalize
    
    def create_comparison_dashboard(self):
        """Karşılaştırma dashboard'u oluştur"""
        if self.comparison_results.empty:
            print("❌ Henüz karşılaştırılacak model bulunmuyor.")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MODEL KARŞILAŞTIRMA DASHBOARD', fontsize=16, fontweight='bold')
        
        # 1. Doğruluk Karşılaştırması
        axes[0,0].bar(range(len(self.comparison_results)), self.comparison_results['test_accuracy'])
        axes[0,0].set_title('Test Doğruluğu')
        axes[0,0].set_xticks(range(len(self.comparison_results)))
        axes[0,0].set_xticklabels(self.comparison_results['model_name'], rotation=45)
        
        # 2. Overfitting Analizi  
        axes[0,1].bar(range(len(self.comparison_results)), self.comparison_results['overfitting_score'])
        axes[0,1].set_title('Overfitting Skoru (Düşük = İyi)')
        axes[0,1].set_xticks(range(len(self.comparison_results)))
        axes[0,1].set_xticklabels(self.comparison_results['model_name'], rotation=45)
        
        # 3. Eğitim Süresi
        axes[0,2].bar(range(len(self.comparison_results)), self.comparison_results['training_time'])
        axes[0,2].set_title('Eğitim Süresi (saniye)')
        axes[0,2].set_xticks(range(len(self.comparison_results)))
        axes[0,2].set_xticklabels(self.comparison_results['model_name'], rotation=45)
        
        # 4. Yakınsama Hızı
        axes[1,0].bar(range(len(self.comparison_results)), self.comparison_results['convergence_epoch'])
        axes[1,0].set_title('Yakınsama Epoch\'u (Düşük = Hızlı)')
        axes[1,0].set_xticks(range(len(self.comparison_results)))
        axes[1,0].set_xticklabels(self.comparison_results['model_name'], rotation=45)
        
        # 5. Kararlılık Skoru
        axes[1,1].bar(range(len(self.comparison_results)), self.comparison_results['stability_score'])
        axes[1,1].set_title('Kararlılık Skoru (Yüksek = İyi)')
        axes[1,1].set_xticks(range(len(self.comparison_results)))
        axes[1,1].set_xticklabels(self.comparison_results['model_name'], rotation=45)
        
        # 6. Genel Performans Radar
        categories = ['Test Accuracy', 'Anti-Overfitting', 'Speed', 'Stability']
        
        # Normalize scores for radar chart (0-1)
        normalized_scores = []
        for idx, row in self.comparison_results.iterrows():
            scores = [
                row['test_accuracy'],
                1 - min(1, row['overfitting_score']),  # Ters çevir, düşük overfitting iyi
                1 / (1 + row['training_time']/60),     # Hız skoru, normalize
                row['stability_score']
            ]
            normalized_scores.append(scores)
        
        # Radar chart için polar subplot
        axes[1,2].remove()
        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Döngüyü kapat
        
        for i, scores in enumerate(normalized_scores):
            values = scores + [scores[0]]  # Döngüyü kapat
            ax_radar.plot(angles, values, 'o-', linewidth=2, 
                         label=self.comparison_results.iloc[i]['model_name'])
            ax_radar.fill(angles, values, alpha=0.25)
            
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Genel Performans Analizi')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.show()
    
    def generate_recommendation_report(self):
        """Model seçimi için öneri raporu oluştur"""
        if self.comparison_results.empty:
            print("❌ Henüz analiz edilecek model bulunmuyor.")
            return
            
        print("📊 MODEL DEĞERLENDİRME RAPORU")
        print("="*60)
        
        # En iyi modeller
        best_accuracy = self.comparison_results.loc[self.comparison_results['test_accuracy'].idxmax()]
        fastest = self.comparison_results.loc[self.comparison_results['training_time'].idxmin()]
        most_stable = self.comparison_results.loc[self.comparison_results['stability_score'].idxmax()]
        least_overfitting = self.comparison_results.loc[self.comparison_results['overfitting_score'].idxmin()]
        
        print(f"\n🏆 EN İYİ DOĞRULUK: {best_accuracy['model_name']}")
        print(f"   Test Doğruluğu: {best_accuracy['test_accuracy']:.4f}")
        print(f"   Mimari: {best_accuracy['architecture']}")
        
        print(f"\n⚡ EN HIZLI EĞİTİM: {fastest['model_name']}")
        print(f"   Eğitim Süresi: {fastest['training_time']:.1f} saniye")
        print(f"   Test Doğruluğu: {fastest['test_accuracy']:.4f}")
        
        print(f"\n🎯 EN KARALI: {most_stable['model_name']}")
        print(f"   Kararlılık Skoru: {most_stable['stability_score']:.4f}")
        print(f"   Test Doğruluğu: {most_stable['test_accuracy']:.4f}")
        
        print(f"\n🛡️  EN AZ OVERFITTING: {least_overfitting['model_name']}")
        print(f"   Overfitting Skoru: {least_overfitting['overfitting_score']:.4f}")
        print(f"   Test Doğruluğu: {least_overfitting['test_accuracy']:.4f}")
        
        # Genel öneri
        print(f"\n💡 ÖNERİ:")
        
        # Composite score hesapla
        self.comparison_results['composite_score'] = (
            self.comparison_results['test_accuracy'] * 0.4 +
            (1 - self.comparison_results['overfitting_score'].clip(0, 1)) * 0.3 +
            self.comparison_results['stability_score'] * 0.2 +
            (1 / (1 + self.comparison_results['training_time']/60)) * 0.1
        )
        
        best_overall = self.comparison_results.loc[self.comparison_results['composite_score'].idxmax()]
        print(f"   Genel performans için en iyi: {best_overall['model_name']}")
        print(f"   Composite Score: {best_overall['composite_score']:.4f}")
        
        # Detaylı karşılaştırma tablosu
        print(f"\n📋 DETAYLI KARŞILAŞTIRMA:")
        print("-"*100)
        print(f"{'Model':<20} {'Test Acc':<10} {'Overfitting':<12} {'Süre(s)':<10} {'Kararlılık':<12} {'Composite':<12}")
        print("-"*100)
        
        for _, row in self.comparison_results.sort_values('composite_score', ascending=False).iterrows():
            print(f"{row['model_name']:<20} {row['test_accuracy']:<10.4f} {row['overfitting_score']:<12.4f} "
                  f"{row['training_time']:<10.1f} {row['stability_score']:<12.4f} {row['composite_score']:<12.4f}")
    
    def save_results(self, filename="model_comparison_results.json"):
        """Sonuçları dosyaya kaydet"""
        if self.models_tested:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.models_tested, f, indent=2, ensure_ascii=False)
            print(f"✅ Sonuçlar {filename} dosyasına kaydedildi.")
    
    def load_results(self, filename="model_comparison_results.json"):
        """Sonuçları dosyadan yükle"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.models_tested = json.load(f)
                self.comparison_results = pd.DataFrame(self.models_tested)
            print(f"✅ Sonuçlar {filename} dosyasından yüklendi.")
        except FileNotFoundError:
            print(f"❌ {filename} dosyası bulunamadı.")

# Demo için test fonksiyonu
def demo_comparison():
    """Demo karşılaştırma"""
    import random
    
    comparator = ModelComparison()
    
    # Sahte model sonuçları
    models = [
        ("Basit Model (2 Katman)", "2x32", 0.82, 0.85, 15),
        ("Derin Model (4 Katman)", "4x64", 0.89, 0.92, 45), 
        ("Geniş Model (2 Katman)", "2x128", 0.85, 0.88, 25),
        ("Karmaşık Model (6 Katman)", "6x32", 0.91, 0.78, 75)  # Overfitting örneği
    ]
    
    class FakeHistory:
        def __init__(self, final_acc, final_val_acc):
            epochs = 20
            # Sahte eğitim geçmişi oluştur
            self.history = {
                'accuracy': [0.5 + (final_acc-0.5) * (i/epochs) + random.uniform(-0.05, 0.05) 
                            for i in range(epochs)],
                'val_accuracy': [0.5 + (final_val_acc-0.5) * (i/epochs) + random.uniform(-0.08, 0.08) 
                                for i in range(epochs)],
                'loss': [1.0 * (1 - i/epochs) + random.uniform(-0.1, 0.1) 
                        for i in range(epochs)],
                'val_loss': [1.0 * (1 - i/epochs) + random.uniform(-0.15, 0.15) 
                            for i in range(epochs)]
            }
    
    for name, arch, test_acc, val_acc, time_taken in models:
        fake_history = FakeHistory(test_acc, val_acc)
        comparator.add_model_result(name, arch, fake_history, test_acc, 
                                   random.uniform(0.2, 0.5), time_taken)
    
    print("🎮 DEMO: Model Karşılaştırma Aracı")
    comparator.generate_recommendation_report()
    
    choice = input("\nGörsel dashboard'u görmek istiyor musunuz? (y/n): ")
    if choice.lower() == 'y':
        comparator.create_comparison_dashboard()
    
    choice2 = input("Sonuçları kaydetmek istiyor musunuz? (y/n): ")
    if choice2.lower() == 'y':
        comparator.save_results()

if __name__ == "__main__":
    demo_comparison()