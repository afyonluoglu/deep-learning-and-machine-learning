"""
Hiperparametre Optimizasyonu Rehberi
Bu modül farklı hiperparametre kombinasyonlarını test eder ve sonuçları karşılaştırır
"""

import itertools
import pandas as pd
import numpy as np
from datetime import datetime

class HyperparameterTuner:
    def __init__(self):
        self.results = []
        
    def suggest_parameters_for_beginners(self):
        """Yeni başlayanlar için önerilen parametre aralıkları"""
        suggestions = {
            "hidden_layers": {
                "values": [1, 2, 3, 4],
                "explanation": "Az katman: Basit problemler, Çok katman: Karmaşık problemler"
            },
            "neurons": {
                "values": [8, 16, 32, 64, 128],
                "explanation": "Az nöron: Hızlı ama basit, Çok nöron: Yavaş ama güçlü"
            },
            "activation": {
                "values": ['relu', 'sigmoid', 'tanh'],
                "explanation": "ReLU: Çoğu durumda en iyi, Sigmoid/Tanh: Özel durumlar için"
            },
            "optimizer": {
                "values": ['adam', 'rmsprop', 'sgd'],
                "explanation": "Adam: Genelde en iyi, RMSprop: RNN'ler için, SGD: Klasik"
            },
            "learning_rate": {
                "values": [0.001, 0.01, 0.1],
                "explanation": "Küçük: Yavaş öğrenme, Büyük: Hızlı ama kararsız"
            },
            "batch_size": {
                "values": [16, 32, 64, 128],
                "explanation": "Küçük: Daha sık güncelleme, Büyük: Daha kararlı gradient"
            }
        }
        
        print("🎯 BAŞLANGIÇ İÇİN HİPERPARAMETRE REHBERİ")
        print("="*60)
        
        for param, info in suggestions.items():
            print(f"\n📊 {param.upper()}:")
            print(f"   Önerilen değerler: {info['values']}")
            print(f"   💡 Açıklama: {info['explanation']}")
        
        return suggestions
    
    def run_parameter_comparison(self, param_grid, train_function, X_train, y_train, X_test, y_test):
        """Farklı parametre kombinasyonlarını test eder"""
        print("\n🔬 PARAMETRE KARŞILAŞTIRMA TESTİ BAŞLIYOR...")
        print("="*60)
        
        # Parametre kombinasyonları oluştur
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        total_tests = len(combinations)
        print(f"Toplam {total_tests} farklı kombinasyon test edilecek...")
        
        for i, combination in enumerate(combinations[:5]):  # İlk 5'ini test et
            params = dict(zip(keys, combination))
            print(f"\n🧪 Test {i+1}/{min(5, total_tests)}: {params}")
            
            try:
                # Model oluştur ve eğit
                start_time = datetime.now()
                model, history = train_function(params, X_train, y_train, X_test, y_test)
                duration = (datetime.now() - start_time).seconds
                
                # Sonuçları kaydet
                result = {
                    'test_id': i+1,
                    'params': params.copy(),
                    'duration': duration,
                    'final_accuracy': history.history.get('accuracy', [0])[-1] if history else 0,
                    'final_loss': history.history.get('loss', [0])[-1] if history else 0,
                    'best_val_accuracy': max(history.history.get('val_accuracy', [0])) if history else 0
                }
                self.results.append(result)
                
                print(f"   ✅ Sonuç: Doğruluk={result['final_accuracy']:.4f}, Süre={duration}s")
                
            except Exception as e:
                print(f"   ❌ Hata: {str(e)}")
        
        self.display_comparison_results()
    
    def display_comparison_results(self):
        """Test sonuçlarını karşılaştırmalı olarak gösterir"""
        if not self.results:
            print("❌ Henüz test sonucu bulunmuyor.")
            return
        
        print("\n📊 TEST SONUÇLARI KARŞILAŞTIRMASI")
        print("="*80)
        
        # En iyi sonuçları bul
        best_accuracy = max(self.results, key=lambda x: x['final_accuracy'])
        fastest = min(self.results, key=lambda x: x['duration'])
        
        print(f"🏆 EN İYİ DOĞRULUK: Test {best_accuracy['test_id']}")
        print(f"   Parametreler: {best_accuracy['params']}")
        print(f"   Doğruluk: {best_accuracy['final_accuracy']:.4f}")
        print(f"   Süre: {best_accuracy['duration']}s")
        
        print(f"\n⚡ EN HIZLI: Test {fastest['test_id']}")
        print(f"   Parametreler: {fastest['params']}")
        print(f"   Doğruluk: {fastest['final_accuracy']:.4f}")
        print(f"   Süre: {fastest['duration']}s")
        
        # Detaylı tablo
        print(f"\n📋 DETAYLI SONUÇLAR:")
        print("-"*80)
        print(f"{'Test':<6} {'Doğruluk':<10} {'Kayıp':<10} {'Süre':<8} {'Parametreler'}")
        print("-"*80)
        
        for result in sorted(self.results, key=lambda x: x['final_accuracy'], reverse=True):
            params_str = str(result['params'])[:50] + "..." if len(str(result['params'])) > 50 else str(result['params'])
            print(f"{result['test_id']:<6} {result['final_accuracy']:<10.4f} {result['final_loss']:<10.4f} "
                  f"{result['duration']:<8}s {params_str}")
    
    def explain_parameter_effects(self):
        """Her parametrenin model üzerindeki etkisini açıklar"""
        explanations = {
            "Learning Rate (Öğrenme Oranı)": """
🎯 Ne işe yarar: Modelin ne kadar hızlı öğreneceğini belirler
📈 Çok yüksek (0.1+): Model çok hızlı öğrenmeye çalışır, optimum noktayı kaçırabilir
📉 Çok düşük (0.0001-): Model çok yavaş öğrenir, eğitim çok uzun sürer  
✅ Önerilen: 0.001 - 0.01 arası başlayın
            """,
            "Batch Size (Toplu Boyut)": """
🎯 Ne işe yarar: Her güncellemede kaç örnek kullanılacağını belirler
📈 Büyük batch (128+): Daha kararlı gradientler, daha az günceleme
📉 Küçük batch (16-32): Daha sık günceleme, daha fazla noise
✅ Önerilen: 32-64 arası deneyerek başlayın
            """,
            "Hidden Layers (Gizli Katmanlar)": """  
🎯 Ne işe yarar: Modelin karmaşıklığını ve öğrenme kapasitesini artırır
📈 Çok katman (5+): Karmaşık problemler için güçlü ama overfitting riski
📉 Az katman (1-2): Basit problemler için yeterli, hızlı eğitim
✅ Önerilen: 2-3 katmanla başlayın, sonra artırın
            """
        }
        
        print("📚 HİPERPARAMETRE ETKİLERİ REHBERİ")
        print("="*60)
        
        for param, explanation in explanations.items():
            print(f"\n{param}")
            print(explanation)
            input("Devam etmek için ENTER tuşuna basınız...")

# Test için örnek fonksiyon
def create_simple_test_function():
    """Basit bir test fonksiyonu oluşturur"""
    def dummy_train_function(params, X_train, y_train, X_test, y_test):
        """Test amaçlı sahte eğitim fonksiyonu"""
        import time
        import random
        
        # Sahte eğitim süresi (parametre karmaşıklığına göre)
        complexity = params.get('neurons', 32) * params.get('hidden_layers', 2) 
        time.sleep(min(complexity / 1000, 3))  # Maksimum 3 saniye bekle
        
        # Sahte sonuçlar
        class FakeHistory:
            def __init__(self):
                epochs = 10
                base_acc = random.uniform(0.7, 0.9)
                self.history = {
                    'accuracy': [base_acc + random.uniform(-0.1, 0.1) for _ in range(epochs)],
                    'loss': [random.uniform(0.1, 0.5) for _ in range(epochs)],
                    'val_accuracy': [base_acc + random.uniform(-0.15, 0.05) for _ in range(epochs)]
                }
        
        return None, FakeHistory()
    
    return dummy_train_function

if __name__ == "__main__":
    tuner = HyperparameterTuner()
    
    print("🔧 HİPERPARAMETRE OPTİMİZASYONU REHBERİ")
    choice = input("""
1. Başlangıç parametreleri önerilerini gör
2. Parametre etkilerini öğren  
3. Sahte test çalıştır (demo)
Seçiminiz: """)
    
    if choice == "1":
        tuner.suggest_parameters_for_beginners()
    elif choice == "2":
        tuner.explain_parameter_effects()
    elif choice == "3":
        # Demo test
        param_grid = {
            'neurons': [16, 32],
            'hidden_layers': [2, 3],
            'learning_rate': [0.001, 0.01]
        }
        dummy_data = (None, None, None, None)  # Sahte veri
        test_func = create_simple_test_function()
        tuner.run_parameter_comparison(param_grid, test_func, *dummy_data)