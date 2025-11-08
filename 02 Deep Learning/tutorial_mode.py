"""
Tutorial Modu - Deep Learning Kavramlarını Adım Adım Öğretici Modül
"""

class DeepLearningTutorial:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 10
        
    def show_concept(self, concept_name, explanation, example_code=None):
        """Kavramları açıklamalı olarak gösterir"""
        print("="*60)
        print(f"📖 KAVRAM: {concept_name}")
        print("="*60)
        print(f"💡 AÇIKLAMA: {explanation}")
        if example_code:
            print("📝 ÖRNEK KOD:")
            print(example_code)
        print("="*60)
        input("Devam etmek için ENTER tuşuna basınız...")
    
    def explain_neural_network_basics(self):
        """Sinir ağı temellerini açıklar"""
        concepts = [
            {
                "name": "Nöron (Neuron)",
                "explanation": """
Nöron, sinir ağının temel yapı taşıdır. Giriş değerlerini alır, 
ağırlıklarla çarpar, bias ekler ve aktivasyon fonksiyonundan geçirir.

Formül: output = activation(sum(input * weight) + bias)
                """,
                "code": """
# Basit bir nöron örneği
import numpy as np

def simple_neuron(inputs, weights, bias, activation='relu'):
    z = np.dot(inputs, weights) + bias
    if activation == 'relu':
        return max(0, z)
    return z

# Örnek kullanım
inputs = [1, 2, 3]
weights = [0.5, -0.2, 0.1]
bias = 0.1
result = simple_neuron(inputs, weights, bias)
print(f"Nöron çıkışı: {result}")
                """
            },
            {
                "name": "Aktivasyon Fonksiyonları",
                "explanation": """
Aktivasyon fonksiyonları, nöronun çıkışını belirler ve ağa doğrusal olmayan 
özellik kazandırır. Yaygın kullanılanlar:
- ReLU: max(0, x) - Basit ve etkili
- Sigmoid: 1/(1+e^-x) - 0-1 arası çıkış
- Tanh: (e^x - e^-x)/(e^x + e^-x) - -1 ile 1 arası
                """,
                "code": """
import numpy as np
import matplotlib.pyplot as plt

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(x, relu(x)); plt.title('ReLU')
plt.subplot(1, 3, 2); plt.plot(x, sigmoid(x)); plt.title('Sigmoid')  
plt.subplot(1, 3, 3); plt.plot(x, tanh(x)); plt.title('Tanh')
plt.show()
                """
            }
        ]
        
        for concept in concepts:
            self.show_concept(concept["name"], concept["explanation"], concept["code"])
    
    def explain_training_process(self):
        """Eğitim sürecini açıklar"""
        steps = [
            "1. Forward Pass: Veri ağdan ileri doğru geçer",
            "2. Loss Calculation: Tahmin ile gerçek değer arasındaki hata hesaplanır",
            "3. Backward Pass: Gradyanlar geri yayılım ile hesaplanır",
            "4. Weight Update: Ağırlıklar optimizer ile güncellenir",
            "5. Epoch tamamlanır ve süreç tekrar eder"
        ]
        
        print("🔄 EĞİTİM SÜRECİ:")
        for step in steps:
            print(f"   {step}")
            input("   Devam etmek için ENTER...")

# Tutorial'ı başlat
def start_tutorial():
    tutorial = DeepLearningTutorial()
    print("🎓 Deep Learning Tutorial Modu Başlatılıyor...")
    
    choice = input("""
Hangi konuyu öğrenmek istiyorsunuz?
1. Sinir Ağı Temelleri
2. Eğitim Süreci
3. Hepsini Göster
Seçiminiz: """)
    
    if choice == "1":
        tutorial.explain_neural_network_basics()
    elif choice == "2":
        tutorial.explain_training_process()
    elif choice == "3":
        tutorial.explain_neural_network_basics()
        tutorial.explain_training_process()

if __name__ == "__main__":
    start_tutorial()