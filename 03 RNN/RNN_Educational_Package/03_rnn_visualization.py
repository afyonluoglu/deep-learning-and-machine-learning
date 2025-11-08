"""
🎨 RNN MİMARİSİ GÖRSELLEŞTİRME
=============================

Bu dosya RNN mimarisini ve çalışma prensibini görsel olarak açıklar.
Farklı görselleştirme teknikleri ile RNN'lerin nasıl çalıştığını gösterir.

Öğreneceğiniz konular:
1. RNN mimarisi görselleştirme
2. Zaman adımları boyunca bilgi akışı
3. Gizli durum evrimi
4. Ağırlık paylaşımı konsepti
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

print("=" * 60)
print("🎨 RNN MİMARİSİ GÖRSELLEŞTİRME")
print("=" * 60)

# Stil ayarları
plt.style.use('default')
sns.set_palette("husl")

def create_rnn_architecture_diagram():
    """RNN mimarisi diyagramı oluşturur"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Zaman adımları
    time_steps = 4
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # RNN hücrelerini çiz
    for t in range(time_steps):
        x_pos = 2 + t * 3
        
        # RNN hücresi
        rnn_box = FancyBboxPatch(
            (x_pos - 0.5, 3), 1, 1,
            boxstyle="round,pad=0.1",
            facecolor=colors[t],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rnn_box)
        ax.text(x_pos, 3.5, f'RNN', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
        
        # Giriş
        input_box = FancyBboxPatch(
            (x_pos - 0.3, 1), 0.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=1
        )
        ax.add_patch(input_box)
        ax.text(x_pos, 1.3, f'x_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Çıktı
        output_box = FancyBboxPatch(
            (x_pos - 0.3, 5.5), 0.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='lightcoral',
            edgecolor='red',
            linewidth=1
        )
        ax.add_patch(output_box)
        ax.text(x_pos, 5.8, f'y_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Gizli durum
        if t > 0:
            hidden_box = FancyBboxPatch(
                (x_pos - 0.8, 3.2), 0.6, 0.6,
                boxstyle="round,pad=0.05",
                facecolor='lightyellow',
                edgecolor='orange',
                linewidth=1
            )
            ax.add_patch(hidden_box)
            ax.text(x_pos - 0.5, 3.5, f'h_{t-1}', ha='center', va='center', fontsize=9)
        
        # Oklar
        # Giriş -> RNN
        ax.arrow(x_pos, 1.6, 0, 1.2, head_width=0.1, head_length=0.1, 
                fc='blue', ec='blue', linewidth=2)
        
        # RNN -> Çıktı
        ax.arrow(x_pos, 4.2, 0, 1.1, head_width=0.1, head_length=0.1, 
                fc='red', ec='red', linewidth=2)
        
        # Gizli durum bağlantısı
        if t < time_steps - 1:
            ax.arrow(x_pos + 0.5, 3.5, 2.0, 0, head_width=0.1, head_length=0.15,
                    fc='orange', ec='orange', linewidth=2, alpha=0.7)
    
    # Başlıklar ve etiketler
    ax.text(7, 6.5, 'Recurrent Neural Network (RNN) Mimarisi', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.text(1, 3.5, 'Gizli Durum\nAkışı', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='orange')
    
    ax.text(7, 0.5, 'Zaman Adımları →', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Zaman etiketleri
    for t in range(time_steps):
        ax.text(2 + t * 3, 0.2, f't={t}', ha='center', va='center', fontsize=10)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_hidden_state_evolution():
    """Gizli durum evrimini görselleştirir"""
    
    print("\n📊 GİZLİ DURUM EVRİMİ SİMÜLASYONU")
    print("-" * 40)
    
    # Parametreler
    sequence_length = 10
    hidden_size = 4
    
    # Rastgele ağırlıklar
    np.random.seed(42)
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    W_xh = np.random.randn(hidden_size, 1) * 0.1
    b = np.zeros((hidden_size, 1))
    
    # Rastgele giriş dizisi
    inputs = np.random.randn(sequence_length, 1)
    
    # Gizli durumları hesapla
    hidden_states = []
    h = np.zeros((hidden_size, 1))
    
    for t in range(sequence_length):
        h = np.tanh(W_hh @ h + W_xh @ inputs[t].reshape(-1, 1) + b)
        hidden_states.append(h.flatten())
    
    hidden_states = np.array(hidden_states)
    
    # Görselleştirme
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Giriş dizisi
    ax1.plot(range(sequence_length), inputs.flatten(), 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Giriş Dizisi', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Zaman Adımı')
    ax1.set_ylabel('Giriş Değeri')
    ax1.grid(True, alpha=0.3)
    
    # Gizli durumların evrimi
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(hidden_size):
        ax2.plot(range(sequence_length), hidden_states[:, i], 
                color=colors[i], marker='o', linewidth=2, markersize=6,
                label=f'Gizli Nöron {i+1}')
    
    ax2.set_title('Gizli Durum Evrimi', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Zaman Adımı')
    ax2.set_ylabel('Gizli Durum Değeri')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gizli durumları heatmap olarak göster
    im = ax3.imshow(hidden_states.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax3.set_title('Gizli Durumlar - Isı Haritası', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Zaman Adımı')
    ax3.set_ylabel('Gizli Nöron')
    ax3.set_yticks(range(hidden_size))
    ax3.set_yticklabels([f'Nöron {i+1}' for i in range(hidden_size)])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Aktivasyon Değeri', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()
    
    return hidden_states

def create_unfolded_rnn_diagram():
    """Açık RNN diyagramı oluşturur"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Katlanmış RNN
    ax1.set_title('Katlanmış RNN (Folded)', fontsize=14, fontweight='bold')
    
    # RNN bloğu
    rnn_box = FancyBboxPatch(
        (1, 1), 2, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='lightblue',
        edgecolor='blue',
        linewidth=2
    )
    ax1.add_patch(rnn_box)
    ax1.text(2, 1.75, 'RNN', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Giriş ve çıktı
    ax1.arrow(2, 0.5, 0, 0.4, head_width=0.1, head_length=0.05, fc='green', ec='green', linewidth=2)
    ax1.text(2, 0.2, 'Input\nSequence', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.arrow(2, 2.6, 0, 0.4, head_width=0.1, head_length=0.05, fc='red', ec='red', linewidth=2)
    ax1.text(2, 3.2, 'Output\nSequence', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Gizli durum döngüsü
    ax1.add_patch(patches.FancyArrowPatch((3, 1.75), (3.5, 1.75),
                                         arrowstyle='->', mutation_scale=20,
                                         color='orange', linewidth=2))
    ax1.add_patch(patches.FancyArrowPatch((3.5, 1.75), (3.5, 0.5),
                                         arrowstyle='->', mutation_scale=20,
                                         color='orange', linewidth=2))
    ax1.add_patch(patches.FancyArrowPatch((3.5, 0.5), (1, 0.5),
                                         arrowstyle='->', mutation_scale=20,
                                         color='orange', linewidth=2))
    ax1.add_patch(patches.FancyArrowPatch((1, 0.5), (1, 1.75),
                                         arrowstyle='->', mutation_scale=20,
                                         color='orange', linewidth=2))
    
    ax1.text(4, 1.2, 'Hidden State\nLoop', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='orange')
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    
    # Açık RNN
    ax2.set_title('Açık RNN (Unfolded)', fontsize=14, fontweight='bold')
    
    time_steps = 5
    for t in range(time_steps):
        x_pos = 1 + t * 2.5
        
        # RNN hücresi
        rnn_box = FancyBboxPatch(
            (x_pos - 0.4, 2), 0.8, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=1.5
        )
        ax2.add_patch(rnn_box)
        ax2.text(x_pos, 2.4, f'RNN', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Giriş
        ax2.arrow(x_pos, 1.2, 0, 0.6, head_width=0.08, head_length=0.05, fc='green', ec='green')
        ax2.text(x_pos, 0.8, f'x_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Çıktı
        ax2.arrow(x_pos, 2.9, 0, 0.6, head_width=0.08, head_length=0.05, fc='red', ec='red')
        ax2.text(x_pos, 3.7, f'y_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Gizli durum bağlantısı
        if t < time_steps - 1:
            ax2.arrow(x_pos + 0.5, 2.4, 1.5, 0, head_width=0.06, head_length=0.1,
                     fc='orange', ec='orange', linewidth=1.5)
            ax2.text(x_pos + 1.25, 2.7, f'h_{t}', ha='center', va='center', 
                     fontsize=9, color='orange', fontweight='bold')
    
    ax2.text(6.5, 1, 'Zaman →', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_xlim(0, 13)
    ax2.set_ylim(0, 4.5)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_weight_sharing():
    """Ağırlık paylaşımını görselleştirir"""
    
    print("\n⚖️ AĞIRLIK PAYLAŞIMI KONSEPTİ")
    print("-" * 40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Sol: Ağırlık paylaşımı ile
    ax1.set_title('RNN - Ağırlık Paylaşımı', fontsize=14, fontweight='bold')
    
    time_steps = 3
    for t in range(time_steps):
        x_pos = 1 + t * 2
        
        # RNN hücresi
        rnn_box = FancyBboxPatch(
            (x_pos - 0.3, 2), 0.6, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=2
        )
        ax1.add_patch(rnn_box)
        ax1.text(x_pos, 2.4, 'RNN', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Ağırlıkları göster (aynı renkler = paylaşılan ağırlıklar)
        ax1.text(x_pos - 0.1, 1.7, 'W', ha='center', va='center', fontsize=8, 
                color='red', fontweight='bold')
        ax1.text(x_pos + 0.1, 1.7, 'U', ha='center', va='center', fontsize=8, 
                color='green', fontweight='bold')
        
        if t < time_steps - 1:
            ax1.arrow(x_pos + 0.35, 2.4, 1.3, 0, head_width=0.1, head_length=0.1,
                     fc='orange', ec='orange', linewidth=2)
    
    # Ağırlık efsanesi
    ax1.text(4, 1, 'W: Giriş ağırlıkları (paylaşılan)', ha='center', va='center', 
             fontsize=10, color='red', fontweight='bold')
    ax1.text(4, 0.7, 'U: Gizli ağırlıklar (paylaşılan)', ha='center', va='center', 
             fontsize=10, color='green', fontweight='bold')
    ax1.text(4, 0.4, 'Gizli durum akışı', ha='center', va='center', 
             fontsize=10, color='orange', fontweight='bold')
    
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 3.5)
    ax1.axis('off')
    
    # Sağ: Ağırlık paylaşımı olmadan
    ax2.set_title('Alternatif - Ağırlık Paylaşımı Yok', fontsize=14, fontweight='bold')
    
    colors = ['red', 'green', 'blue']
    for t in range(time_steps):
        x_pos = 1 + t * 2
        
        # Farklı renkli kutular (farklı ağırlıklar)
        rnn_box = FancyBboxPatch(
            (x_pos - 0.3, 2), 0.6, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=colors[t],
            alpha=0.3,
            edgecolor=colors[t],
            linewidth=2
        )
        ax2.add_patch(rnn_box)
        ax2.text(x_pos, 2.4, f'NN_{t}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Farklı ağırlıkları göster
        ax2.text(x_pos - 0.1, 1.7, f'W_{t}', ha='center', va='center', fontsize=8, 
                color=colors[t], fontweight='bold')
        ax2.text(x_pos + 0.2, 1.7, f'U_{t}', ha='center', va='center', fontsize=8, 
                color=colors[t], fontweight='bold')
    
    ax2.text(4, 1, 'Her zaman adımında farklı ağırlıklar', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    ax2.text(4, 0.7, 'Çok fazla parametre', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    ax2.text(4, 0.4, 'Genelleme zorluğu', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 3.5)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

    print("💡 Ağırlık paylaşımının faydaları:")
    print("  ✅ Daha az parametre")
    print("  ✅ Daha iyi genelleme")
    print("  ✅ Translation invariance")
    print("  ✅ Daha hızlı eğitim")

# Ana fonksiyonları çalıştır
print("🎨 RNN Mimarisi Diyagramı Oluşturuluyor...")
create_rnn_architecture_diagram()

print("📊 Gizli Durum Evrimi Hesaplanıyor...")
hidden_states = visualize_hidden_state_evolution()

print("📖 Açık RNN Diyagramı Oluşturuluyor...")
create_unfolded_rnn_diagram()

print("⚖️ Ağırlık Paylaşımı Gösteriliyor...")
visualize_weight_sharing()

print("\n" + "=" * 60)
print("📋 GÖRSELLEŞTİRME ÖZETİ")
print("=" * 60)

print("✅ Bu dosyada öğrendiğiniz görselleştirmeler:")
print("  1. 🏗️  RNN mimarisi diyagramı")
print("  2. 📊  Gizli durum evrimi")
print("  3. 📖  Katlanmış vs Açık RNN")
print("  4. ⚖️  Ağırlık paylaşımı konsepti")
print("")
print("💡 Bu görselleştirmeler sayesinde:")
print("  • RNN'lerin zaman boyunca nasıl çalıştığını gördünüz")
print("  • Gizli durumların nasıl evrim geçirdiğini anladınız")
print("  • Ağırlık paylaşımının önemini kavradınız")
print("")
print("📚 Sonraki dosya: 04_vanilla_rnn.py")
print("Vanilla RNN'lerin detaylı implementasyonunu göreceğiz!")

print("\n" + "=" * 60)
print("✅ GÖRSELLEŞTİRME TAMAMLANDI!")
print("=" * 60)