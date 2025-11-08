"""
Görselleştirme Modülü
Self-Attention mekanizmasının görselleştirilmesi
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional
import os
from datetime import datetime


class VisualizationPanel(ctk.CTkFrame):
    """Görselleştirme paneli"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Matplotlib stili
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Notebook (tab) yapısı
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tablar
        self.tab_attention = self.tabview.add("🔍 Attention Map")
        self.tab_qkv = self.tabview.add("📊 Q, K, V Matrisleri")
        self.tab_scores = self.tabview.add("🎯 Attention Skorları")
        self.tab_training = self.tabview.add("📈 Eğitim Grafiği")
        self.tab_explanation = self.tabview.add("💡 Açıklama")
        
        # Canvas'ları başlat
        self.setup_tabs()
        
        # Attention skorlarını sakla
        self.attention_scores = None
        
    def setup_tabs(self):
        """Tabları hazırla"""
        
        # Attention Map tab
        self.attention_canvas_frame = ctk.CTkFrame(self.tab_attention)
        self.attention_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # QKV tab
        self.qkv_canvas_frame = ctk.CTkFrame(self.tab_qkv)
        self.qkv_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Attention Scores tab
        self.scores_frame = ctk.CTkFrame(self.tab_scores)
        self.scores_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Training tab
        self.training_canvas_frame = ctk.CTkFrame(self.tab_training)
        self.training_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Explanation tab
        self.setup_explanation_tab()
        
    def setup_explanation_tab(self):
        """Açıklama tabını hazırla"""
        
        explanation_text = """
        🎯 SELF-ATTENTION MEKANİZMASI
        
        Self-Attention, bir dizideki her elemanın diğer tüm elemanlarla 
        ilişkisini öğrenen güçlü bir mekanizmadır.
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        📖 TEMEL KAVRAMLAR:
        
        • Query (Q): "Neyi arıyorum?" sorusunun cevabı
          Mevcut token'ın diğer token'lardan ne istediği
        
        • Key (K): "Ben neyim?" sorusunun cevabı  
          Her token'ın kendini tanımladığı vektör
        
        • Value (V): "Ne bilgi taşıyorum?" sorusunun cevabı
          Token'ın taşıdığı gerçek bilgi
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        🔢 ATTENTION HESAPLAMA:
        
        1. Her token için Q, K, V vektörleri hesaplanır:
           Q = X × W_q
           K = X × W_k  
           V = X × W_v
        
        2. Attention skorları hesaplanır:
           Scores = (Q × K^T) / sqrt(d_k)
        
        3. Softmax uygulanır:
           Attention_Weights = softmax(Scores)
        
        4. Value'lar ağırlıklandırılır:
           Output = Attention_Weights × V
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        🎨 MULTI-HEAD ATTENTION:
        
        Tek bir attention yerine birden fazla "head" kullanarak
        farklı ilişki türlerini öğrenebiliriz:
        
        • Her head farklı bir bakış açısı sunar
        • Paralel olarak çalışırlar
        • Sonuçlar birleştirilerek zengin bir temsil elde edilir
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        💡 PARAMETRELERIN ETKİLERİ:
        
        d_model (Embedding Boyutu):
        • Daha büyük = Daha zengin temsil
        • Daha küçük = Daha hızlı hesaplama
        • Tipik değerler: 64, 128, 256, 512
        
        num_heads (Head Sayısı):
        • Daha fazla = Daha çeşitli ilişkiler
        • d_model'e tam bölünmeli
        • Tipik değerler: 4, 8, 12, 16
        
        dropout:
        • Overfitting'i önler
        • 0.0 = dropout yok
        • Tipik değerler: 0.1, 0.2, 0.3
        
        learning_rate:
        • Öğrenme hızı
        • Çok büyük = Kararsız eğitim
        • Çok küçük = Yavaş öğrenme
        • Tipik değerler: 0.0001 - 0.01
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        🎯 KULLANIM ÖRNEKLERİ:
        
        1. NLP (Doğal Dil İşleme):
           • Çeviri sistemleri
           • Metin özetleme
           • Duygu analizi
        
        2. Bilgisayar Görüsü:
           • Görüntü sınıflandırma
           • Nesne tespiti
           • Görüntü segmentasyonu
        
        3. Zaman Serisi:
           • Hisse senedi tahmini
           • Hava durumu tahmini
           • Anomali tespiti
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        📊 GÖRSELLEŞTİRMELERİ OKUMA:
        
        Attention Map:
        • Satırlar: Query token'ları
        • Sütunlar: Key token'ları  
        • Renk yoğunluğu: İlişki gücü
        • Koyu renkler: Güçlü ilişki
        
        Q, K, V Matrisleri:
        • Her satır bir token'ı temsil eder
        • Renkler değerlerin büyüklüğünü gösterir
        • Pozitif değerler: Sıcak renkler (kırmızı)
        • Negatif değerler: Soğuk renkler (mavi)
        
        Eğitim Grafiği:
        • X ekseni: Epoch sayısı
        • Y ekseni: Loss değeri
        • İdeal: Azalan trend
        • Platolar: Öğrenme durması
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        # Textbox doğrudan tab'e ekleniyor - kendi scroll bar'ı var
        text_widget = ctk.CTkTextbox(
            self.tab_explanation, 
            wrap="word", 
            font=("Courier New", 13),
            activate_scrollbars=True
        )
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", explanation_text)
        text_widget.configure(state="disabled")  # Salt okunur yap
        
        
    def visualize_all(self, tokens: List[str], attention_weights: Optional[np.ndarray],
                     qkv_matrices: Optional[Dict], history: Dict, config: Dict = None):
        """Tüm görselleştirmeleri güncelle"""
        
        # Varsayılan config
        if config is None:
            config = {}
        
        # Attention Map
        if attention_weights is not None:
            self.visualize_attention_map(tokens, attention_weights, config)
            
            # Attention skorlarını hesapla ve görselleştir
            self.calculate_and_visualize_scores(tokens, attention_weights, qkv_matrices, config)
        
        # QKV Matrisleri
        if qkv_matrices is not None:
            self.visualize_qkv_matrices(tokens, qkv_matrices, config)
        
        # Eğitim geçmişi
        if history and len(history.get('loss', [])) > 0:
            self.visualize_training_history(history, config)
    
    def visualize_attention_map(self, tokens: List[str], attention_weights: np.ndarray, config: Dict = None):
        """Attention map'i görselleştir"""
        
        if config is None:
            config = {}
        
        # Önceki canvas'ı temizle
        for widget in self.attention_canvas_frame.winfo_children():
            widget.destroy()
        
        # Yeni figure oluştur - parametre metni için daha fazla alan
        fig = Figure(figsize=(12, 10), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        
        # Heatmap çiz
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto', 
                       interpolation='nearest')
        
        # Eksen etiketleri - FONT BOYUTU ARTTIRILDI
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', color='white', fontsize=14)
        ax.set_yticklabels(tokens, color='white', fontsize=14)
        
        # Grid
        ax.set_xticks(np.arange(len(tokens))-0.5, minor=True)
        ax.set_yticks(np.arange(len(tokens))-0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        
        # Değerleri göster
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors='white')
        
        # Başlık ve etiketler - FONT BOYUTU ARTTIRILDI
        ax.set_title('Attention Haritası\n(Satır: Query | Sütun: Key)', 
                    color='white', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Key Tokens →', color='white', fontsize=14)
        ax.set_ylabel('Query Tokens →', color='white', fontsize=14)
        
        # Parametre bilgilerini ekle
        param_text = self._format_parameters(config)
        fig.text(0.5, 0.02, param_text, ha='center', va='bottom', 
                color='white', fontsize=10, wrap=True)
        
        fig.tight_layout(rect=[0, 0.08, 1, 1])  # Parametre metni için alt boşluk
        
        # Canvas'a ekle
        canvas = FigureCanvasTkAgg(fig, master=self.attention_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Kaydet - Tarih-saat ile
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
        output_path = os.path.join(output_dir, f"{timestamp}_attention_map.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#2b2b2b')
        
    def visualize_qkv_matrices(self, tokens: List[str], qkv_matrices: Dict, config: Dict = None):
        """Q, K, V matrislerini görselleştir"""
        
        if config is None:
            config = {}
        
        # Önceki canvas'ı temizle
        for widget in self.qkv_canvas_frame.winfo_children():
            widget.destroy()
        
        # Yeni figure oluştur (3 subplot) - parametre metni için daha fazla alan
        fig = Figure(figsize=(16, 7), dpi=100, facecolor='#2b2b2b')
        
        matrices = [('Query (Q)', qkv_matrices['Q']), 
                   ('Key (K)', qkv_matrices['K']),
                   ('Value (V)', qkv_matrices['V'])]
        
        for idx, (name, matrix) in enumerate(matrices):
            ax = fig.add_subplot(1, 3, idx + 1)
            
            # İlk 16 boyutu göster (görselleştirme için)
            display_matrix = matrix[:, :16] if matrix.shape[1] > 16 else matrix
            
            # Heatmap
            im = ax.imshow(display_matrix, cmap='RdBu_r', aspect='auto')
            
            # Eksen etiketleri - FONT BOYUTU ARTTIRILDI
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_yticklabels(tokens, color='white', fontsize=14)
            ax.set_xlabel('Boyut', color='white', fontsize=14)
            
            # Başlık - FONT BOYUTU ARTTIRILDI
            ax.set_title(name, color='white', fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(colors='white')
            
            # Tick colors
            ax.tick_params(colors='white')
        
        fig.suptitle('Query, Key, Value Matrisleri', color='white', 
                    fontsize=16, fontweight='bold')
        
        # Parametre bilgilerini ekle
        param_text = self._format_parameters(config)
        fig.text(0.5, 0.02, param_text, ha='center', va='bottom', 
                color='white', fontsize=10, wrap=True)
        
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])  # Parametre metni için alt boşluk
        
        # Canvas'a ekle
        canvas = FigureCanvasTkAgg(fig, master=self.qkv_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Kaydet - Tarih-saat ile
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
        output_path = os.path.join(output_dir, f"{timestamp}_qkv_matrices.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#2b2b2b')
        
    def visualize_training_history(self, history: Dict, config: Dict = None):
        """Eğitim geçmişini görselleştir"""
        
        if config is None:
            config = {}
        
        # Önceki canvas'ı temizle
        for widget in self.training_canvas_frame.winfo_children():
            widget.destroy()
        
        # Yeni figure oluştur - parametre metni için daha fazla alan
        fig = Figure(figsize=(12, 8), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        
        # Loss grafiği
        epochs = history['epoch']
        losses = history['loss']
        
        ax.plot(epochs, losses, 'o-', color="#ad5400", linewidth=2, 
               markersize=5, label='Training Loss')
        
        # Grid
        ax.grid(True, alpha=0.3, color='gray')
        
        # Etiketler
        ax.set_xlabel('Epoch', color='white', fontsize=14)
        ax.set_ylabel('Loss', color='white', fontsize=14)
        ax.set_title('Eğitim Süreci - Loss Değişimi', color='white', 
                    fontsize=16, fontweight='bold')
        
        # Tick colors
        ax.tick_params(colors='white')
        
        # Spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        # Legend
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        
        # Son loss değerini göster
        if losses:
            final_loss = losses[-1]
            ax.text(0.02, 0.98, f'Son Loss: {final_loss:.4f}', 
                   transform=ax.transAxes, color='white',
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.8))
        
        # Parametre bilgilerini ekle
        param_text = self._format_parameters(config)
        fig.text(0.5, 0.02, param_text, ha='center', va='bottom', 
                color='white', fontsize=10, wrap=True)
        
        fig.tight_layout(rect=[0, 0.08, 1, 1])  # Parametre metni için alt boşluk
        
        # Canvas'a ekle
        canvas = FigureCanvasTkAgg(fig, master=self.training_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Kaydet - Tarih-saat ile
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
        output_path = os.path.join(output_dir, f"{timestamp}_training_history.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#2b2b2b')
    
    def calculate_and_visualize_scores(self, tokens: List[str], attention_weights: np.ndarray, 
                                       qkv_matrices: Optional[Dict], config: Dict = None):
        """Attention skorlarını hesapla ve görselleştir"""
        
        if config is None:
            config = {}
        
        # Boyut kontrolü
        if attention_weights.shape[0] != len(tokens) or attention_weights.shape[1] != len(tokens):
            print(f"⚠️ UYARI: Attention weights boyutu ({attention_weights.shape}) token sayısı ({len(tokens)}) ile uyuşmuyor!")
            # Hata gösterme frame'i oluştur
            for widget in self.scores_frame.winfo_children():
                widget.destroy()
            
            error_label = ctk.CTkLabel(
                self.scores_frame,
                text=f"⚠️ Boyut Uyumsuzluğu\n\n"
                     f"Attention weights: {attention_weights.shape}\n"
                     f"Token sayısı: {len(tokens)}\n\n"
                     f"Yeni bir eğitim yapın.",
                font=ctk.CTkFont(size=14),
                text_color="orange"
            )
            error_label.pack(expand=True, pady=50)
            return
        
        # Attention skorlarını hesapla
        scores_data = self._calculate_attention_scores(tokens, attention_weights, qkv_matrices)
        
        # Skorları sakla (kaydetmek için)
        self.attention_scores = scores_data
        
        # Önceki widget'ları temizle
        for widget in self.scores_frame.winfo_children():
            widget.destroy()
        
        # Scrollable frame oluştur
        scrollable = ctk.CTkScrollableFrame(self.scores_frame)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Başlık
        title = ctk.CTkLabel(
            scrollable,
            text="🎯 TOKEN ATTENTION SKORLARI",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=10)
        
        # Açıklama
        description = ctk.CTkLabel(
            scrollable,
            text="Her token için diğer token'larla olan attention ilişkileri ve istatistikler",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        description.pack(pady=5)
        
        # Her token için skor kartı
        for token_data in scores_data['tokens']:
            self._create_score_card(scrollable, token_data, scores_data['avg_attention'])
    
    def _calculate_attention_scores(self, tokens: List[str], attention_weights: np.ndarray,
                                    qkv_matrices: Optional[Dict]) -> Dict:
        """Attention skorlarını detaylı hesapla"""
        
        scores_data = {
            'tokens': [],
            'avg_attention': np.mean(attention_weights),
            'max_attention': np.max(attention_weights),
            'min_attention': np.min(attention_weights)
        }
        
        # Her token için
        for i, token in enumerate(tokens):
            # Bu token'ın diğerlerine verdiği attention (query olarak)
            attention_given = attention_weights[i, :]
            
            # Bu token'ın diğerlerinden aldığı attention (key olarak)
            attention_received = attention_weights[:, i]
            
            # En çok attention verdiği token'lar
            top_given_indices = np.argsort(attention_given)[::-1][:3]
            top_given = [(tokens[idx], attention_given[idx]) for idx in top_given_indices]
            
            # En çok attention aldığı token'lar
            top_received_indices = np.argsort(attention_received)[::-1][:3]
            top_received = [(tokens[idx], attention_received[idx]) for idx in top_received_indices]
            
            # İstatistikler
            token_data = {
                'token': token,
                'index': i,
                'avg_given': np.mean(attention_given),
                'avg_received': np.mean(attention_received),
                'max_given': np.max(attention_given),
                'max_received': np.max(attention_received),
                'self_attention': attention_weights[i, i],
                'top_given': top_given,
                'top_received': top_received,
                'attention_given': attention_given.tolist(),
                'attention_received': attention_received.tolist()
            }
            
            # QKV norm değerleri (varsa)
            if qkv_matrices:
                token_data['q_norm'] = np.linalg.norm(qkv_matrices['Q'][i, :])
                token_data['k_norm'] = np.linalg.norm(qkv_matrices['K'][i, :])
                token_data['v_norm'] = np.linalg.norm(qkv_matrices['V'][i, :])
            
            scores_data['tokens'].append(token_data)
        
        return scores_data
    
    def _create_score_card(self, parent, token_data: Dict, avg_attention: float):
        """Tek bir token için skor kartı oluştur"""
        
        # Kart frame
        card = ctk.CTkFrame(parent, fg_color="#1e1e1e", corner_radius=10)
        card.pack(fill="x", padx=10, pady=10)
        
        # Token başlığı
        header = ctk.CTkFrame(card, fg_color="#2d2d30")
        header.pack(fill="x", padx=5, pady=5)
        
        token_label = ctk.CTkLabel(
            header,
            text=f"🔤 Token: {token_data['token']}",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        token_label.pack(side="left", padx=10, pady=5)
        
        # Self-attention göstergesi
        self_att_color = self._get_attention_color(token_data['self_attention'], avg_attention)
        self_att_label = ctk.CTkLabel(
            header,
            text=f"Self: {token_data['self_attention']:.3f}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self_att_color
        )
        self_att_label.pack(side="right", padx=10, pady=5)
        
        # İçerik frame'i
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="x", padx=10, pady=5)
        
        # Sol kolon: Verilen Attention
        left_col = ctk.CTkFrame(content, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True, padx=5)
        
        left_title = ctk.CTkLabel(
            left_col,
            text="📤 Verilen Attention (Query)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#61afef"
        )
        left_title.pack(anchor="w", pady=(0, 5))
        
        for target_token, score in token_data['top_given']:
            color = self._get_attention_color(score, avg_attention)
            score_label = ctk.CTkLabel(
                left_col,
                text=f"  → {target_token}: {score:.3f}",
                font=ctk.CTkFont(size=12),
                text_color=color,
                anchor="w"
            )
            score_label.pack(anchor="w", pady=2)
        
        avg_label = ctk.CTkLabel(
            left_col,
            text=f"  📊 Ortalama: {token_data['avg_given']:.3f}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        )
        avg_label.pack(anchor="w", pady=(5, 0))
        
        # Sağ kolon: Alınan Attention
        right_col = ctk.CTkFrame(content, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=5)
        
        right_title = ctk.CTkLabel(
            right_col,
            text="📥 Alınan Attention (Key)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#98c379"
        )
        right_title.pack(anchor="w", pady=(0, 5))
        
        for source_token, score in token_data['top_received']:
            color = self._get_attention_color(score, avg_attention)
            score_label = ctk.CTkLabel(
                right_col,
                text=f"  ← {source_token}: {score:.3f}",
                font=ctk.CTkFont(size=12),
                text_color=color,
                anchor="w"
            )
            score_label.pack(anchor="w", pady=2)
        
        avg_label = ctk.CTkLabel(
            right_col,
            text=f"  📊 Ortalama: {token_data['avg_received']:.3f}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        )
        avg_label.pack(anchor="w", pady=(5, 0))
        
        # QKV normları (varsa)
        if 'q_norm' in token_data:
            qkv_frame = ctk.CTkFrame(card, fg_color="transparent")
            qkv_frame.pack(fill="x", padx=10, pady=(5, 10))
            
            qkv_label = ctk.CTkLabel(
                qkv_frame,
                text=f"📐 Q norm: {token_data['q_norm']:.2f} | K norm: {token_data['k_norm']:.2f} | V norm: {token_data['v_norm']:.2f}",
                font=ctk.CTkFont(size=11),
                text_color="#e5c07b"
            )
            qkv_label.pack(pady=2)
    
    def _get_attention_color(self, score: float, avg_score: float) -> str:
        """Attention skoruna göre renk döndür"""
        if score > avg_score * 1.5:
            return "#e06c75"  # Yüksek - Kırmızı
        elif score > avg_score:
            return "#e5c07b"  # Orta-Yüksek - Sarı
        elif score > avg_score * 0.5:
            return "#98c379"  # Orta - Yeşil
        else:
            return "#61afef"  # Düşük - Mavi
    
    def get_attention_scores(self) -> Optional[Dict]:
        """Hesaplanmış attention skorlarını döndür"""
        return self.attention_scores
    
    def _format_parameters(self, config: Dict) -> str:
        """Parametreleri formatlanmış metin olarak döndür"""
        if not config:
            return "Parametreler: Bilgi yok"
        
        param_parts = []
        
        # Model parametreleri
        if 'd_model' in config:
            param_parts.append(f"d_model={config['d_model']}")
        if 'num_heads' in config:
            param_parts.append(f"num_heads={config['num_heads']}")
        if 'num_layers' in config:
            param_parts.append(f"num_layers={config['num_layers']}")
        if 'dropout' in config:
            param_parts.append(f"dropout={config['dropout']}")
        if 'learning_rate' in config:
            param_parts.append(f"lr={config['learning_rate']}")
        
        # Eğitim parametreleri
        if 'epochs' in config:
            param_parts.append(f"epochs={config['epochs']}")
        if 'batch_size' in config:
            param_parts.append(f"batch_size={config['batch_size']}")
        
        # Veri bilgisi
        if 'vocab_size' in config:
            param_parts.append(f"vocab_size={config['vocab_size']}")
        elif 'vocab' in config and config['vocab']:
            param_parts.append(f"vocab_size={len(config['vocab'])}")
        
        if param_parts:
            return "Parametreler: " + " | ".join(param_parts)
        else:
            return "Parametreler: Bilgi yok"
