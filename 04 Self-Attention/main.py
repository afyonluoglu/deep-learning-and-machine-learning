"""
Self-Attention Öğrenme Aracı
Yapay sinir ağlarında Self-Attention mekanizmasını öğrenmek ve uygulamak için interaktif bir araç

Date: Sept 2025
"""

import customtkinter as ctk
from self_attention_module import SelfAttentionTrainer
from visualization_module import VisualizationPanel
from model_manager import ModelManager
import tkinter as tk
from tkinter import messagebox
import webbrowser
import os
import sys


class SelfAttentionApp(ctk.CTk):
    """Ana uygulama sınıfı"""
    
    def __init__(self):
        super().__init__()
        
        # Pencere ayarları
        self.title("Self-Attention Öğrenme Aracı v1.0")
        self.geometry("1500x900")  # Sol panel genişlediği için 1400'den 1500'e çıkarıldı
        
        # Pencereyi ortala
        self.center_window(1500, 900)
        
        # Pencereyi en üste getir
        self.attributes('-topmost', True)
        self.after(100, lambda: self.attributes('-topmost', False))  # Sadece açılışta
        
        # Tema ayarı
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ana bileşenler
        self.trainer = SelfAttentionTrainer()
        self.model_manager = ModelManager()
        
        # UI oluştur
        self.create_ui()
        
        # Varsayılan değerleri yükle
        self.load_default_example()
    
    def center_window(self, width, height):
        """Pencereyi ekranın ortasına yerleştir"""
        # Ekran boyutlarını al
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Merkez koordinatlarını hesapla
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Pencereyi konumlandır
        self.geometry(f"{width}x{height}+{x}+{y}")
        
    def create_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        
        # Ana container
        self.grid_columnconfigure(0, weight=0, minsize=520)  # Sol panel için sabit genişlik
        self.grid_columnconfigure(1, weight=1)  # Sağ panel genişleyebilir
        self.grid_rowconfigure(0, weight=1)
        
        # Sol panel - Kontroller
        self.create_control_panel()
        
        # Sağ panel - Görselleştirmeler
        self.create_visualization_panel()
        
    def create_control_panel(self):
        """Sol kontrol panelini oluştur"""
        
        control_frame = ctk.CTkFrame(self, width=520)  # 450'den 520'ye artırıldı - başlıklar sığıyor
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        control_frame.grid_propagate(False)
        
        # Başlık
        title_label = ctk.CTkLabel(
            control_frame, 
            text="Self-Attention Eğitim Paneli",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=10, padx=10)
        
        # Scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(control_frame)
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Örnek Veri Seçimi
        self.create_data_section(scrollable_frame)
        
        # Parametreler
        self.create_parameters_section(scrollable_frame)
        
        # Eğitim Kontrolleri
        self.create_training_section(scrollable_frame)
        
        # Model Yönetimi
        self.create_model_management_section(scrollable_frame)
        
        # Yardım butonu
        help_btn = ctk.CTkButton(
            control_frame,
            text="📖 Yardım ve Öğretici",
            command=self.show_help,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        help_btn.pack(side="bottom", pady=10, padx=10, fill="x")
        
    def create_data_section(self, parent):
        """Veri seçim bölümü"""
        
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=10, padx=5)
        
        label = ctk.CTkLabel(
            section, 
            text="📊 Örnek Veri Seti",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(pady=5, padx=10, anchor="w")
        
        # Örnek seçimi
        self.example_var = ctk.StringVar(value="Kelime Dizisi")
        examples = [
            "Kelime Dizisi",
            "Cümle Analizi",
            "Zaman Serisi",
            "Görüntü Parçaları",
            "Özel Veri"
        ]
        
        example_menu = ctk.CTkOptionMenu(
            section,
            values=examples,
            variable=self.example_var,
            command=self.on_example_change
        )
        example_menu.pack(pady=5, padx=10, fill="x")
        
        # Veri girişi
        data_label = ctk.CTkLabel(section, text="Giriş Verisi (her satır bir token):")
        data_label.pack(pady=(10,5), padx=10, anchor="w")
        
        self.data_text = ctk.CTkTextbox(section, height=100)
        self.data_text.pack(pady=5, padx=10, fill="x")
        
    def create_parameters_section(self, parent):
        """Parametre ayar bölümü"""
        
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=10, padx=5)
        
        label = ctk.CTkLabel(
            section, 
            text="⚙️ Self-Attention Parametreleri",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(pady=5, padx=10, anchor="w")
        
        # Embedding boyutu
        self.d_model_slider = self.create_parameter_slider(
            section, "Embedding Boyutu (d_model):", 
            32, 512, 64, self.set_d_model
        )
        
        # Head sayısı
        self.num_heads_slider = self.create_parameter_slider(
            section, "Attention Head Sayısı (num_heads):", 
            1, 16, 4, self.set_num_heads
        )
        
        # Dropout
        self.dropout_slider = self.create_parameter_slider(
            section, "Dropout Oranı:", 
            0, 0.5, 0.1, self.set_dropout, step=0.05
        )
        
        # Learning rate
        self.learning_rate_slider = self.create_parameter_slider(
            section, "Öğrenme Hızı (Learning Rate):", 
            0.0001, 0.01, 0.001, self.set_learning_rate, step=0.0001
        )
        
    def create_parameter_slider(self, parent, text, from_, to, default, command, step=1):
        """Parametre slider'ı oluştur"""
        
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=5, padx=10)
        
        # Başlık (üstte)
        label = ctk.CTkLabel(container, text=text, anchor="w")
        label.pack(anchor="w", pady=(0, 2))
        
        # Değer label'ı (sağda, slider'ın üstünde)
        value_frame = ctk.CTkFrame(container, fg_color="transparent")
        value_frame.pack(fill="x")
        
        value_label = ctk.CTkLabel(
            value_frame, 
            text=str(int(default)) if default == int(default) else str(default),
            anchor="e",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#61afef"
        )
        value_label.pack(side="right")
                
        # Slider oluştur
        slider = ctk.CTkSlider(
            container,
            from_=from_,
            to=to,
            number_of_steps=int((to-from_)/step),
            command=lambda v: self.update_slider(v, value_label, command)
        )
        slider.set(default)
        slider.pack(fill="x", pady=(2, 0))
        
        # Slider'a parametreleri ekle (dialog için)
        slider._from = from_
        slider._to = to
        slider._step = step
        slider._value_label = value_label
        # slider._command = command
        slider._param_name = text
        
        # Çift tıklama için bind ekle - slider referansı ile
        label.bind("<Double-Button-1>", 
                   lambda e: self.open_value_dialog(slider))
        label.configure(cursor="hand2")  # El imleci göster
        
        return slider
    
    def update_slider(self, value, label, command):
        """Slider değerini güncelle"""
        try:
            # Akıllı formatlama: tam sayılar için .0 gösterme, ondalıklılar için 4 basamak
            if value == int(value):
                formatted_value = str(int(value))
            else:
                formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
            
            # Label'ı güncelle
            label.configure(text=formatted_value)
            label.update_idletasks()  # GUI'yi güncelle
            
            # print(f"[DEBUG - Update Slider] Slider güncellendi: {formatted_value}")
            # Command'ı çağır
            command(value)
        except Exception as e:
            print(f"[HATA] Slider güncelleme hatası: {e}")
            import traceback
            traceback.print_exc()
    
    def open_value_dialog(self, slider):
        """Parametre değeri için dialog aç"""
        
        # Slider'dan bilgileri al
        param_name = slider._param_name
        from_ = slider._from
        to = slider._to
        step = slider._step
        value_label = slider._value_label
        current_value = slider.get()  
        
        # Dialog penceresi oluştur
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"Değer Gir: {param_name}")
        dialog.geometry("400x280")
        dialog.transient(self)
        dialog.grab_set()
        
        # Pencereyi ortala
        dialog.update_idletasks()
        width = 400
        height = 280
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Başlık
        title_label = ctk.CTkLabel(
            dialog, 
            text=param_name,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=15)
        
        # Mevcut değer göster
        current_label = ctk.CTkLabel(
            dialog,
            text=f"Mevcut değer: {current_value if current_value != int(current_value) else int(current_value)}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#98c379"
        )
        current_label.pack(pady=5)
        
        # Aralık bilgisi
        range_label = ctk.CTkLabel(
            dialog,
            text=f"Geçerli aralık: {from_} - {to}",
            font=ctk.CTkFont(size=11)
        )
        range_label.pack(pady=5)
        
        # Giriş alanı
        entry_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        entry_frame.pack(pady=10)
        
        entry_label = ctk.CTkLabel(entry_frame, text="Değer:")
        entry_label.pack(side="left", padx=5)
        
        entry = ctk.CTkEntry(entry_frame, width=150)
        entry.insert(0, str(current_value))
        entry.pack(side="left", padx=5)
        entry.focus()
        
        # Hata mesajı label'ı
        error_label = ctk.CTkLabel(dialog, text="", text_color="red")
        error_label.pack(pady=5)
        
        def apply_value():
            """Değeri uygula"""
            try:
                new_value = float(entry.get())                
                # Aralık kontrolü
                if new_value < from_ or new_value > to:
                    error_label.configure(text=f"Hata: Değer {from_} - {to} aralığında olmalı!")
                    return
                
                # Slider'ı güncelle
                slider.set(new_value)
                slider._command(slider.get())
                
                # Label'ı güncelle (akıllı formatlama)
                if new_value == int(new_value):
                    formatted = str(int(new_value))
                else:
                    formatted = f"{new_value:.4f}".rstrip('0').rstrip('.')
                
                value_label.configure(text=formatted)
                value_label.update()
                # print(f"[DEBUG] - Dialog: Label yeni değeri: {value_label.cget('text')}  {self.trainer.d_model}")
                                
                dialog.destroy()
                
            except ValueError as e:
                error_label.configure(text="Hata: Geçerli bir sayı girin!")
                print(f"[DEBUG] ValueError: {e}")
        
        # Butonlar
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=15)
        
        ok_btn = ctk.CTkButton(
            button_frame,
            text="✓ Tamam",
            command=apply_value,
            width=100,
            fg_color="green"
        )
        ok_btn.pack(side="left", padx=5)
        
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="✗ İptal",
            command=dialog.destroy,
            width=100,
            fg_color="gray"
        )
        cancel_btn.pack(side="left", padx=5)
        
        # Enter tuşu ile onayla
        entry.bind("<Return>", lambda e: apply_value())
        entry.bind("<Escape>", lambda e: dialog.destroy())
        
    def create_training_section(self, parent):
        """Eğitim kontrol bölümü"""
        
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=10, padx=5)
        
        label = ctk.CTkLabel(
            section, 
            text="🎯 Eğitim Kontrolleri",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(pady=5, padx=10, anchor="w")
        
        # Epoch sayısı
        epoch_frame = ctk.CTkFrame(section, fg_color="transparent")
        epoch_frame.pack(fill="x", pady=5, padx=10)
        
        epoch_label = ctk.CTkLabel(epoch_frame, text="Epoch Sayısı:")
        epoch_label.pack(side="left")
        
        self.epoch_entry = ctk.CTkEntry(epoch_frame, width=100)
        self.epoch_entry.insert(0, "50")
        self.epoch_entry.pack(side="right")
        
        # Batch size
        batch_frame = ctk.CTkFrame(section, fg_color="transparent")
        batch_frame.pack(fill="x", pady=5, padx=10)
        
        batch_label = ctk.CTkLabel(batch_frame, text="Batch Size:")
        batch_label.pack(side="left")
        
        self.batch_entry = ctk.CTkEntry(batch_frame, width=100)
        self.batch_entry.insert(0, "8")
        self.batch_entry.pack(side="right")
        
        # Eğitim butonu
        self.train_btn = ctk.CTkButton(
            section,
            text="🚀 Eğitimi Başlat",
            command=self.start_training,
            height=40,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.train_btn.pack(pady=10, padx=10, fill="x")
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(section)
        self.progress.pack(pady=5, padx=10, fill="x")
        self.progress.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            section, 
            text="Hazır",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=5)
        
    def create_model_management_section(self, parent):
        """Model yönetim bölümü"""
        
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=10, padx=5)
        
        label = ctk.CTkLabel(
            section, 
            text="💾 Model Yönetimi",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(pady=5, padx=10, anchor="w")
        
        # Model adı
        name_frame = ctk.CTkFrame(section, fg_color="transparent")
        name_frame.pack(fill="x", pady=5, padx=10)
        
        name_label = ctk.CTkLabel(name_frame, text="Model Adı:")
        name_label.pack(side="left")
        
        self.model_name_entry = ctk.CTkEntry(name_frame, width=200)
        self.model_name_entry.insert(0, "self_attention_model")
        self.model_name_entry.pack(side="right")
        
        # Kaydet butonu
        save_btn = ctk.CTkButton(
            section,
            text="💾 Modeli Kaydet",
            command=self.save_model,
            height=35
        )
        save_btn.pack(pady=5, padx=10, fill="x")
        
        # Yükle butonu
        load_btn = ctk.CTkButton(
            section,
            text="📂 Model Yükle",
            command=self.load_model,
            height=35
        )
        load_btn.pack(pady=5, padx=10, fill="x")
        
    def create_visualization_panel(self):
        """Sağ görselleştirme panelini oluştur"""
        
        viz_frame = ctk.CTkFrame(self)
        viz_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Görselleştirme paneli
        self.viz_panel = VisualizationPanel(viz_frame)
        self.viz_panel.pack(fill="both", expand=True)
        
    def load_default_example(self):
        """Varsayılan örneği yükle"""
        default_text = "Ben\nBugün\nMarkete\nve\nOkula\nGittim"
        self.data_text.delete("1.0", "end")
        self.data_text.insert("1.0", default_text)
        
    def on_example_change(self, choice):
        """Örnek değiştiğinde"""
        examples = {
            "Kelime Dizisi": "Ben\nBugün\nMarkete\nve\nOkula\nGittim",
            "Cümle Analizi": "Kedi\nMat\nÜzerinde\nOturdu",
            "Zaman Serisi": "Pazartesi\nSalı\nÇarşamba\nPerşembe\nCuma",
            "Görüntü Parçaları": "Patch_1\nPatch_2\nPatch_3\nPatch_4",
            "Özel Veri": ""
        }
        
        text = examples.get(choice, "")
        self.data_text.delete("1.0", "end")
        self.data_text.insert("1.0", text)
        
    def set_d_model(self, value):
        """d_model parametresini ayarla"""
        self.trainer.set_d_model(int(value))
        # print(f"[DEBUG - main.py] d_model set to {int(value)}")
        
    def set_num_heads(self, value):
        """Head sayısını ayarla"""
        self.trainer.set_num_heads(int(value))
        
    def set_dropout(self, value):
        """Dropout oranını ayarla"""
        self.trainer.set_dropout(float(value))
        
    def set_learning_rate(self, value):
        """Learning rate'i ayarla"""
        self.trainer.set_learning_rate(float(value))
        
    def start_training(self):
        """Eğitimi başlat"""
        try:
            if self.trainer.d_model % self.trainer.num_heads != 0:
                alt_kat = (self.trainer.d_model // self.trainer.num_heads) * self.trainer.num_heads
                ust_kat = alt_kat + self.trainer.num_heads
                solution = f"'d_model' için önerilen en yakın değerler: {alt_kat} veya {ust_kat}"

                print(f"[HATA] d_model ({self.trainer.d_model}), num_heads'in ({self.trainer.num_heads}) katı değil.\n{solution}")
                messagebox.showwarning(
                    "Uyarı", 
                    f"d_model değeri ({self.trainer.d_model}), num_heads değerinin ({self.trainer.num_heads}) katı olmalıdır!\n{solution}"
                )
                return

            # Veriyi al
            data_text = self.data_text.get("1.0", "end").strip()
            if not data_text:
                messagebox.showwarning("Uyarı", "Lütfen veri girişi yapın!")
                return
            
            tokens = [line.strip() for line in data_text.split("\n") if line.strip()]
            
            # Parametreleri al
            epochs = int(self.epoch_entry.get())
            batch_size = int(self.batch_entry.get())
            
            # UI'ı güncelle
            self.train_btn.configure(state="disabled")
            self.status_label.configure(text="Eğitim başlıyor...")
            self.progress.set(0)
            self.update()
            
            print("  Eğitim başlıyor...")
            # Eğitimi başlat
            history = self.trainer.train(
                tokens, 
                epochs=epochs, 
                batch_size=batch_size,
                progress_callback=self.update_progress
            )
            print("🎉 Eğitim tamamlandı.")
            
            # Görselleştirmeleri güncelle
            self.visualize_results(tokens)
            
            # UI'ı güncelle
            self.train_btn.configure(state="normal")
            self.status_label.configure(text="Eğitim tamamlandı! ✓")
            self.progress.set(1)
            
            messagebox.showinfo("Başarılı", "Model eğitimi başarıyla tamamlandı!")
            
        except Exception as e:
            self.train_btn.configure(state="normal")
            self.status_label.configure(text="Hata oluştu!")
            messagebox.showerror("Hata", f"Eğitim sırasında hata:\n{str(e)}")
            
    def update_progress(self, epoch, total_epochs, loss):
        """İlerleme çubuğunu güncelle"""
        progress = (epoch + 1) / total_epochs
        self.progress.set(progress)
        self.status_label.configure(text=f"Epoch {epoch+1}/{total_epochs} - Loss: {loss:.4f}")
        self.update()
        
    def visualize_results(self, tokens):
        """Sonuçları görselleştir"""
        # Attention map'i al
        attention_weights = self.trainer.get_attention_weights(tokens)
        
        # QKV matrislerini al
        qkv_matrices = self.trainer.get_qkv_matrices(tokens)
        
        # Eğitim geçmişini al
        history = self.trainer.get_training_history()
        
        # Config bilgisini al ve eğitim parametrelerini ekle
        config = self.trainer.get_config()
        config['epochs'] = int(self.epoch_entry.get())
        config['batch_size'] = int(self.batch_entry.get())
        config['vocab_size'] = len(tokens)
        
        # Görselleştir
        self.viz_panel.visualize_all(
            tokens=tokens,
            attention_weights=attention_weights,
            qkv_matrices=qkv_matrices,
            history=history,
            config=config
        )
        
    def save_model(self):
        """Modeli kaydet"""
        try:
            model_name = self.model_name_entry.get().strip()
            if not model_name:
                messagebox.showwarning("Uyarı", "Lütfen model adı girin!")
                return
            
            # Config'den vocab'u al (eğitim sırasında kullanılan gerçek token listesi)
            config = self.trainer.get_config()
            tokens = config.get('vocab', [])
            
            # Veriyi text box'tan da al (kullanıcı görsün diye)
            data_text = self.data_text.get("1.0", "end").strip()
            input_tokens = [line.strip() for line in data_text.split("\n") if line.strip()]
            
            print(f"\n{'='*60}")
            print(f"💾 MODEL KAYDETME İŞLEMİ BAŞLIYOR")
            print(f"{'='*60}")
            print(f"Model adı: {model_name}")
            print(f"Input tokens (text box): {input_tokens}")
            print(f"Vocab tokens (model): {tokens}")
            print(f"Vocab boyutu: {len(tokens)}")
            
            # Tüm verileri al
            try:
                attention_scores = self.viz_panel.get_attention_scores()
                print(f"✓ Attention skorları alındı")
            except Exception as e:
                print(f"✗ Attention skorları alınamadı: {e}")
                attention_scores = None
            
            try:
                # Vocab ile attention weights al
                attention_weights = self.trainer.get_attention_weights(tokens)
                print(f"✓ Attention weights alındı: shape={attention_weights.shape if attention_weights is not None else 'None'}")
            except Exception as e:
                print(f"✗ Attention weights alınamadı: {e}")
                import traceback
                traceback.print_exc()
                attention_weights = None
            
            try:
                # Vocab ile QKV matrisleri al
                qkv_matrices = self.trainer.get_qkv_matrices(tokens)
                if qkv_matrices:
                    print(f"✓ QKV matrisleri alındı: Q={qkv_matrices['Q'].shape}, K={qkv_matrices['K'].shape}, V={qkv_matrices['V'].shape}")
                else:
                    print(f"✗ QKV matrisleri None")
            except Exception as e:
                print(f"✗ QKV matrisleri alınamadı: {e}")
                import traceback
                traceback.print_exc()
                qkv_matrices = None
            
            try:
                history = self.trainer.get_training_history()
                print(f"✓ Eğitim geçmişi alındı")
            except Exception as e:
                print(f"✗ Eğitim geçmişi alınamadı: {e}")
                history = None
            
            # Modeli kaydet
            print(f"\n📦 Model kaydediliyor...")
            self.model_manager.save_model(
                model=self.trainer.model,
                config=self.trainer.get_config(),
                name=model_name,
                attention_scores=attention_scores,
                attention_weights=attention_weights,
                qkv_matrices=qkv_matrices,
                history=history
            )
            
            print(f"{'='*60}\n")
            messagebox.showinfo("Başarılı", f"Model '{model_name}' olarak kaydedildi!")
            
        except Exception as e:
            print(f"\n❌ MODEL KAYDETME HATASI:")
            print(f"{str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            messagebox.showerror("Hata", f"Model kaydedilemedi:\n{str(e)}")
            
    def load_model(self):
        """Model yükle"""
        try:
            # Model listesini göster
            models = self.model_manager.list_models()
            
            if not models:
                messagebox.showinfo("Bilgi", "Kaydedilmiş model bulunamadı!")
                return
            
            # Model seçim dialogu
            dialog = ModelSelectionDialog(self, models)
            self.wait_window(dialog)
            
            if dialog.selected_model:
                print(f"\n{'='*60}")
                print(f"📂 MODEL YÜKLEME İŞLEMİ BAŞLIYOR")
                print(f"{'='*60}")
                print(f"Seçilen model: {dialog.selected_model}")
                
                # Modeli ve tüm verileri yükle
                model, config, attention_weights, qkv_matrices, attention_scores, history = \
                    self.model_manager.load_model(dialog.selected_model)
                
                print(f"\n📋 Config bilgileri:")
                print(f"   vocab: {config.get('vocab', 'N/A')}")
                print(f"   tokens: {config.get('tokens', 'N/A')}")
                print(f"   d_model: {config.get('d_model', 'N/A')}")
                print(f"   num_heads: {config.get('num_heads', 'N/A')}")
                
                if attention_weights is not None:
                    print(f"\n📊 Yüklenen veri boyutları:")
                    print(f"   Attention weights: {attention_weights.shape}")
                if qkv_matrices:
                    print(f"   Q matrix: {qkv_matrices['Q'].shape}")
                    print(f"   K matrix: {qkv_matrices['K'].shape}")
                    print(f"   V matrix: {qkv_matrices['V'].shape}")
                
                self.trainer.load_model(model, config)
                
                # UI'ı güncelle - Model ismi
                self.model_name_entry.delete(0, "end")
                self.model_name_entry.insert(0, dialog.selected_model)
                
                # Parametreleri UI'a yansıt
                if 'd_model' in config:
                    self.d_model_slider.set(config['d_model'])
                    self.d_model_slider._value_label.configure(text=str(config['d_model']))
                    
                if 'num_heads' in config:
                    self.num_heads_slider.set(config['num_heads'])
                    self.num_heads_slider._value_label.configure(text=str(config['num_heads']))
                    
                if 'dropout' in config:
                    self.dropout_slider.set(config['dropout'])
                    dropout_text = f"{config['dropout']:.2f}".rstrip('0').rstrip('.')
                    self.dropout_slider._value_label.configure(text=dropout_text)
                    
                if 'learning_rate' in config:
                    self.learning_rate_slider.set(config['learning_rate'])
                    lr_text = f"{config['learning_rate']:.4f}".rstrip('0').rstrip('.')
                    self.learning_rate_slider._value_label.configure(text=lr_text)
                
                # Epoch ve batch size'ı güncelle
                if 'epochs' in config:
                    self.epoch_entry.delete(0, "end")
                    self.epoch_entry.insert(0, str(config['epochs']))
                    
                if 'batch_size' in config:
                    self.batch_entry.delete(0, "end")
                    self.batch_entry.insert(0, str(config['batch_size']))
                
                # Veri metnini güncelle
                tokens_text = ""
                if 'vocab' in config and config['vocab']:
                    tokens_text = "\n".join(config['vocab'])
                elif 'tokens' in config and config['tokens']:
                    tokens_text = "\n".join(config['tokens'])
                
                if tokens_text:
                    self.data_text.delete("1.0", "end")
                    self.data_text.insert("1.0", tokens_text)
                
                # Grafikleri göster (eğer kaydedilmiş veriler varsa)
                if attention_weights is not None and qkv_matrices is not None:
                    tokens = config.get('vocab', config.get('tokens', []))
                    
                    print(f"\n🔍 Boyut kontrolleri:")
                    print(f"   Token listesi: {tokens}")
                    print(f"   Token sayısı: {len(tokens)}")
                    print(f"   Attention weights shape: {attention_weights.shape}")
                    print(f"   Beklenen boyut: ({len(tokens)}, {len(tokens)})")
                    
                    # Boyut uyumluluğunu kontrol et
                    expected_size = len(tokens)
                    
                    if attention_weights.shape[0] != expected_size or attention_weights.shape[1] != expected_size:
                        print(f"\n⚠️ BOYUT UYUMSUZLUĞU TESPIT EDİLDİ!")
                        print(f"   Attention weights: {attention_weights.shape}")
                        print(f"   Beklenen: ({expected_size}, {expected_size})")
                        print(f"{'='*60}\n")
                        
                        messagebox.showwarning(
                            "Uyarı",
                            f"Attention weights boyutu ({attention_weights.shape}) token sayısı ({expected_size}) ile uyuşmuyor!\n\n"
                            f"Tokens: {tokens}\n\n"
                            f"Model yüklendi ama grafikler gösterilemiyor.\n"
                            f"Yeni bir eğitim yapın."
                        )
                        return
                    
                    if qkv_matrices['Q'].shape[0] != expected_size:
                        messagebox.showwarning(
                            "Uyarı",
                            f"QKV matrisleri boyutu token sayısı ile uyuşmuyor!\n\n"
                            f"Model yüklendi ama grafikler gösterilemiyor.\n"
                            f"Yeni bir eğitim yapın."
                        )
                        return
                    
                    try:
                        # Yüklenen verileri trainer'a ekle (tekrar eğitim yapmadan kullanmak için)
                        self.trainer.attention_weights = attention_weights
                        self.trainer.qkv_matrices = qkv_matrices
                        if history:
                            self.trainer.history = history
                        
                        # Görselleştirmeleri güncelle
                        self.viz_panel.visualize_all(
                            tokens,
                            attention_weights,
                            qkv_matrices,
                            history or {},
                            config
                        )
                        
                        # Attention skorlarını yükle (varsa)
                        if attention_scores:
                            self.viz_panel.attention_scores = attention_scores
                            # Skor tab'ını yeniden oluştur
                            self.viz_panel.calculate_and_visualize_scores(
                                tokens, attention_weights, qkv_matrices, config
                            )
                    except Exception as viz_error:
                        print(f"\n❌ GÖRSELLEŞTİRME HATASI:")
                        print(f"{str(viz_error)}")
                        import traceback
                        traceback.print_exc()
                        print(f"{'='*60}\n")
                        
                        messagebox.showerror(
                            "Görselleştirme Hatası",
                            f"Grafikler oluşturulurken hata:\n{str(viz_error)}\n\n"
                            f"Model yüklendi ama grafikler gösterilemiyor.\n"
                            f"Yeni bir eğitim yapın."
                        )
                        return
                    
                    print(f"\n✅ Grafikler başarıyla oluşturuldu!")
                    print(f"{'='*60}\n")
                    
                    messagebox.showinfo(
                        "Başarılı! ✅", 
                        f"Model '{dialog.selected_model}' yüklendi!\n\n"
                        f"📊 Yüklenen Veriler:\n"
                        f"✓ Model ağırlıkları\n"
                        f"✓ Attention weights\n"
                        f"✓ Q, K, V matrisleri\n"
                        f"{'✓ Eğitim geçmişi' if history else '✗ Eğitim geçmişi yok'}\n"
                        f"{'✓ Attention skorları' if attention_scores else '✗ Attention skorları yok'}\n\n"
                        f"🎨 Tüm grafikler hazır!"
                    )
                else:
                    print(f"\n⚠️ Grafik verileri bulunamadı")
                    print(f"{'='*60}\n")
                    
                    messagebox.showinfo(
                        "Başarılı", 
                        f"Model '{dialog.selected_model}' yüklendi!\n\n"
                        f"⚠️ Grafik verileri bulunamadı.\n"
                        f"Parametreler güncellendi.\n\n"
                        f"Grafikleri görmek için eğitim yapın."
                    )
                
        except Exception as e:
            print(f"\n❌ MODEL YÜKLEME HATASI:")
            print(f"{str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            messagebox.showerror("Hata", f"Model yüklenemedi:\n{str(e)}")
            
    def show_help(self):
        """Yardım ekranını göster"""
        help_file = os.path.join(os.path.dirname(__file__), "help.html")
        
        if os.path.exists(help_file):
            webbrowser.open('file://' + os.path.realpath(help_file))
        else:
            messagebox.showwarning(
                "Uyarı", 
                "Yardım dosyası bulunamadı!\nLütfen help.html dosyasının varlığını kontrol edin."
            )


class ModelSelectionDialog(ctk.CTkToplevel):
    """Model seçim dialog penceresi"""
    
    def __init__(self, parent, models):
        super().__init__(parent)
        
        self.selected_model = None
        
        self.title("Model Seçin")
        self.geometry("400x300")
        
        # Pencereyi ortala
        self.center_window(400, 300)
        
        # Dialog özelliklerini ayarla
        self.transient(parent)
        self.grab_set()
        
        # Pencereyi en üste getir
        self.attributes('-topmost', True)
        self.lift()
        self.focus_force()
        
        # Başlık
        label = ctk.CTkLabel(
            self, 
            text="Yüklenecek Modeli Seçin:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(pady=10)
        
        # Model listesi
        self.listbox = tk.Listbox(self, font=("Arial", 12))
        self.listbox.pack(fill="both", expand=True, padx=20, pady=10)
        
        for model in models:
            self.listbox.insert(tk.END, model)
        
        # Butonlar
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)
        
        select_btn = ctk.CTkButton(
            btn_frame,
            text="Seç",
            command=self.on_select,
            width=100
        )
        select_btn.pack(side="left", padx=5)
        
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="İptal",
            command=self.destroy,
            width=100
        )
        cancel_btn.pack(side="left", padx=5)
        
    def on_select(self):
        """Seçim yapıldığında"""
        selection = self.listbox.curselection()
        if selection:
            self.selected_model = self.listbox.get(selection[0])
            self.destroy()
    
    def center_window(self, width, height):
        """Pencereyi ekranın ortasına yerleştir"""
        # Ana pencere konumunu al
        parent_x = self.master.winfo_x()
        parent_y = self.master.winfo_y()
        parent_width = self.master.winfo_width()
        parent_height = self.master.winfo_height()
        
        # Dialog'u ana pencerenin ortasına yerleştir
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        # Pencereyi konumlandır
        self.geometry(f"{width}x{height}+{x}+{y}")


def main():
    """Ana fonksiyon"""
    app = SelfAttentionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
