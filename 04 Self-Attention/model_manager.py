"""
Model Yönetim Modülü
Modellerin kaydedilmesi ve yüklenmesi için fonksiyonlar
"""

import torch
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import torch.nn as nn
import numpy as np


class ModelManager:
    """Model kaydetme ve yükleme sınıfı"""
    
    def __init__(self, models_dir: str = "models"):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(CURRENT_DIR, models_dir)        
        print(f"🟢 Model klasör yolu: {self.models_dir}")        

        # Models klasörünü oluştur
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def save_model(self, model: nn.Module, config: Dict, name: str, attention_scores: Dict = None,
                   attention_weights: Optional[np.ndarray] = None, qkv_matrices: Optional[Dict] = None, 
                   history: Optional[Dict] = None):
        """
        Modeli ve konfigürasyonu kaydet
        
        Args:
            model: PyTorch modeli
            config: Model konfigürasyonu
            name: Model adı
            attention_scores: Attention skorları (opsiyonel)
            attention_weights: Attention weight matrisi (opsiyonel)
            qkv_matrices: QKV matrisleri (opsiyonel)
            history: Eğitim geçmişi (opsiyonel)
        """
        # Zaman damgası ekle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{name}_{timestamp}"
        
        # Klasör oluştur
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Model ağırlıklarını kaydet
        weights_path = os.path.join(model_path, "model_weights.pth")
        torch.save(model.state_dict(), weights_path)
        
        # Tam modeli kaydet (architecture + weights)
        full_model_path = os.path.join(model_path, "full_model.pth")
        torch.save(model, full_model_path)
        
        # Konfigürasyonu kaydet
        config_path = os.path.join(model_path, "config.json")
        
        # Token mapping'leri string key'lere çevir
        config_to_save = config.copy()
        if 'idx_to_token' in config_to_save and config_to_save['idx_to_token']:
            config_to_save['idx_to_token'] = {
                str(k): v for k, v in config_to_save['idx_to_token'].items()
            }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        
        # Model bilgilerini kaydet
        info = {
            'name': name,
            'timestamp': timestamp,
            'full_name': model_name,
            'config': config_to_save,
            'save_date': datetime.now().isoformat()
        }
        
        info_path = os.path.join(model_path, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Model kaydedildi: {model_path}")
        
        # Model özetini oluştur ve kaydet
        summary_file = os.path.join(model_path, "model_summary.txt")
        self._generate_and_save_summary(info, summary_file)
        
        # Attention skorlarını kaydet (varsa)
        if attention_scores:
            scores_file = os.path.join(model_path, "attention_scores.json")
            self._save_attention_scores(attention_scores, scores_file)
        
        # Attention weights'i kaydet (varsa)
        if attention_weights is not None:
            weights_file = os.path.join(model_path, "attention_weights.npy")
            np.save(weights_file, attention_weights)
            print(f"💾 Attention weights kaydedildi: attention_weights.npy")
        
        # QKV matrislerini kaydet (varsa)
        if qkv_matrices:
            qkv_file = os.path.join(model_path, "qkv_matrices.npz")
            np.savez(qkv_file, 
                    Q=qkv_matrices['Q'], 
                    K=qkv_matrices['K'], 
                    V=qkv_matrices['V'])
            print(f"💾 QKV matrisleri kaydedildi: qkv_matrices.npz")
        
        # Eğitim geçmişini kaydet (varsa)
        if history:
            history_file = os.path.join(model_path, "training_history.json")
            # NumPy tiplerini dönüştür
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            history_to_save = convert_numpy_types(history)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=4, ensure_ascii=False)
            print(f"💾 Eğitim geçmişi kaydedildi: training_history.json")
        
        return model_path
    
    def load_model(self, name: str) -> Tuple[nn.Module, Dict, Optional[np.ndarray], 
                                              Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        Modeli, konfigürasyonu ve tüm ilgili verileri yükle
        
        Args:
            name: Model adı (tam klasör adı)
            
        Returns:
            (model, config, attention_weights, qkv_matrices, attention_scores, history) tuple'ı
        """
        model_path = os.path.join(self.models_dir, name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        
        # Konfigürasyonu yükle
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # idx_to_token'ı integer key'lere çevir
        if 'idx_to_token' in config and config['idx_to_token']:
            config['idx_to_token'] = {
                int(k): v for k, v in config['idx_to_token'].items()
            }
        
        # Tam modeli yükle
        full_model_path = os.path.join(model_path, "full_model.pth")
        
        if not os.path.exists(full_model_path):
            raise FileNotFoundError("Model dosyası bulunamadı!")
        
        # PyTorch 2.6+ güvenlik ayarı: weights_only=False gerekli
        # Bu dosya güvenilir kaynaktan geldiği için (kendi modelimiz) güvenli
        model = torch.load(full_model_path, weights_only=False)
        print(f"✅ Model yüklendi: {model_path}")
        
        # Attention weights'i yükle (varsa)
        attention_weights = None
        weights_file = os.path.join(model_path, "attention_weights.npy")
        if os.path.exists(weights_file):
            attention_weights = np.load(weights_file)
            print(f"✅ Attention weights yüklendi")
        
        # QKV matrislerini yükle (varsa)
        qkv_matrices = None
        qkv_file = os.path.join(model_path, "qkv_matrices.npz")
        if os.path.exists(qkv_file):
            qkv_data = np.load(qkv_file)
            qkv_matrices = {
                'Q': qkv_data['Q'],
                'K': qkv_data['K'],
                'V': qkv_data['V']
            }
            print(f"✅ QKV matrisleri yüklendi")
        
        # Attention skorlarını yükle (varsa)
        attention_scores = None
        scores_file = os.path.join(model_path, "attention_scores.json")
        if os.path.exists(scores_file):
            with open(scores_file, 'r', encoding='utf-8') as f:
                attention_scores = json.load(f)
            print(f"✅ Attention skorları yüklendi")
        
        # Eğitim geçmişini yükle (varsa)
        history = None
        history_file = os.path.join(model_path, "training_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            print(f"✅ Eğitim geçmişi yüklendi")
        
        return model, config, attention_weights, qkv_matrices, attention_scores, history
    
    def list_models(self) -> List[str]:
        """Kaydedilmiş modellerin listesini döndür"""
        print(f"Modeller listeleniyor: {self.models_dir}")

        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                # Model info dosyası var mı kontrol et
                info_path = os.path.join(item_path, "model_info.json")
                if os.path.exists(info_path):
                    models.append(item)
        
        return sorted(models, reverse=True)  # En yeni önce
    
    def get_model_info(self, name: str) -> Dict:
        """Model bilgilerini al"""
        model_path = os.path.join(self.models_dir, name)
        info_path = os.path.join(model_path, "model_info.json")
        
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
    
    def delete_model(self, name: str):
        """Modeli sil"""
        import shutil
        model_path = os.path.join(self.models_dir, name)
        
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            print(f"Model silindi: {model_path}")
        else:
            print(f"Model bulunamadı: {model_path}")
    
    def _save_attention_scores(self, attention_scores: Dict, output_file: str):
        """
        Attention skorlarını dosyaya kaydet
        
        Args:
            attention_scores: Attention skorları
            output_file: Çıktı dosyası
        """
        # NumPy tiplerini Python native tiplerine çevir
        def convert_numpy_types(obj):
            """NumPy tiplerini JSON-serializable tiplere çevir"""
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Skorları dönüştür
        scores_to_save = convert_numpy_types(attention_scores)
        
        # JSON dosyası olarak kaydet
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scores_to_save, f, indent=4, ensure_ascii=False)
        
        # Okunabilir metin dosyası da oluştur
        txt_file = output_file.replace('.json', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("═══════════════════════════════════════════════════════════\n")
            f.write("ATTENTION SKORLARI RAPORU\n")
            f.write("═══════════════════════════════════════════════════════════\n\n")
            
            f.write(f"📊 Genel İstatistikler:\n")
            f.write(f"   • Ortalama Attention: {attention_scores['avg_attention']:.4f}\n")
            f.write(f"   • Maksimum Attention: {attention_scores['max_attention']:.4f}\n")
            f.write(f"   • Minimum Attention: {attention_scores['min_attention']:.4f}\n")
            f.write(f"   • Token Sayısı: {len(attention_scores['tokens'])}\n\n")
            
            f.write("───────────────────────────────────────────────────────────\n")
            f.write("TOKEN BAZLI DETAYLAR\n")
            f.write("───────────────────────────────────────────────────────────\n\n")
            
            for token_data in attention_scores['tokens']:
                f.write(f"\n🔤 Token: {token_data['token']}\n")
                f.write(f"   Index: {token_data['index']}\n")
                f.write(f"   Self-Attention: {token_data['self_attention']:.4f}\n\n")
                
                f.write(f"   📤 Verilen Attention (Query olarak):\n")
                f.write(f"      Ortalama: {token_data['avg_given']:.4f}\n")
                f.write(f"      Maksimum: {token_data['max_given']:.4f}\n")
                f.write(f"      En çok attention verdiği tokenlar:\n")
                for target, score in token_data['top_given']:
                    f.write(f"         → {target}: {score:.4f}\n")
                
                f.write(f"\n   📥 Alınan Attention (Key olarak):\n")
                f.write(f"      Ortalama: {token_data['avg_received']:.4f}\n")
                f.write(f"      Maksimum: {token_data['max_received']:.4f}\n")
                f.write(f"      En çok attention aldığı tokenlar:\n")
                for source, score in token_data['top_received']:
                    f.write(f"         ← {source}: {score:.4f}\n")
                
                if 'q_norm' in token_data:
                    f.write(f"\n   📐 QKV Norm Değerleri:\n")
                    f.write(f"      Q norm: {token_data['q_norm']:.4f}\n")
                    f.write(f"      K norm: {token_data['k_norm']:.4f}\n")
                    f.write(f"      V norm: {token_data['v_norm']:.4f}\n")
                
                f.write("\n" + "─"*60 + "\n")
            
            f.write("\n═══════════════════════════════════════════════════════════\n")
            f.write(f"Rapor Oluşturma Zamanı: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write("═══════════════════════════════════════════════════════════\n")
        
        print(f"📊 Attention skorları kaydedildi:")
        print(f"   • JSON: {output_file}")
        print(f"   • TXT:  {txt_file}\n")
    
    def _generate_and_save_summary(self, info: Dict, output_file: str):
        """
        Model özetini oluştur ve kaydet (Internal method)
        
        Args:
            info: Model bilgileri
            output_file: Çıktı dosyası yolu
        """
        # Tarih formatını daha okunabilir yap
        save_date = info.get('save_date', 'N/A')
        if save_date != 'N/A':
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(save_date)
                save_date = dt.strftime("%d.%m.%Y %H:%M:%S")
            except:
                pass
        
        # Özet oluştur
        summary = f"""═══════════════════════════════════════════════════════════
MODEL ÖZETİ - SELF-ATTENTION ÖĞRENME ARACI
═══════════════════════════════════════════════════════════

📋 GENEL BİLGİLER
───────────────────────────────────────────────────────────
Model Adı          : {info.get('name', 'N/A')}
Kayıt Tarihi       : {save_date}
Tam Klasör Adı     : {info.get('full_name', 'N/A')}

⚙️ MODEL PARAMETRELERİ
───────────────────────────────────────────────────────────
d_model            : {info['config'].get('d_model', 'N/A')} (Embedding Boyutu)
num_heads          : {info['config'].get('num_heads', 'N/A')} (Attention Head Sayısı)
num_layers         : {info['config'].get('num_layers', 'N/A')} (Katman Sayısı)
dropout            : {info['config'].get('dropout', 'N/A')}
learning_rate      : {info['config'].get('learning_rate', 'N/A')}

📚 EĞİTİM PARAMETRELERİ
───────────────────────────────────────────────────────────
epochs             : {info['config'].get('epochs', 'N/A')}
batch_size         : {info['config'].get('batch_size', 'N/A')}
vocab_size         : {info['config'].get('vocab_size', len(info['config'].get('vocab', [])))}

🔤 VOCABULARY
───────────────────────────────────────────────────────────
Vocabulary Boyutu  : {len(info['config'].get('vocab', []))}
Tokenlar           : {', '.join(info['config'].get('vocab', [])[:20])}{'...' if len(info['config'].get('vocab', [])) > 20 else ''}

═══════════════════════════════════════════════════════════
Oluşturulma Zamanı: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}
═══════════════════════════════════════════════════════════
"""
        
        # Konsola yazdır
        print("\n" + "="*63)
        print("📊 MODEL ÖZETİ OLUŞTURULDU")
        print("="*63)
        print(summary)
        
        # Dosyaya kaydet
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"💾 Özet dosyası kaydedildi: {output_file}\n")
    
    def export_model_summary(self, name: str, output_file: str = None):
        """
        Model özetini dışa aktar (Public API)
        
        Args:
            name: Model adı (klasör adı)
            output_file: Opsiyonel çıktı dosyası
        """
        info = self.get_model_info(name)
        
        if not info:
            print("❌ Model bilgisi bulunamadı!")
            return None
        
        # Özet oluştur
        if output_file:
            self._generate_and_save_summary(info, output_file)
        else:
            # Sadece konsola yazdır
            model_path = os.path.join(self.models_dir, name)
            temp_file = os.path.join(model_path, "temp_summary.txt")
            self._generate_and_save_summary(info, temp_file)
            
            # Temp dosyayı oku ve sil
            with open(temp_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            os.remove(temp_file)
            
            return summary
