"""
🛠️ RNN EĞİTİM PAKETİ - YARDIMCI FONKSİYONLAR
==========================================

Bu dosya RNN eğitim paketindeki tüm örneklerde kullanılan
ortak fonksiyonları içerir.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

# Stil ayarları
plt.style.use('default')
sns.set_palette("husl")

def print_section(title, char="=", width=50):
    """Bölüm başlığı yazdırır"""
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

def create_time_series_data(length=365, complexity='simple'):
    """
    Farklı karmaşıklıkta zaman serisi verisi oluşturur
    
    Args:
        length: Veri uzunluğu
        complexity: 'simple', 'medium', 'complex'
    
    Returns:
        np.array: Zaman serisi verisi
    """
    np.random.seed(42)
    time = np.arange(length)
    
    if complexity == 'simple':
        # Basit sinüs dalgası + gürültü
        data = 10 + 5 * np.sin(time * 0.1) + np.random.normal(0, 1, size=length)
        
    elif complexity == 'medium':
        # Trend + mevsimsellik + gürültü
        trend = np.linspace(0, 10, length)
        seasonal = 5 * np.sin(time * 2 * np.pi / 30) + 2 * np.cos(time * 2 * np.pi / 7)
        noise = np.random.normal(0, 2, size=length)
        data = 15 + trend + seasonal + noise
        
    elif complexity == 'complex':
        # Çoklu trend + mevsimsellik + volatilite
        trend = 10 + 0.01 * time + 0.0001 * time**2
        annual = 8 * np.sin(time * 2 * np.pi / 365)
        monthly = 3 * np.sin(time * 2 * np.pi / 30)
        weekly = 1.5 * np.sin(time * 2 * np.pi / 7)
        
        # GARCH benzeri volatilite
        volatility = np.zeros(length)
        volatility[0] = 1
        for i in range(1, length):
            volatility[i] = 0.1 + 0.85 * volatility[i-1] + 0.05 * np.random.randn()**2
        
        noise = np.random.randn(length) * np.sqrt(volatility)
        data = trend + annual + monthly + weekly + noise * 2
    
    return data

def create_sequences(data, window_size, prediction_steps=1):
    """
    Zaman serisi verisini RNN için sequence'lere dönüştürür
    
    Args:
        data: Ham zaman serisi verisi
        window_size: Giriş pencere boyutu
        prediction_steps: Tahmin adım sayısı
    
    Returns:
        X, y: Giriş ve hedef dizileri
    """
    X, y = [], []
    for i in range(len(data) - window_size - prediction_steps + 1):
        X.append(data[i:i + window_size])
        if prediction_steps == 1:
            y.append(data[i + window_size])
        else:
            y.append(data[i + window_size:i + window_size + prediction_steps])
    
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred, set_name="", verbose=True):
    """
    Regresyon metrikleri hesaplar
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        set_name: Set adı (eğitim, test, vs.)
        verbose: Yazdırma kontrolü
    
    Returns:
        dict: Metrik sonuçları
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }
    
    if verbose:
        print(f"📊 {set_name} Metrikleri:" if set_name else "📊 Metrikler:")
        print(f"   MSE:  {mse:.6f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²:   {r2:.4f}")
    
    return metrics

def plot_training_history(history, title="Model Eğitim Geçmişi"):
    """
    Model eğitim geçmişini görselleştirir
    
    Args:
        history: Keras training history
        title: Grafik başlığı
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss grafiği
    axes[0].plot(history.history['loss'], 'b-', label='Eğitim Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], 'r-', label='Validasyon Loss', linewidth=2)
    axes[0].set_title('📉 Model Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE grafiği (varsa)
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], 'b-', label='Eğitim MAE', linewidth=2)
        if 'val_mae' in history.history:
            axes[1].plot(history.history['val_mae'], 'r-', label='Validasyon MAE', linewidth=2)
        axes[1].set_title('📊 Mean Absolute Error', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Accuracy varsa
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], 'b-', label='Eğitim Accuracy', linewidth=2)
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history['val_accuracy'], 'r-', label='Validasyon Accuracy', linewidth=2)
            axes[1].set_title('📈 Model Accuracy', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, title="Tahmin Sonuçları", n_samples=200):
    """
    Tahmin sonuçlarını görselleştirir
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        title: Grafik başlığı
        n_samples: Gösterilecek örnek sayısı
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    axes[0].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.6, s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_title('🎯 Gerçek vs Tahmin', fontweight='bold')
    axes[0].set_xlabel('Gerçek Değerler')
    axes[0].set_ylabel('Tahmin Edilen Değerler')
    axes[0].grid(True, alpha=0.3)
    
    # Time series plot
    n_samples = min(n_samples, len(y_true))
    indices = range(n_samples)
    axes[1].plot(indices, y_true[-n_samples:].flatten(), 'b-', 
                label='Gerçek', linewidth=2, alpha=0.8)
    axes[1].plot(indices, y_pred[-n_samples:].flatten(), 'r-', 
                label='Tahmin', linewidth=2, alpha=0.8)
    axes[1].set_title(f'📈 Son {n_samples} Örnek - Zaman Serisi', fontweight='bold')
    axes[1].set_xlabel('Zaman')
    axes[1].set_ylabel('Değer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_error_analysis(y_true, y_pred, title="Hata Analizi"):
    """
    Detaylı hata analizi yapar
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        title: Grafik başlığı
    """
    errors = y_true.flatten() - y_pred.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hata histogramı
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('📊 Hata Dağılımı', fontweight='bold')
    axes[0, 0].set_xlabel('Hata (Gerçek - Tahmin)')
    axes[0, 0].set_ylabel('Frekans')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hata vs gerçek değerler
    axes[0, 1].scatter(y_true.flatten(), errors, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('🎯 Hata vs Gerçek Değerler', fontweight='bold')
    axes[0, 1].set_xlabel('Gerçek Değerler')
    axes[0, 1].set_ylabel('Hata')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mutlak hatalar
    abs_errors = np.abs(errors)
    axes[1, 0].plot(abs_errors, 'b-', alpha=0.7, linewidth=1)
    axes[1, 0].set_title('📈 Mutlak Hata Zaman Serisi', fontweight='bold')
    axes[1, 0].set_xlabel('Örnek İndeksi')
    axes[1, 0].set_ylabel('Mutlak Hata')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hata istatistikleri
    axes[1, 1].axis('off')
    stats_text = f"""
    📊 HATA İSTATİSTİKLERİ
    
    Ortalama Hata: {np.mean(errors):.4f}
    Standart Sapma: {np.std(errors):.4f}
    Min Hata: {np.min(errors):.4f}
    Max Hata: {np.max(errors):.4f}
    
    Mutlak Hata Ort.: {np.mean(abs_errors):.4f}
    Medyan Abs Hata: {np.median(abs_errors):.4f}
    
    % 95 Güven Aralığı:
    [{np.percentile(errors, 2.5):.4f}, {np.percentile(errors, 97.5):.4f}]
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

class EarlyStopping:
    """Basit early stopping implementasyonu"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def create_model_comparison_plot(models_results, title="Model Karşılaştırması"):
    """
    Birden fazla modelin sonuçlarını karşılaştırır
    
    Args:
        models_results: Dictionary of model results
        title: Grafik başlığı
    """
    model_names = list(models_results.keys())
    metrics = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R²']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    
    for i, metric in enumerate(metrics):
        values = []
        for model_name in model_names:
            if metric.lower().replace('²', '2') in models_results[model_name]:
                values.append(models_results[model_name][metric.lower().replace('²', '2')])
            else:
                values.append(0)
        
        bars = axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'📊 {metric}', fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # En iyi değeri vurgula
        if metric != 'MAPE':  # MAPE için en düşük daha iyi
            best_idx = np.argmax(values) if metric == 'R²' else np.argmin(values)
        else:
            best_idx = np.argmin(values)
        
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_model_summary(model, model_name="Model"):
    """Model özetini güzel formatta yazdırır"""
    print(f"\n🏗️ {model_name.upper()} ÖZETİ")
    print("="*50)
    
    total_params = model.count_params() if hasattr(model, 'count_params') else 0
    trainable_params = total_params
    
    print(f"📊 Parametre Bilgileri:")
    print(f"   Toplam parametre: {total_params:,}")
    print(f"   Eğitilebilir parametre: {trainable_params:,}")
    
    if hasattr(model, 'layers'):
        print(f"   Katman sayısı: {len(model.layers)}")
        print(f"\n🏗️ Katman Detayları:")
        for i, layer in enumerate(model.layers):
            layer_params = layer.count_params() if hasattr(layer, 'count_params') else 0
            print(f"   {i+1}. {layer.__class__.__name__}: {layer_params:,} parametre")

# Test fonksiyonu
if __name__ == "__main__":
    print("🛠️ RNN Yardımcı Fonksiyonları Test Ediliyor...")
    
    # Örnek zaman serisi verisi oluştur
    data = create_time_series_data(length=100, complexity='medium')
    print(f"✅ {len(data)} uzunlukta zaman serisi verisi oluşturuldu")
    
    # Sequence'ler oluştur
    X, y = create_sequences(data, window_size=10)
    print(f"✅ {len(X)} sequence oluşturuldu")
    
    print("🎉 Tüm fonksiyonlar başarıyla test edildi!")