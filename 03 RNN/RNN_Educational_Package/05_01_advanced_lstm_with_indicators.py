"""
Gelişmiş LSTM Modeli - Teknik İndikatörler ile
Bu dosya LSTM modeline teknik indikatörler ekleyerek performansı artırır
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Teknik indikatör hesaplama sınıfı"""
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average - Basit Hareketli Ortalama"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average - Üstel Hareketli Ortalama"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index - Göreceli Güç Endeksi"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bantları"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
        """Stokastik Osilatör"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

def create_advanced_financial_data(n_samples=1000):
    """Gelişmiş finansal veri simülasyonu"""
    np.random.seed(42)
    
    # Trend bileşenleri
    trend = np.cumsum(np.random.randn(n_samples) * 0.01)
    seasonality = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 252)  # Yıllık döngü
    noise = np.random.randn(n_samples) * 0.02
    
    # Fiyat serisi
    prices = 100 + trend + seasonality + noise
    
    # OHLC verisi oluştur
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    open_prices = prices + np.random.randn(n_samples) * 0.005
    volume = np.random.randint(1000, 10000, n_samples)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })

def add_technical_indicators(df):
    """Teknik indikatörleri veri setine ekle"""
    indicators = TechnicalIndicators()
    
    # Temel indikatörler
    df['sma_10'] = indicators.sma(df['close'], 10)
    df['sma_30'] = indicators.sma(df['close'], 30)
    df['ema_12'] = indicators.ema(df['close'], 12)
    df['ema_26'] = indicators.ema(df['close'], 26)
    
    # RSI
    df['rsi'] = indicators.rsi(df['close'])
    
    # MACD
    macd_line, signal_line, histogram = indicators.macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    # Bollinger Bantları
    bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Stokastik Osilatör
    k_percent, d_percent = indicators.stochastic_oscillator(
        df['high'], df['low'], df['close']
    )
    df['stoch_k'] = k_percent
    df['stoch_d'] = d_percent
    
    # Volume indikatörleri
    df['volume_sma'] = indicators.sma(df['volume'], 20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Fiyat değişim oranları
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['volatility'] = df['price_change'].rolling(20).std()
    
    return df

def prepare_lstm_data(df, sequence_length=60, target_col='close'):
    """LSTM için veri hazırlığı - teknik indikatörler dahil"""
    # NaN değerleri temizle
    df = df.dropna()
    
    # Hedef değişken
    target = df[target_col].values
    
    # Özellik seçimi - sadece sayısal değerler
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_30', 'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'stoch_k', 'stoch_d', 'volume_ratio',
        'price_change', 'price_change_5', 'volatility'
    ]
    
    features = df[feature_columns].values
    
    # Normalizasyon
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Sequence oluştur
    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(target_scaled[i])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, feature_scaler, target_scaler, feature_columns

def create_advanced_lstm_model(input_shape):
    """Gelişmiş LSTM modeli oluştur"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    """Ana fonksiyon"""
    print("🚀 Gelişmiş LSTM Modeli - Teknik İndikatörler ile")
    print("=" * 50)
    
    # 1. Veri oluştur
    print("\n📊 Finansal veri oluşturuluyor...")
    df = create_advanced_financial_data(1200)
    
    # 2. Teknik indikatörleri ekle
    print("📈 Teknik indikatörler hesaplanıyor...")
    df = add_technical_indicators(df)
    
    print(f"📋 Toplam özellik sayısı: {len(df.columns)}")
    print("✅ Eklenen indikatörler:")
    technical_indicators = [
        'SMA (10, 30)', 'EMA (12, 26)', 'RSI', 
        'MACD', 'Bollinger Bands', 'Stochastic', 
        'Volume Indicators', 'Volatility'
    ]
    for indicator in technical_indicators:
        print(f"   • {indicator}")
    
    # 3. LSTM verisi hazırla
    print("\n🔧 LSTM için veri hazırlanıyor...")
    X, y, feature_scaler, target_scaler, feature_columns = prepare_lstm_data(df)
    
    print(f"📊 Veri boyutları: X={X.shape}, y={y.shape}")
    print(f"🎯 Özellik sayısı: {X.shape[2]}")
    
    # 4. Train/Test ayır
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 5. Model oluştur ve eğit
    print("\n🤖 LSTM modeli oluşturuluyor...")
    model = create_advanced_lstm_model((X.shape[1], X.shape[2]))
    
    print("📚 Model eğitiliyor...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 6. Tahminler
    print("\n🔮 Tahminler yapılıyor...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Inverse transform
    train_pred = target_scaler.inverse_transform(train_pred)
    test_pred = target_scaler.inverse_transform(test_pred)
    y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 7. Performans metrikleri
    train_mse = mean_squared_error(y_train_actual, train_pred)
    test_mse = mean_squared_error(y_test_actual, test_pred)
    train_mae = mean_absolute_error(y_train_actual, train_pred)
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    
    print("\n📊 PERFORMANS METRİKLERİ")
    print("=" * 30)
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE:  {test_mse:.6f}")
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Test MAE:  {test_mae:.6f}")
    
    # MAPE hesapla
    train_mape = np.mean(np.abs((y_train_actual - train_pred) / y_train_actual)) * 100
    test_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100
    print(f"Train MAPE: {train_mape:.2f}%")
    print(f"Test MAPE:  {test_mape:.2f}%")
    print(f"Model Doğruluğu: {100 - test_mape:.2f}%")
    
    # 8. Özellik önem analizi
    print("\n🔍 ÖZELLİK ÖNEMİ ANALİZİ")
    print("=" * 30)
    feature_importance = []
    
    # Basit özellik önem analizi (correlation ile)
    df_clean = df.dropna()
    correlations = df_clean[feature_columns].corrwith(df_clean['close']).abs().sort_values(ascending=False)
    
    print("En önemli özellikler (korelasyon bazlı):")
    for i, (feature, corr) in enumerate(correlations.head(10).items()):
        print(f"{i+1:2d}. {feature:15s}: {corr:.4f}")
    
    # 9. Grafikler
    plt.figure(figsize=(20, 15))
    
    # Loss grafiği
    plt.subplot(3, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss - Teknik İndikatörler ile')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Tahmin vs Gerçek (Test)
    plt.subplot(3, 3, 2)
    plt.scatter(y_test_actual, test_pred, alpha=0.6)
    plt.plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Değerleri')
    plt.title('Tahmin vs Gerçek (Test Set)')
    plt.grid(True)
    
    # Zaman serisi karşılaştırması
    plt.subplot(3, 3, 3)
    test_indices = range(len(y_test_actual))
    plt.plot(test_indices[:100], y_test_actual[:100].flatten(), label='Gerçek', linewidth=2)
    plt.plot(test_indices[:100], test_pred[:100].flatten(), label='Tahmin', linewidth=2)
    plt.title('Zaman Serisi Karşılaştırması (İlk 100 Test)')
    plt.xlabel('Zaman')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    
    # Hata dağılımı
    plt.subplot(3, 3, 4)
    errors = y_test_actual.flatten() - test_pred.flatten()
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Hata Dağılımı')
    plt.xlabel('Hata')
    plt.ylabel('Frekans')
    plt.grid(True)
    
    # Özellik korelasyonları
    plt.subplot(3, 3, 5)
    top_features = correlations.head(8)
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.title('En Önemli Özellikler')
    plt.xlabel('Korelasyon')
    plt.grid(True)
    
    # RSI grafiği
    plt.subplot(3, 3, 6)
    plt.plot(df_clean['close'].iloc[-200:].values, label='Fiyat')
    plt.title('Son 200 Günlük Fiyat Hareketi')
    plt.xlabel('Gün')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    
    # RSI ve Stochastic
    plt.subplot(3, 3, 7)
    plt.plot(df_clean['rsi'].iloc[-200:].values, label='RSI', alpha=0.8)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Aşırı Alım')
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Aşırı Satım')
    plt.title('RSI İndikatörü')
    plt.xlabel('Gün')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # MACD
    plt.subplot(3, 3, 8)
    plt.plot(df_clean['macd'].iloc[-200:].values, label='MACD', linewidth=2)
    plt.plot(df_clean['macd_signal'].iloc[-200:].values, label='Signal', linewidth=2)
    plt.fill_between(range(200), df_clean['macd_histogram'].iloc[-200:].values, 
                     alpha=0.3, label='Histogram')
    plt.title('MACD İndikatörü')
    plt.xlabel('Gün')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    
    # Bollinger Bands
    plt.subplot(3, 3, 9)
    indices = range(200)
    plt.plot(indices, df_clean['close'].iloc[-200:].values, label='Fiyat', linewidth=2)
    plt.plot(indices, df_clean['bb_upper'].iloc[-200:].values, 'r--', alpha=0.7, label='Üst Band')
    plt.plot(indices, df_clean['bb_lower'].iloc[-200:].values, 'g--', alpha=0.7, label='Alt Band')
    plt.plot(indices, df_clean['bb_middle'].iloc[-200:].values, 'b--', alpha=0.7, label='Orta Band')
    plt.fill_between(indices, 
                     df_clean['bb_upper'].iloc[-200:].values, 
                     df_clean['bb_lower'].iloc[-200:].values, 
                     alpha=0.1)
    plt.title('Bollinger Bantları')
    plt.xlabel('Gün')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 10. Model kaydet
    print("\n💾 Model kaydediliyor...")
    model.save('advanced_lstm_with_indicators.keras')
    
    print("\n✅ Gelişmiş LSTM modeli başarıyla oluşturuldu!")
    print(f"🎯 Final Model Doğruluğu: {100 - test_mape:.2f}%")
    print(f"📊 Kullanılan özellik sayısı: {len(feature_columns)}")

if __name__ == "__main__":
    main()