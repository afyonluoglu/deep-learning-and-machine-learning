"""
📈 ZAMAN SERİSİ TAHMİNİ - RNN İLE GELIŞMIŞ ZAMAN SERİSİ ANALİZİ
==============================================================

Bu dosya RNN ile kapsamlı zaman serisi tahmini yapmayı öğretir.
Gerçek dünya veri setleri, çok adımlı tahminler ve gelişmiş teknikler.

Zaman Serisi Tahmini Nedir?
- Geçmiş değerleri kullanarak gelecekteki değerleri tahmin etme
- Trend, mevsimsellik ve rastgele bileşenlerin analizi
- Univariate ve multivariate time series
- Short-term ve long-term forecasting

RNN Avantajları:
- Sequential pattern recognition
- Variable length sequences
- Memory of historical context
- Non-linear relationship modeling
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, 
                                   Bidirectional, Input, 
                                   Conv1D, MaxPooling1D,
                                   MultiHeadAttention, LayerNormalization,
                                   GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📈 ZAMAN SERİSİ TAHMİNİ - RNN İLE GELIŞMIŞ ZAMAN SERİSİ ANALİZİ")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("KARMAŞIK ZAMAN SERİSİ VERİ SETİ OLUŞTURMA")

def create_complex_time_series(n_points=5000, noise_level=0.1):
    """Gerçekçi karmaşık zaman serisi oluşturur"""
    
    print("📊 Karmaşık zaman serisi veri seti oluşturuluyor...")
    
    np.random.seed(42)
    time = np.arange(n_points)
    dt = 1.0  # Daily data
    
    # Base components
    base_value = 100
    
    # 1. Long-term trend (polynomial + exponential)
    trend = (base_value + 0.002 * time + 
             0.0000001 * time**2 - 
             0.000000000001 * time**3)
    
    # 2. Multiple seasonal patterns
    seasonal_yearly = 15 * np.sin(2 * np.pi * time / 365.25)  # Annual
    seasonal_monthly = 8 * np.sin(2 * np.pi * time / 30.44)   # Monthly  
    seasonal_weekly = 4 * np.sin(2 * np.pi * time / 7)        # Weekly
    seasonal_daily = 2 * np.sin(2 * np.pi * time / 1)         # Daily (if hourly data)
    
    # 3. Cyclical patterns (business cycles)
    cycle_1 = 12 * np.sin(2 * np.pi * time / (7 * 365.25))    # 7-year cycle
    cycle_2 = 6 * np.sin(2 * np.pi * time / (3.5 * 365.25))   # 3.5-year cycle
    
    # 4. ARCH/GARCH-like volatility clustering
    volatility = np.zeros(n_points)
    volatility[0] = 1.0
    alpha = 0.1  # ARCH parameter
    beta = 0.85  # GARCH parameter
    
    for i in range(1, n_points):
        volatility[i] = (0.05 + 
                        alpha * (np.random.randn()**2) + 
                        beta * volatility[i-1])
    
    # 5. Heteroscedastic noise
    noise = np.random.randn(n_points) * np.sqrt(volatility) * noise_level * base_value
    
    # 6. Structural breaks
    structural_breaks = [n_points//4, n_points//2, 3*n_points//4]
    break_effects = [0, 10, -15, 8]  # Level shifts
    
    break_component = np.zeros(n_points)
    for i, break_point in enumerate(structural_breaks):
        break_component[break_point:] += break_effects[i]
    
    # 7. Outliers (rare events)
    outlier_prob = 0.005
    outliers = np.random.binomial(1, outlier_prob, n_points)
    outlier_magnitude = np.random.normal(0, 3*base_value*noise_level, n_points)
    outlier_component = outliers * outlier_magnitude
    
    # 8. Regime switching (changing parameters)
    regime_1 = np.sin(2 * np.pi * time / 1000) > 0
    regime_2 = ~regime_1
    regime_effect = regime_1 * 5 - regime_2 * 3
    
    # Combine all components
    ts_data = (trend + 
               seasonal_yearly + seasonal_monthly + seasonal_weekly +
               cycle_1 + cycle_2 +
               break_component +
               outlier_component +
               regime_effect +
               noise)
    
    # Create DataFrame with additional features
    df = pd.DataFrame({
        'timestamp': pd.date_range('2010-01-01', periods=n_points, freq='D'),
        'value': ts_data,
        'trend': trend,
        'seasonal_yearly': seasonal_yearly,
        'seasonal_monthly': seasonal_monthly,
        'seasonal_weekly': seasonal_weekly,
        'cycles': cycle_1 + cycle_2,
        'volatility': volatility,
        'regime': regime_1.astype(int)
    })
    
    # Additional derived features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Moving averages
    df['ma_7'] = df['value'].rolling(window=7).mean()
    df['ma_30'] = df['value'].rolling(window=30).mean()
    df['ma_365'] = df['value'].rolling(window=365).mean()
    
    # Returns and differences
    df['returns'] = df['value'].pct_change()
    df['diff_1'] = df['value'].diff()
    df['diff_7'] = df['value'].diff(7)
    df['diff_30'] = df['value'].diff(30)
    
    # Volatility measures
    df['rolling_std_7'] = df['returns'].rolling(window=7).std()
    df['rolling_std_30'] = df['returns'].rolling(window=30).std()
    
    print(f"✅ Zaman serisi oluşturuldu:")
    print(f"   Zaman aralığı: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"   Toplam nokta: {len(df)}")
    print(f"   Ortalama değer: {df['value'].mean():.2f}")
    print(f"   Standart sapma: {df['value'].std():.2f}")
    print(f"   Min-Max: {df['value'].min():.2f} - {df['value'].max():.2f}")
    
    return df

# Veri seti oluştur
df = create_complex_time_series(n_points=3000)

print_section("EKSPLORATİF VERİ ANALİZİ")

def exploratory_time_series_analysis(df):
    """Kapsamlı zaman serisi EDA"""
    
    print("🔍 Exploratory Data Analysis başlıyor...")
    
    # Temel istatistikler
    print("\n📊 Temel İstatistikler:")
    print(df['value'].describe())
    
    # Visualizations
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle('Zaman Serisi Exploratory Analysis', fontsize=16, fontweight='bold')
    
    # 1. Ana zaman serisi
    axes[0, 0].plot(df['timestamp'], df['value'], 'b-', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title('Orijinal Zaman Serisi', fontweight='bold')
    axes[0, 0].set_xlabel('Tarih')
    axes[0, 0].set_ylabel('Değer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Decomposition
    axes[0, 1].plot(df['timestamp'], df['trend'], 'r-', label='Trend', linewidth=2)
    axes[0, 1].plot(df['timestamp'], df['seasonal_yearly'], 'g-', label='Yıllık', alpha=0.7)
    axes[0, 1].plot(df['timestamp'], df['cycles'], 'm-', label='Cycles', alpha=0.7)
    axes[0, 1].set_title('Bileşen Ayrıştırma', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Volatility
    axes[0, 2].plot(df['timestamp'], df['volatility'], 'orange', alpha=0.8)
    axes[0, 2].set_title('Volatilite Evrimi', fontweight='bold')
    axes[0, 2].set_xlabel('Tarih')
    axes[0, 2].set_ylabel('Volatilite')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribution
    axes[1, 0].hist(df['value'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(df['value'].mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].axvline(df['value'].median(), color='green', linestyle='--', label='Median')
    axes[1, 0].set_title('Değer Dağılımı', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Returns distribution
    axes[1, 1].hist(df['returns'].dropna(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Returns Dağılımı', fontweight='bold')
    axes[1, 1].set_xlabel('Returns')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Autocorrelation (approximate)
    lags = range(1, 101)
    autocorr = []
    for lag in lags:
        if lag < len(df):
            values = df['value'].dropna()
            if len(values) > lag:
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorr.append(corr)
            else:
                autocorr.append(0)
        else:
            autocorr.append(0)
    
    axes[1, 2].plot(lags, autocorr, 'b-o', markersize=2)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 2].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('Autocorrelation Function', fontweight='bold')
    axes[1, 2].set_xlabel('Lag')
    axes[1, 2].set_ylabel('ACF')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Seasonality analysis
    monthly_avg = df.groupby('month')['value'].mean()
    axes[2, 0].bar(monthly_avg.index, monthly_avg.values, alpha=0.7, color='orange')
    axes[2, 0].set_title('Aylık Ortalamalar', fontweight='bold')
    axes[2, 0].set_xlabel('Ay')
    axes[2, 0].set_ylabel('Ortalama Değer')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Weekly patterns
    weekly_avg = df.groupby('day_of_week')['value'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[2, 1].bar(range(7), weekly_avg.values, alpha=0.7, color='green')
    axes[2, 1].set_title('Haftalık Ortalamalar', fontweight='bold')
    axes[2, 1].set_xlabel('Gün')
    axes[2, 1].set_ylabel('Ortalama Değer')
    axes[2, 1].set_xticks(range(7))
    axes[2, 1].set_xticklabels(day_names)
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Moving averages
    axes[2, 2].plot(df['timestamp'], df['value'], 'b-', alpha=0.3, label='Original')
    axes[2, 2].plot(df['timestamp'], df['ma_7'], 'r-', label='MA-7', linewidth=2)
    axes[2, 2].plot(df['timestamp'], df['ma_30'], 'g-', label='MA-30', linewidth=2)
    axes[2, 2].plot(df['timestamp'], df['ma_365'], 'orange', label='MA-365', linewidth=2)
    axes[2, 2].set_title('Moving Averages', fontweight='bold')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # 10. Volatility clustering
    axes[3, 0].plot(df['timestamp'], df['rolling_std_30'], 'purple', alpha=0.8)
    axes[3, 0].set_title('30-Day Rolling Volatility', fontweight='bold')
    axes[3, 0].set_xlabel('Tarih')
    axes[3, 0].set_ylabel('Rolling Std')
    axes[3, 0].grid(True, alpha=0.3)
    
    # 11. Regime analysis
    regime_colors = df['regime'].map({0: 'red', 1: 'blue'})
    scatter = axes[3, 1].scatter(df['timestamp'], df['value'], 
                                c=regime_colors, alpha=0.6, s=1)
    axes[3, 1].set_title('Regime Switching', fontweight='bold')
    axes[3, 1].set_xlabel('Tarih')
    axes[3, 1].set_ylabel('Değer')
    axes[3, 1].grid(True, alpha=0.3)
    
    # 12. Fourier Analysis (approximation)
    sample_data = df['value'].dropna().values[:1024]  # Power of 2 for FFT
    fft_vals = fft(sample_data)
    fft_freq = fftfreq(len(sample_data), d=1.0)
    
    # Only positive frequencies
    pos_freq_idx = fft_freq > 0
    axes[3, 2].plot(fft_freq[pos_freq_idx], np.abs(fft_vals[pos_freq_idx]), 'b-')
    axes[3, 2].set_title('Frequency Domain Analysis', fontweight='bold')
    axes[3, 2].set_xlabel('Frequency')
    axes[3, 2].set_ylabel('Magnitude')
    axes[3, 2].set_xlim(0, 0.1)  # Focus on low frequencies
    axes[3, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests (simplified)
    print("\n📊 İstatistiksel Özellikler:")
    
    # Stationarity check (simplified)
    diff_data = df['value'].diff().dropna()
    print(f"   Differenced mean: {diff_data.mean():.6f}")
    print(f"   Differenced std: {diff_data.std():.6f}")
    
    # Seasonality strength
    seasonal_strength = df['seasonal_yearly'].std() / df['value'].std()
    print(f"   Seasonal strength: {seasonal_strength:.4f}")
    
    # Trend strength
    trend_strength = df['trend'].std() / df['value'].std()
    print(f"   Trend strength: {trend_strength:.4f}")
    
    # Volatility clustering (simplified Ljung-Box test)
    returns_sq = (df['returns'].dropna() ** 2)
    ljung_box_approx = returns_sq.autocorr(lag=1)
    print(f"   Volatility clustering (approx): {ljung_box_approx:.4f}")
    
    return df

# EDA gerçekleştir
df = exploratory_time_series_analysis(df)

print_section("FEATURE ENGİNEERİNG VE VERİ HAZIRLAMA")

def advanced_feature_engineering(df):
    """Gelişmiş feature engineering"""
    
    print("🔧 Gelişmiş feature engineering...")
    
    df_features = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 30, 90, 365]:
        df_features[f'lag_{lag}'] = df_features['value'].shift(lag)
    
    # Rolling statistics
    windows = [7, 14, 30, 60, 90]
    for window in windows:
        df_features[f'rolling_mean_{window}'] = df_features['value'].rolling(window).mean()
        df_features[f'rolling_std_{window}'] = df_features['value'].rolling(window).std()
        df_features[f'rolling_min_{window}'] = df_features['value'].rolling(window).min()
        df_features[f'rolling_max_{window}'] = df_features['value'].rolling(window).max()
        df_features[f'rolling_median_{window}'] = df_features['value'].rolling(window).median()
    
    # Exponential moving averages
    alphas = [0.1, 0.3, 0.5]
    for alpha in alphas:
        df_features[f'ema_alpha_{alpha}'] = df_features['value'].ewm(alpha=alpha).mean()
    
    # Technical indicators
    # RSI approximation
    delta = df_features['value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD approximation
    ema_12 = df_features['value'].ewm(span=12).mean()
    ema_26 = df_features['value'].ewm(span=26).mean()
    df_features['macd'] = ema_12 - ema_26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    bb_window = 20
    bb_std = 2
    df_features['bb_middle'] = df_features['value'].rolling(window=bb_window).mean()
    bb_rolling_std = df_features['value'].rolling(window=bb_window).std()
    df_features['bb_upper'] = df_features['bb_middle'] + (bb_rolling_std * bb_std)
    df_features['bb_lower'] = df_features['bb_middle'] - (bb_rolling_std * bb_std)
    df_features['bb_position'] = (df_features['value'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
    
    # Relative position features
    for window in [30, 60, 90]:
        rolling_min = df_features['value'].rolling(window).min()
        rolling_max = df_features['value'].rolling(window).max()
        df_features[f'relative_position_{window}'] = (df_features['value'] - rolling_min) / (rolling_max - rolling_min)
    
    # Cycle features (Fourier features for seasonality)
    for period in [7, 30.44, 91.33, 365.25]:  # weekly, monthly, quarterly, yearly
        df_features[f'sin_{period}'] = np.sin(2 * np.pi * df_features.index / period)
        df_features[f'cos_{period}'] = np.cos(2 * np.pi * df_features.index / period)
    
    # Time-based features (expanded)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['timestamp'].dt.hour / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['timestamp'].dt.hour / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_month'] / 31)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_month'] / 31)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    
    # Difference features
    for diff_period in [1, 7, 30, 365]:
        df_features[f'diff_{diff_period}'] = df_features['value'].diff(diff_period)
        df_features[f'pct_change_{diff_period}'] = df_features['value'].pct_change(diff_period)
    
    print(f"✅ Feature engineering tamamlandı:")
    print(f"   Toplam feature sayısı: {len(df_features.columns)}")
    print(f"   Orijinal features: {len(df.columns)}")
    print(f"   Yeni features: {len(df_features.columns) - len(df.columns)}")
    
    return df_features

# Feature engineering uygula
df_features = advanced_feature_engineering(df)

def prepare_multivariate_sequences(df, target_col, feature_cols, 
                                 lookback_window, forecast_horizon,
                                 train_ratio=0.7, val_ratio=0.15):
    """Multivariate sequence'ler oluştur"""
    
    print("📊 Multivariate sequence preparation...")
    
    # NaN değerleri temizle
    df_clean = df[feature_cols + [target_col]].dropna()
    
    print(f"   Temiz veri boyutu: {len(df_clean)}")
    
    # Scaling
    target_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()
    
    # Target scaling
    target_scaled = target_scaler.fit_transform(df_clean[[target_col]])
    
    # Feature scaling  
    features_scaled = feature_scaler.fit_transform(df_clean[feature_cols])
    
    # Sequence oluşturma
    def create_sequences(target_data, feature_data, lookback, horizon):
        X, y = [], []
        for i in range(lookback, len(target_data) - horizon + 1):
            # Features: lookback window
            X.append(feature_data[i-lookback:i])
            # Target: next horizon steps
            y.append(target_data[i:i+horizon])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(target_scaled, features_scaled, lookback_window, forecast_horizon)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Train/validation/test split
    n_samples = len(X)
    train_idx = int(n_samples * train_ratio)
    val_idx = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_idx]
    y_train = y[:train_idx]
    X_val = X[train_idx:val_idx]
    y_val = y[train_idx:val_idx]
    X_test = X[val_idx:]
    y_test = y[val_idx:]
    
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            target_scaler, feature_scaler, feature_cols)

# En iyi feature'ları seç (basit correlation-based selection)
target_col = 'value'

# Numeric columns only
numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target_col)  # Target'ı çıkar

# Correlation ile feature selection
correlations = df_features[numeric_cols + [target_col]].corr()[target_col].abs()
top_features = correlations.nlargest(21).index.tolist()  # Top 20 + target
top_features.remove(target_col)  # Target'ı tekrar çıkar

print(f"\n🎯 En iyi 20 feature:")
for i, feature in enumerate(top_features):
    corr_val = correlations[feature]
    print(f"   {i+1:2d}. {feature:25s} (corr: {corr_val:.4f})")

# Sequence'leri hazırla
LOOKBACK_WINDOW = 60
FORECAST_HORIZON = 10

(X_train, y_train, X_val, y_val, X_test, y_test,
 target_scaler, feature_scaler, feature_cols) = prepare_multivariate_sequences(
    df_features, target_col, top_features,
    LOOKBACK_WINDOW, FORECAST_HORIZON
)

print_section("GELIŞMIŞ RNN MODEL MİMARİLERİ")

def create_advanced_time_series_models(input_shape, output_steps):
    """Gelişmiş zaman serisi modelleri oluştur"""
    
    models = {}
    
    print("🏗️ Gelişmiş zaman serisi modelleri oluşturuluyor...")
    
    # 1. Vanilla LSTM
    vanilla_lstm = Sequential([
        LSTM(128, input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_steps)
    ], name='Vanilla_LSTM')
    
    # 2. Stacked LSTM
    stacked_lstm = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_steps)
    ], name='Stacked_LSTM')
    
    # 3. Bidirectional LSTM
    bidirectional_lstm = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_steps)
    ], name='Bidirectional_LSTM')
    
    # 4. GRU Model
    gru_model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_steps)
    ], name='Stacked_GRU')
    
    # 5. CNN-LSTM Hybrid
    cnn_lstm = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_steps)
    ], name='CNN_LSTM')
    
    # 6. Encoder-Decoder LSTM
    def create_encoder_decoder_lstm():
        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoder_lstm = LSTM(128, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(output_steps, 1))  # Teacher forcing için
        decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # Dense layer
        decoder_dense = Dense(1, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model
    
    # Basit encoder-decoder (teacher forcing olmadan)
    encoder_decoder = Sequential([
        LSTM(128, input_shape=input_shape),
        tf.keras.layers.RepeatVector(output_steps),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ], name='Encoder_Decoder')
    
    # 7. Attention-like mechanism (simplified)
    def create_attention_lstm():
        inputs = Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        
        # Attention mechanism (simplified)
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        attended = tf.keras.layers.multiply([lstm_out, attention])
        attended = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attended)
        
        # Output layers
        output = Dense(64, activation='relu')(attended)
        output = Dropout(0.2)(output)
        output = Dense(output_steps)(output)
        
        model = Model(inputs, output)
        return model
    
    attention_lstm = create_attention_lstm()
    attention_lstm._name = 'Attention_LSTM'
    
    models = {
        'Vanilla LSTM': vanilla_lstm,
        'Stacked LSTM': stacked_lstm,
        'Bidirectional LSTM': bidirectional_lstm,
        'Stacked GRU': gru_model,
        'CNN-LSTM': cnn_lstm,
        'Encoder-Decoder': encoder_decoder,
        'Attention LSTM': attention_lstm
    }
    
    # Compile models
    for name, model in models.items():
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        print(f"✅ {name}: {model.count_params():,} parameters")
    
    return models

# Modelleri oluştur
input_shape = (X_train.shape[1], X_train.shape[2])
models = create_advanced_time_series_models(input_shape, FORECAST_HORIZON)

print_section("MODEL EĞİTİMİ VE DEĞERLENDİRME")

def train_time_series_models(models, X_train, y_train, X_val, y_val):
    """Zaman serisi modellerini eğit"""
    
    print("🚀 Modeller eğitiliyor...")
    
    histories = {}
    
    for name, model in models.items():
        print(f"\n📊 {name} eğitiliyor...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=0.0001,
            verbose=0
        )
        
        # Eğitim
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        histories[name] = history
        
        # Validation sonuçları
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        epochs = len(history.history['loss'])
        
        print(f"   ✅ Best val_loss : {val_loss:.6f}")
        print(f"   📊 Best val_mae  : {val_mae:.6f}")
        print(f"   ⏱️ Epochs trained: {epochs}")
    
    return histories

# Modelleri eğit
histories = train_time_series_models(models, X_train, y_train, X_val, y_val)

print_section("TEST PERFORMANSI VE KARŞILAŞTIRMA")

def evaluate_models_on_test(models, X_test, y_test, target_scaler, histories):
    """Test setinde model performansını değerlendir"""
    
    print("🎯 Test seti değerlendirmesi...")
    
    results = {}
    
    for name, model in models.items():
        # Test predictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse scaling
        y_test_original = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_original = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        
        # Ensure shapes match for MAPE calculation
        if y_test_original.shape != y_pred_original.shape:
            # Make sure both have the same shape
            y_test_original = y_test_original.squeeze()
            y_pred_original = y_pred_original.squeeze()
        
        # Metrics
        mse = mean_squared_error(y_test_original.flatten(), y_pred_original.flatten())
        mae = mean_absolute_error(y_test_original.flatten(), y_pred_original.flatten())
        rmse = np.sqrt(mse)
        
        # R² score
        r2 = r2_score(y_test_original.flatten(), y_pred_original.flatten())
        
        # MAPE (Mean Absolute Percentage Error) - use flattened arrays to avoid broadcasting issues
        y_test_flat = y_test_original.flatten()
        y_pred_flat = y_pred_original.flatten()
        # Avoid division by zero
        mask = y_test_flat != 0
        mape = np.mean(np.abs((y_test_flat[mask] - y_pred_flat[mask]) / y_test_flat[mask])) * 100
        
        # Directional accuracy (için trend doğruluk oranı)
        if len(y_test_original.shape) > 1 and y_test_original.shape[1] > 1:
            # For multi-step predictions
            y_test_diff = np.diff(y_test_original, axis=1)
            y_pred_diff = np.diff(y_pred_original, axis=1)
        else:
            # For single-step predictions, use differences between consecutive samples
            y_test_flat = y_test_original.flatten()
            y_pred_flat = y_pred_original.flatten()
            y_test_diff = np.diff(y_test_flat)
            y_pred_diff = np.diff(y_pred_flat)
        
        directional_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
        
        results[name] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred_original,
            'parameters': model.count_params(),
            'training_epochs': len(histories[name].history['loss'])
        }
        
        print(f"\n🎯 {name}:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.4f}")
    
    return results, y_test_original

# Test değerlendirmesi
results, y_test_original = evaluate_models_on_test(models, X_test, y_test, target_scaler, histories)

# En iyi modeli bul
best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
print(f"\n🏆 EN İYİ MODEL: {best_model_name}")
print(f"   MAE: {results[best_model_name]['mae']:.4f}")
print(f"   RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"   R²: {results[best_model_name]['r2']:.4f}")

print_section("SONUÇLAR VE GÖRSELLEŞTİRME")

# Comprehensive results visualization
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
fig.suptitle('Time Series Forecasting - Comprehensive Results', fontsize=16, fontweight='bold')

# 1. Training curves - Loss
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
for i, (name, history) in enumerate(histories.items()):
    axes[0, 0].plot(history.history['loss'], color=colors[i], 
                   label=f'{name}', linewidth=2, alpha=0.7)
    axes[0, 0].plot(history.history['val_loss'], color=colors[i], 
                   linestyle='--', linewidth=2, alpha=0.7)
axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. Training curves - MAE
for i, (name, history) in enumerate(histories.items()):
    axes[0, 1].plot(history.history['mae'], color=colors[i], 
                   label=f'{name}', linewidth=2, alpha=0.7)
    axes[0, 1].plot(history.history['val_mae'], color=colors[i], 
                   linestyle='--', linewidth=2, alpha=0.7)
axes[0, 1].set_title('Training & Validation MAE', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].grid(True, alpha=0.3)

# 3. Test performance comparison
model_names = list(results.keys())
mae_values = [results[name]['mae'] for name in model_names]
rmse_values = [results[name]['rmse'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0, 2].bar(x - width/2, mae_values, width, 
                      label='MAE', alpha=0.7, color='blue')
bars2 = axes[0, 2].bar(x + width/2, rmse_values, width,
                      label='RMSE', alpha=0.7, color='red')

axes[0, 2].set_title('Test Performance Comparison', fontweight='bold')
axes[0, 2].set_xlabel('Model')
axes[0, 2].set_ylabel('Error')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. R² scores
r2_values = [results[name]['r2'] for name in model_names]
bars = axes[1, 0].bar(range(len(model_names)), r2_values,
                     color=colors[:len(model_names)], alpha=0.7)
axes[1, 0].set_title('R² Scores', fontweight='bold')
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('R²')
axes[1, 0].set_xticks(range(len(model_names)))
axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3)

# Add value labels
for bar, r2 in zip(bars, r2_values):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. MAPE values
mape_values = [results[name]['mape'] for name in model_names]
bars = axes[1, 1].bar(range(len(model_names)), mape_values,
                     color=colors[:len(model_names)], alpha=0.7)
axes[1, 1].set_title('MAPE Values', fontweight='bold')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].set_xticks(range(len(model_names)))
axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3)

# 6. Directional Accuracy
dir_acc_values = [results[name]['directional_accuracy'] for name in model_names]
bars = axes[1, 2].bar(range(len(model_names)), dir_acc_values,
                     color=colors[:len(model_names)], alpha=0.7)
axes[1, 2].set_title('Directional Accuracy', fontweight='bold')
axes[1, 2].set_xlabel('Model')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].set_xticks(range(len(model_names)))
axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_ylim(0, 1)

# 7. Parameter count vs Performance
param_counts = [results[name]['parameters'] for name in model_names]
axes[2, 0].scatter(param_counts, mae_values, 
                  s=150, c=colors[:len(model_names)], alpha=0.7)
for i, name in enumerate(model_names):
    axes[2, 0].annotate(name.replace(' ', '\n'), (param_counts[i], mae_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[2, 0].set_title('Parameters vs MAE', fontweight='bold')
axes[2, 0].set_xlabel('Parameter Count')
axes[2, 0].set_ylabel('MAE')
axes[2, 0].grid(True, alpha=0.3)

# 8. Training efficiency
training_epochs = [results[name]['training_epochs'] for name in model_names]
efficiency = [mae / epoch for mae, epoch in zip(mae_values, training_epochs)]
axes[2, 1].bar(range(len(model_names)), efficiency,
              color=colors[:len(model_names)], alpha=0.7)
axes[2, 1].set_title('Training Efficiency (MAE/Epoch)', fontweight='bold')
axes[2, 1].set_xlabel('Model')
axes[2, 1].set_ylabel('MAE per Epoch')
axes[2, 1].set_xticks(range(len(model_names)))
axes[2, 1].set_xticklabels(model_names, rotation=45, ha='right')
axes[2, 1].grid(True, alpha=0.3)

# 9. Prediction samples - Best model
best_predictions = results[best_model_name]['predictions']
n_samples = min(5, len(y_test_original))
sample_indices = np.random.choice(len(y_test_original), n_samples, replace=False)

for i, idx in enumerate(sample_indices):
    if i < 2:  # Only show first 2 samples
        ax = axes[2, 2] if i == 0 else axes[3, 0]
        
        steps = range(1, FORECAST_HORIZON + 1)
        actual = y_test_original[idx]
        predicted = best_predictions[idx]
        
        ax.plot(steps, actual, 'bo-', label='Actual', linewidth=2, markersize=6)
        ax.plot(steps, predicted, 'ro-', label='Predicted', linewidth=2, markersize=6)
        ax.fill_between(steps, actual - np.std(actual), actual + np.std(actual),
                       alpha=0.2, color='blue')
        
        ax.set_title(f'Sample {i+1}: {FORECAST_HORIZON}-Step Forecast', fontweight='bold')
        ax.set_xlabel('Forecast Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

# 10. Residual analysis
best_predictions_flat = results[best_model_name]['predictions'].flatten()
y_test_flat = y_test_original.flatten()
residuals = y_test_flat - best_predictions_flat

axes[3, 1].scatter(best_predictions_flat, residuals, alpha=0.5, s=1)
axes[3, 1].axhline(y=0, color='red', linestyle='--')
axes[3, 1].set_title(f'Residual Plot - {best_model_name}', fontweight='bold')
axes[3, 1].set_xlabel('Predicted Values')
axes[3, 1].set_ylabel('Residuals')
axes[3, 1].grid(True, alpha=0.3)

# 11. Residual distribution
axes[3, 2].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[3, 2].axvline(np.mean(residuals), color='red', linestyle='--', label='Mean')
axes[3, 2].axvline(0, color='green', linestyle='-', label='Zero')
axes[3, 2].set_title('Residual Distribution', fontweight='bold')
axes[3, 2].set_xlabel('Residual Value')
axes[3, 2].set_ylabel('Frequency')
axes[3, 2].legend()
axes[3, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu Time Series Forecasting modülünde öğrendikleriniz:")
print("  1. 📊 Karmaşık zaman serisi veri seti oluşturma")
print("  2. 🔍 Comprehensive exploratory data analysis")
print("  3. 🔧 Advanced feature engineering")
print("  4. 🏗️ Çok çeşitli RNN mimarileri")
print("  5. 📈 Multi-step ahead forecasting")
print("  6. 🎯 Comprehensive model evaluation")
print("  7. 📊 Residual analysis and diagnostics")
print("  8. ⚖️ Performance vs complexity trade-offs")

print(f"\n🏆 PERFORMANS ÖZETİ:")
for name, result in results.items():
    print(f"   {name:20s}: MAE={result['mae']:6.3f}, R²={result['r2']:5.3f}, MAPE={result['mape']:5.1f}%")

print(f"\n🏆 EN İYİ MODEL: {best_model_name}")
print(f"   MAE: {results[best_model_name]['mae']:.4f}")
print(f"   RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"   R²: {results[best_model_name]['r2']:.4f}")
print(f"   MAPE: {results[best_model_name]['mape']:.2f}%")
print(f"   Directional Accuracy: {results[best_model_name]['directional_accuracy']:.4f}")

print("\n💡 Ana çıkarımlar:")
print("  • Feature engineering zaman serisi performansını büyük ölçüde artırır")
print("  • Multivariate modeller genellikle univariate'den daha iyi")
print("  • Bidirectional ve attention modelleri güçlü performans gösterir")
print("  • Model karmaşıklığı her zaman daha iyi performans anlamına gelmez")
print("  • Regularization ve dropout overfitting'i önlemede kritik")

print("\n🚀 İyileştirme önerileri:")
print("  1. Ensemble methods (model averaging, voting)")
print("  2. Advanced attention mechanisms")
print("  3. Transformer-based architectures")
print("  4. External data integration (weather, holidays)")
print("  5. Online learning for concept drift")
print("  6. Probabilistic forecasting (uncertainty quantification)")

print("\n📚 Sonraki modül: 10_stock_price_prediction.py")
print("RNN ile hisse senedi fiyat tahmini öğreneceğiz!")

print("\n" + "=" * 70)
print("✅ ZAMAN SERİSİ TAHMİNİ MODÜLÜ TAMAMLANDI!")
print("=" * 70)