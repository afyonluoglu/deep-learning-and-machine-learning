"""
📊 STOCK PRICE PREDICTION - HİSSE SENEDİ FİYAT TAHMİNİ
====================================================

Bu dosya RNN ile hisse senedi fiyat tahmininin teorisi ve uygulamasını açıklar.
Finansal time series analysis ve technical indicators kullanımı.

Stock Prediction Challenges:
- High volatility ve noise
- Market sentiment effects
- External factors
- Non-stationary nature
- Complex dependencies

Kullanım Alanları:
- Portfolio management
- Risk assessment  
- Algorithmic trading
- Market analysis
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 STOCK PRICE PREDICTION - HİSSE SENEDİ FİYAT TAHMİNİ")
print("=" * 70)

def print_section(title, char="=", width=50):
    print(f"\n{char*width}")
    print(f"📋 {title}")
    print(f"{char*width}")

print_section("STOCK PREDICTION TEORİSİ")

print("📈 Stock Price Prediction Nedir?")
print("-" * 40)
print("• Geçmiş fiyat hareketlerinden gelecek fiyatları tahmin etme")
print("• Technical analysis + Machine Learning kombinasyonu") 
print("• High-frequency data ve complex patterns")
print("• Risk management ve portfolio optimization")

# Simple synthetic stock data generation
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
returns = np.random.normal(0.001, 0.02, 1000)
prices = [100]
for return_rate in returns[1:]:
    prices.append(prices[-1] * (1 + return_rate))

stock_data = pd.DataFrame({
    'Date': dates,
    'Close': prices
})

print_section("STOCK DATA ANALİZİ")

print("📊 Stock data oluşturuldu...")
print(f"Data shape: {stock_data.shape}")
print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")

# Add technical indicators
def calculate_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    return df

stock_data = calculate_technical_indicators(stock_data)
stock_data = stock_data.dropna()

print(f"Data with indicators: {len(stock_data)} days")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('📊 Stock Analysis', fontsize=16, fontweight='bold')

axes[0, 0].plot(stock_data['Date'], stock_data['Close'], 'b-', alpha=0.8)
axes[0, 0].plot(stock_data['Date'], stock_data['SMA_20'], 'r--', alpha=0.7, label='SMA 20')
axes[0, 0].set_title('📈 Price with SMA', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

daily_returns = stock_data['Close'].pct_change().dropna()
axes[0, 1].hist(daily_returns, bins=30, alpha=0.7, color='green')
axes[0, 1].set_title('📊 Daily Returns', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(stock_data['Date'], stock_data['RSI'], 'purple')
axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7)
axes[1, 0].set_title('📊 RSI', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(stock_data['Date'], stock_data['MACD'], 'orange')
axes[1, 1].set_title('📈 MACD', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_section("RNN MODEL PREPARATION")

def prepare_stock_data(df, sequence_length=60):
    features = ['Close', 'SMA_20', 'RSI', 'MACD']
    df_features = df[features].dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

sequence_length = 60
X, y, scaler = prepare_stock_data(stock_data, sequence_length)

print(f"📊 Data prepared for RNN:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"  Train size: {len(X_train)}")
print(f"  Test size: {len(X_test)}")

print_section("RNN MODEL BUILDING")

models = {}

simple_model = Sequential([
    LSTM(50, input_shape=(sequence_length, 4)),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

deep_model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, 4)),
    Dropout(0.3),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

bi_model = Sequential([
    Bidirectional(LSTM(50), input_shape=(sequence_length, 4)),
    Dropout(0.3),
    Dense(25),
    Dense(1)
])

models['Simple LSTM'] = simple_model
models['Deep LSTM'] = deep_model
models['Bidirectional LSTM'] = bi_model

for name, model in models.items():
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    print(f"{name}: {model.count_params():,} parameters")

print_section("MODEL TRAINING")

results = {}
print("🚀 Training models...")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    test_pred = model.predict(X_test, verbose=0)
    
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    results[name] = {
        'mse': test_mse,
        'mae': test_mae,
        'predictions': test_pred
    }
    
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")

print_section("RESULTS ANALYSIS")

best_model = min(results.keys(), key=lambda x: results[x]['mse'])
print(f"🏆 Best Model: {best_model}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('📊 Stock Prediction Results', fontsize=16, fontweight='bold')

model_names = list(results.keys())
mse_values = [results[name]['mse'] for name in model_names]

bars = axes[0, 0].bar(model_names, mse_values, color=['blue', 'green', 'red'], alpha=0.7)
axes[0, 0].set_title('📉 Model MSE Comparison', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

best_pred = results[best_model]['predictions']
axes[0, 1].scatter(y_test, best_pred, alpha=0.6)
min_val, max_val = y_test.min(), y_test.max()
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[0, 1].set_title(f'🎯 {best_model}: Actual vs Predicted', fontweight='bold')
axes[0, 1].set_xlabel('Actual')
axes[0, 1].set_ylabel('Predicted')
axes[0, 1].grid(True, alpha=0.3)

n_show = min(100, len(y_test))
axes[1, 0].plot(range(n_show), y_test[:n_show], 'b-', label='Actual', alpha=0.8)
axes[1, 0].plot(range(n_show), best_pred[:n_show], 'r-', label='Predicted', alpha=0.8)
axes[1, 0].set_title(f'📈 First {n_show} Test Predictions', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

errors = y_test - best_pred.flatten()
axes[1, 1].hist(errors, bins=25, alpha=0.7, color='orange')
axes[1, 1].axvline(x=0, color='red', linestyle='--')
axes[1, 1].set_title('📊 Prediction Errors', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print_section("ÖZET VE SONUÇLAR")

print("✅ Bu Stock Price Prediction modülünde öğrendikleriniz:")
print("  1. 📊 Stock data analysis ve technical indicators")
print("  2. 🏗️ RNN model architectures for financial data") 
print("  3. 📈 Model comparison ve evaluation")
print("  4. 🎯 Prediction accuracy assessment")

print(f"\n🏆 MODEL PERFORMANCE:")
for name, result in results.items():
    print(f"   {name:20s}: MSE={result['mse']:.6f}")

print("\n💡 Ana çıkarımlar:")
print("  • Stock prediction çok challenging (high noise)")
print("  • Technical indicators help ama tek başına yeterli değil")
print("  • Model selection önemli (complexity vs overfitting)")

print("\n⚠️ ÖNEMLİ UYARILAR:")
print("  • Bu sadece eğitim amaçlı örnek")
print("  • Gerçek trading için kullanmayın")
print("  • Finansal risk içerir")

print("\n📚 Sonraki modül: 11_bidirectional_rnn.py")

print("\n" + "=" * 70)
print("✅ STOCK PRICE PREDICTION MODÜLÜ TAMAMLANDI!")
print("=" * 70)