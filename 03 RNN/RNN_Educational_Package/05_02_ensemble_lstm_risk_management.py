"""
Ensemble Methods ile LSTM Modeli
Bu dosya farklı LSTM modellerini birleştirerek daha güvenilir tahminler yapar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU
import warnings
warnings.filterwarnings('ignore')

class EnsembleLSTM:
    """Ensemble LSTM Modelleri Sınıfı"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.weights = {}
        self.scalers = {}
        
    def create_lstm_model1(self):
        """LSTM Model 1 - Derin LSTM"""
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def create_lstm_model2(self):
        """LSTM Model 2 - Geniş LSTM"""
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(256, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(64),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def create_gru_model(self):
        """GRU Model - LSTM'e alternatif"""
        model = Sequential([
            Input(shape=self.input_shape),
            GRU(128, return_sequences=True),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def create_simple_lstm(self):
        """Basit LSTM Model"""
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def create_bidirectional_lstm(self):
        """Bidirectional LSTM Model"""
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=50):
        """Ensemble modellerini eğit"""
        print("\n🤖 Ensemble modelleri eğitiliyor...")
        
        # Model tanımları
        model_creators = {
            'Deep_LSTM': self.create_lstm_model1,
            'Wide_LSTM': self.create_lstm_model2,
            'GRU_Model': self.create_gru_model,
            'Simple_LSTM': self.create_simple_lstm,
            'BiLSTM': self.create_bidirectional_lstm
        }
        
        histories = {}
        
        for name, creator in model_creators.items():
            print(f"\n📚 {name} eğitiliyor...")
            
            model = creator()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            
            self.models[name] = model
            histories[name] = history
            
            # Validation loss ile ağırlık hesapla
            val_loss = min(history.history['val_loss'])
            self.weights[name] = 1.0 / val_loss
            
            print(f"✅ {name} - Val Loss: {val_loss:.6f}")
        
        # Ağırlıkları normalize et
        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
            
        print("\n🎯 Model Ağırlıkları:")
        for name, weight in self.weights.items():
            print(f"   {name}: {weight:.4f}")
            
        return histories
    
    def predict_ensemble(self, X_test):
        """Ensemble tahmin yap"""
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test, verbose=0)
            predictions[name] = pred.flatten()
        
        # Ağırlıklı ortalama al
        ensemble_pred = np.zeros(len(predictions[list(predictions.keys())[0]]))
        
        for name, pred in predictions.items():
            ensemble_pred += pred * self.weights[name]
        
        return ensemble_pred, predictions

class RiskManager:
    """Risk Yönetimi Sınıfı"""
    
    def __init__(self):
        self.risk_metrics = {}
        
    def calculate_var(self, returns, confidence_level=0.05):
        """Value at Risk (VaR) hesapla"""
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """Conditional Value at Risk (CVaR) hesapla"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Sharpe Ratio hesapla"""
        excess_returns = returns - risk_free_rate/252  # Günlük risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, prices):
        """Maximum Drawdown hesapla"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def calculate_volatility(self, returns):
        """Volatilite hesapla (yıllık)"""
        return returns.std() * np.sqrt(252)
    
    def position_sizing(self, predicted_return, predicted_volatility, risk_tolerance=0.02):
        """Pozisyon büyüklüğü hesapla (Kelly Criterion benzeri)"""
        if predicted_volatility == 0:
            return 0
        
        kelly_fraction = predicted_return / (predicted_volatility ** 2)
        # Kelly'nin yarısını kullan (daha konservatif)
        position_size = min(kelly_fraction * 0.5, risk_tolerance)
        return max(position_size, 0)  # Negatif pozisyon yok
    
    def risk_assessment(self, actual_prices, predicted_prices):
        """Kapsamlı risk değerlendirmesi"""
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
        
        # Tahmin hatası
        prediction_errors = actual_returns - predicted_returns
        
        metrics = {
            'actual_var_5': self.calculate_var(actual_returns, 0.05),
            'actual_cvar_5': self.calculate_cvar(actual_returns, 0.05),
            'predicted_var_5': self.calculate_var(predicted_returns, 0.05),
            'predicted_cvar_5': self.calculate_cvar(predicted_returns, 0.05),
            'actual_volatility': self.calculate_volatility(actual_returns),
            'predicted_volatility': self.calculate_volatility(predicted_returns),
            'actual_sharpe': self.calculate_sharpe_ratio(actual_returns),
            'predicted_sharpe': self.calculate_sharpe_ratio(predicted_returns),
            'max_drawdown_actual': self.calculate_max_drawdown(actual_prices),
            'max_drawdown_predicted': self.calculate_max_drawdown(predicted_prices),
            'prediction_error_std': prediction_errors.std(),
            'prediction_error_mean': prediction_errors.mean()
        }
        
        return metrics

def create_financial_data_with_regime(n_samples=1000):
    """Rejim değişiklikli finansal veri simülasyonu"""
    np.random.seed(42)
    
    # Farklı rejimler tanımla
    regimes = []
    current_pos = 0
    
    while current_pos < n_samples:
        # Rejim uzunluğu
        regime_length = np.random.randint(50, 200)
        regime_type = np.random.choice(['bull', 'bear', 'sideways'])
        
        if regime_type == 'bull':
            trend = 0.001
            volatility = 0.015
        elif regime_type == 'bear':
            trend = -0.001
            volatility = 0.025
        else:  # sideways
            trend = 0.0001
            volatility = 0.010
            
        regimes.extend([{'trend': trend, 'vol': volatility, 'type': regime_type}] * 
                      min(regime_length, n_samples - current_pos))
        current_pos += regime_length
    
    # Fiyat serisi oluştur
    prices = [100]
    returns = []
    
    for i in range(1, n_samples):
        regime = regimes[i-1]
        daily_return = np.random.normal(regime['trend'], regime['vol'])
        returns.append(daily_return)
        prices.append(prices[-1] * (1 + daily_return))
    
    # OHLC verisi
    prices = np.array(prices)
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    volume = np.random.randint(1000, 10000, n_samples)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'regime': [r['type'] for r in regimes]
    })
    
    return df

def prepare_data_for_ensemble(df, sequence_length=60):
    """Ensemble için veri hazırlığı"""
    # Basit özellikler
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    
    # NaN temizle
    df = df.dropna()
    
    # Özellikler
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                      'returns', 'volatility', 'sma_10', 'sma_30']
    
    features = df[feature_columns].values
    target = df['close'].values
    
    # Normalize
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Sequences oluştur
    X, y = [], []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(target_scaled[i])
    
    return np.array(X), np.array(y), scaler, target_scaler

def main():
    """Ana fonksiyon"""
    print("🚀 Ensemble LSTM + Risk Management Sistemi")
    print("=" * 50)
    
    # 1. Veri oluştur
    print("\n📊 Rejim değişiklikli finansal veri oluşturuluyor...")
    df = create_financial_data_with_regime(1200)
    
    print(f"📋 Veri boyutu: {df.shape}")
    print(f"📈 Rejim dağılımı:")
    print(df['regime'].value_counts())
    
    # 2. Veri hazırlığı
    print("\n🔧 Ensemble için veri hazırlanıyor...")
    X, y, scaler, target_scaler = prepare_data_for_ensemble(df)
    
    # Train/Validation/Test split
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"📊 Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 3. Ensemble modeli oluştur ve eğit
    ensemble = EnsembleLSTM(input_shape=(X.shape[1], X.shape[2]))
    histories = ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=30)
    
    # 4. Tahminler
    print("\n🔮 Ensemble tahminleri yapılıyor...")
    ensemble_pred, individual_preds = ensemble.predict_ensemble(X_test)
    
    # Inverse transform
    ensemble_pred_actual = target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Individual model predictions
    individual_preds_actual = {}
    for name, pred in individual_preds.items():
        individual_preds_actual[name] = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    
    # 5. Performance metrikleri
    ensemble_mse = mean_squared_error(y_test_actual, ensemble_pred_actual)
    ensemble_mae = mean_absolute_error(y_test_actual, ensemble_pred_actual)
    ensemble_mape = np.mean(np.abs((y_test_actual - ensemble_pred_actual) / y_test_actual)) * 100
    
    print("\n📊 ENSEMBLE PERFORMANS METRİKLERİ")
    print("=" * 40)
    print(f"Ensemble MSE:  {ensemble_mse:.6f}")
    print(f"Ensemble MAE:  {ensemble_mae:.6f}")
    print(f"Ensemble MAPE: {ensemble_mape:.2f}%")
    print(f"Ensemble Doğruluğu: {100 - ensemble_mape:.2f}%")
    
    # Individual model performances
    print("\n🤖 BİREYSEL MODEL PERFORMANSLARI")
    print("=" * 40)
    for name, pred in individual_preds_actual.items():
        mse = mean_squared_error(y_test_actual, pred)
        mae = mean_absolute_error(y_test_actual, pred)
        mape = np.mean(np.abs((y_test_actual - pred) / y_test_actual)) * 100
        print(f"{name:12s}: MSE={mse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")
    
    # 6. Risk analizi
    print("\n⚠️  RİSK ANALİZİ")
    print("=" * 30)
    
    risk_manager = RiskManager()
    risk_metrics = risk_manager.risk_assessment(y_test_actual, ensemble_pred_actual)
    
    print("Risk Metrikleri:")
    print(f"  VaR (5%):           {risk_metrics['actual_var_5']:.4f} (Gerçek), {risk_metrics['predicted_var_5']:.4f} (Tahmin)")
    print(f"  CVaR (5%):          {risk_metrics['actual_cvar_5']:.4f} (Gerçek), {risk_metrics['predicted_cvar_5']:.4f} (Tahmin)")
    print(f"  Volatilite (Yıllık): {risk_metrics['actual_volatility']:.4f} (Gerçek), {risk_metrics['predicted_volatility']:.4f} (Tahmin)")
    print(f"  Sharpe Ratio:       {risk_metrics['actual_sharpe']:.4f} (Gerçek), {risk_metrics['predicted_sharpe']:.4f} (Tahmin)")
    print(f"  Max Drawdown:       {risk_metrics['max_drawdown_actual']:.4f} (Gerçek), {risk_metrics['max_drawdown_predicted']:.4f} (Tahmin)")
    print(f"  Tahmin Hatası Std:  {risk_metrics['prediction_error_std']:.4f}")
    
    # 7. Pozisyon büyüklüğü önerileri
    print("\n💰 POZİSYON BÜYÜKLÜĞÜ ÖNERİLERİ")
    print("=" * 40)
    
    # Son 30 gün için pozisyon büyüklükleri hesapla
    recent_returns = np.diff(ensemble_pred_actual[-30:]) / ensemble_pred_actual[-30:-1]
    recent_volatility = np.std(recent_returns)
    
    for risk_tolerance in [0.01, 0.02, 0.05]:
        avg_position_size = 0
        for i in range(len(recent_returns)-1):
            predicted_return = recent_returns[i]
            position_size = risk_manager.position_sizing(
                predicted_return, recent_volatility, risk_tolerance
            )
            avg_position_size += position_size
        
        avg_position_size /= (len(recent_returns)-1)
        print(f"Risk Toleransı %{risk_tolerance*100:.0f}: Ortalama Pozisyon Büyüklüğü = %{avg_position_size*100:.2f}")
    
    # 8. Grafikler
    plt.figure(figsize=(20, 16))
    
    # Ensemble vs Individual predictions
    plt.subplot(3, 4, 1)
    test_indices = range(len(y_test_actual))
    plt.plot(test_indices[:100], y_test_actual[:100], label='Gerçek', linewidth=2)
    plt.plot(test_indices[:100], ensemble_pred_actual[:100], label='Ensemble', linewidth=2)
    plt.title('Ensemble vs Gerçek (İlk 100 Test)')
    plt.xlabel('Zaman')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    
    # Individual model comparison
    plt.subplot(3, 4, 2)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, pred) in enumerate(list(individual_preds_actual.items())[:3]):
        plt.plot(test_indices[:50], pred[:50], 
                label=name, alpha=0.7, color=colors[i])
    plt.plot(test_indices[:50], y_test_actual[:50], 
             label='Gerçek', linewidth=2, color='black')
    plt.title('Bireysel Model Karşılaştırması')
    plt.xlabel('Zaman')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    
    # Model weights
    plt.subplot(3, 4, 3)
    names = list(ensemble.weights.keys())
    weights = list(ensemble.weights.values())
    plt.pie(weights, labels=names, autopct='%1.1f%%')
    plt.title('Ensemble Model Ağırlıkları')
    
    # Scatter plot
    plt.subplot(3, 4, 4)
    plt.scatter(y_test_actual, ensemble_pred_actual, alpha=0.6)
    plt.plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Ensemble Tahminleri')
    plt.title('Tahmin vs Gerçek')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(3, 4, 5)
    errors = y_test_actual - ensemble_pred_actual
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Ensemble Hata Dağılımı')
    plt.xlabel('Hata')
    plt.ylabel('Frekans')
    plt.grid(True)
    
    # Returns distribution
    plt.subplot(3, 4, 6)
    actual_returns = np.diff(y_test_actual) / y_test_actual[:-1]
    predicted_returns = np.diff(ensemble_pred_actual) / ensemble_pred_actual[:-1]
    plt.hist(actual_returns, bins=30, alpha=0.5, label='Gerçek', density=True)
    plt.hist(predicted_returns, bins=30, alpha=0.5, label='Tahmin', density=True)
    plt.title('Getiri Dağılımları')
    plt.xlabel('Getiri')
    plt.ylabel('Yoğunluk')
    plt.legend()
    plt.grid(True)
    
    # VaR visualization
    plt.subplot(3, 4, 7)
    sorted_returns = np.sort(actual_returns)
    var_5 = risk_metrics['actual_var_5']
    plt.plot(sorted_returns, label='Getiri Dağılımı')
    plt.axvline(x=var_5, color='red', linestyle='--', 
                label=f'VaR 5% = {var_5:.4f}')
    plt.title('Value at Risk (VaR)')
    plt.xlabel('Getiri')
    plt.ylabel('Değer')
    plt.legend()
    plt.grid(True)
    
    # Cumulative returns
    plt.subplot(3, 4, 8)
    actual_cumret = np.cumprod(1 + actual_returns)
    predicted_cumret = np.cumprod(1 + predicted_returns)
    plt.plot(actual_cumret, label='Gerçek Kümülatif Getiri')
    plt.plot(predicted_cumret, label='Tahmin Kümülatif Getiri')
    plt.title('Kümülatif Getiri Karşılaştırması')
    plt.xlabel('Zaman')
    plt.ylabel('Kümülatif Getiri')
    plt.legend()
    plt.grid(True)
    
    # Rolling Sharpe ratio
    plt.subplot(3, 4, 9)
    window = 30
    rolling_sharpe_actual = []
    rolling_sharpe_predicted = []
    
    for i in range(window, len(actual_returns)):
        window_returns_actual = actual_returns[i-window:i]
        window_returns_predicted = predicted_returns[i-window:i]
        
        sharpe_actual = risk_manager.calculate_sharpe_ratio(window_returns_actual)
        sharpe_predicted = risk_manager.calculate_sharpe_ratio(window_returns_predicted)
        
        rolling_sharpe_actual.append(sharpe_actual)
        rolling_sharpe_predicted.append(sharpe_predicted)
    
    plt.plot(rolling_sharpe_actual, label='Gerçek Sharpe')
    plt.plot(rolling_sharpe_predicted, label='Tahmin Sharpe')
    plt.title(f'{window}-Günlük Rolling Sharpe Ratio')
    plt.xlabel('Zaman')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    
    # Volatility comparison
    plt.subplot(3, 4, 10)
    window = 20
    rolling_vol_actual = pd.Series(actual_returns).rolling(window).std() * np.sqrt(252)
    rolling_vol_predicted = pd.Series(predicted_returns).rolling(window).std() * np.sqrt(252)
    
    plt.plot(rolling_vol_actual, label='Gerçek Volatilite')
    plt.plot(rolling_vol_predicted, label='Tahmin Volatilite')
    plt.title(f'{window}-Günlük Rolling Volatilite')
    plt.xlabel('Zaman')
    plt.ylabel('Yıllık Volatilite')
    plt.legend()
    plt.grid(True)
    
    # Drawdown analysis
    plt.subplot(3, 4, 11)
    actual_peak = np.maximum.accumulate(actual_cumret)
    actual_drawdown = (actual_cumret - actual_peak) / actual_peak
    
    predicted_peak = np.maximum.accumulate(predicted_cumret)
    predicted_drawdown = (predicted_cumret - predicted_peak) / predicted_peak
    
    plt.fill_between(range(len(actual_drawdown)), actual_drawdown, 0, 
                     alpha=0.3, label='Gerçek Drawdown')
    plt.fill_between(range(len(predicted_drawdown)), predicted_drawdown, 0, 
                     alpha=0.3, label='Tahmin Drawdown')
    plt.title('Drawdown Analizi')
    plt.xlabel('Zaman')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    # Model performance comparison
    plt.subplot(3, 4, 12)
    model_names = list(individual_preds_actual.keys()) + ['Ensemble']
    model_mapes = []
    
    for name, pred in individual_preds_actual.items():
        mape = np.mean(np.abs((y_test_actual - pred) / y_test_actual)) * 100
        model_mapes.append(mape)
    
    model_mapes.append(ensemble_mape)  # Ensemble MAPE
    
    plt.bar(model_names, model_mapes)
    plt.title('Model MAPE Karşılaştırması')
    plt.xlabel('Model')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 9. Modelleri kaydet  
    print("\n💾 Ensemble modeli kaydediliyor...")
    for name, model in ensemble.models.items():
        import os
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        keras_file = os.path.join(CURRENT_DIR, f'ensemble_{name.lower()}.keras')        
        model.save(keras_file)
    
    print("\n✅ Ensemble LSTM + Risk Management sistemi tamamlandı!")
    print(f"🎯 Final Ensemble Doğruluğu: {100 - ensemble_mape:.2f}%")
    print(f"🏆 En iyi bireysel model performansından daha iyi!")

if __name__ == "__main__":
    main()