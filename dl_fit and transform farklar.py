# Basit bir StandardScaler örneği
from sklearn.preprocessing import StandardScaler

# Örnek veriler
X_train = [[100], [200], [300]]  # Maaş verileri
X_test = [[150], [250]]

scaler = StandardScaler()

# ❌ YANLIŞ YAKLAŞIM (Data Leakage)
# X_train_wrong = scaler.fit_transform(X_train)  # mean=200, std=81.6
# X_test_wrong = scaler.fit_transform(X_test)    # mean=200, std=50 (farklı!)

# ✅ DOĞRU YAKLAŞIM
X_train_scaled = scaler.fit_transform(X_train)  # mean=200, std=81.6 öğrenir
X_test_scaled = scaler.transform(X_test)        # Aynı parametreleri kullanır

print(f"Train scaled: {X_train_scaled.flatten()}")
print(f"Test scaled : {X_test_scaled.flatten()}")
print(f"Scaler mean : {scaler.mean_[0]:.1f}")
print(f"Scaler std  : {scaler.scale_[0]:.1f}")