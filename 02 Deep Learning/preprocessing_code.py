import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

# Veriyi yükle
data = pd.read_csv(r'c:\Users\ASUS\Desktop\Python with AI\Yapay Zeka - AI Traning Kits\02 Deep Learning\demo_dataset.csv')

# Veri setinin temel bilgilerini göster
print('📊 Veri Seti Bilgileri:')
print(f'Satır sayısı: {len(data)}')
print(f'Sütun sayısı: {len(data.columns)}')
print(f'Sütunlar: {list(data.columns)}')
print('\nİlk 5 satır:')
print(data.head())

# Eksik değerleri kontrol et
missing_data = data.isnull().sum()
if missing_data.any():
    print('\n⚠️ Eksik Değerler:')
    for col, missing_count in missing_data[missing_data > 0].items():
        print(f'  {col}: {missing_count}')

    # Eksik değerleri doldur
    data.loc[data['salary'].isnull(), 'salary'] = data['salary'].median()

else:
    print('✅ Eksik değer yok')

# Kategorik sütunları encode et
categorical_encoders = {}

# gender sütunu için Label Encoding (2 kategori)
categorical_encoders['gender'] = LabelEncoder()
data['gender'] = categorical_encoders['gender'].fit_transform(data['gender'])

# education sütunu için One-Hot Encoding (4 kategori)
data = pd.get_dummies(data, columns=['education'], prefix='education')

# Sayısal sütunları normalize et (hedef sütun hariç)
scaler = StandardScaler()
numeric_features = ['age', 'salary', 'experience', 'score']
print(f'\n📏 Normalize edilecek sayısal sütunlar: {numeric_features}')

# Normalizasyon öncesi istatistikler
print('\n📊 Normalizasyon Öncesi İstatistikler:')
print(data[numeric_features].describe())

# Normalizasyonu uygula
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Normalizasyon sonrası istatistikler
print('\n📊 Normalizasyon Sonrası İstatistikler:')
print(data[numeric_features].describe())

# Hedef sütunu (high_earner) sayısal - ek işlem gerekmiyor
print(f'\n🎯 Hedef sütunu: high_earner (sayısal)')

# Son veri seti durumu
print(f'\n📋 İşlenmiş Veri Seti:')
print(f'Boyut: {data.shape}')
print(f'Sütunlar: {list(data.columns)}')

# Özellik ve hedef değişkenleri ayır
if 'high_earner' in data.columns:
    X = data.drop('high_earner', axis=1)
    y = data['high_earner']
else:
    print('❌ Hedef sütunu bulunamadı!')
    print('Mevcut sütunlar:', list(data.columns))
    # En son sütunu hedef olarak kabul et
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print(f'Son sütun hedef olarak seçildi: {y.name}')

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sonuçları göster
print('\n✅ Veri ön işleme tamamlandı!')
print(f'📊 Eğitim seti boyutu: {X_train.shape}')
print(f'📊 Test seti boyutu: {X_test.shape}')
print(f'🎯 Hedef değişken dağılımı:')
if hasattr(y, 'value_counts'):
    print(y.value_counts())
else:
    print(f'Min: {y.min()}, Max: {y.max()}, Ortalama: {y.mean():.2f}')

# Opsiyonel: Sonuçları kaydet
save_choice = input('\nİşlenmiş veriyi kaydetmek istiyor musunuz? (y/n): ')
if save_choice.lower() == 'y':
    processed_filename = 'processed_data.csv'
    data.to_csv(processed_filename, index=False)
    print(f'✅ İşlenmiş veri {processed_filename} dosyasına kaydedildi.')