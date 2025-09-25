
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Hazırlayan: Dr. Mustafa AFYONLUOĞLU - Eylül 2025 (https://afyonluoglu.org/)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*80)
print("🟢 Bu program ile KNN sınıflandırma modeli oluşturulacak ve eğitilecektir.")
print("🟢 Veri seti 'purchase_history.csv' dosyasından okunacaktır.")
print("🟢 Eğitilen model ve scaler verileri 'knn_model.pickle' ve 'knn_scaler.pickle' isimleriyle kaydedilecektir.")
print("="*80, "\n")

CSV_File_Path = os.path.join(CURRENT_DIR, "knn_purchase_history.csv")
df = pd.read_csv(CSV_File_Path)

print(f"🟠 Veri Seti Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
print("➡️  Veri seti ilk 5 satır", "="*30)
print(df.head())

# male/female değerlerini içeren sütun eğitim için 0/1'e çevrilmeli
gender_encoded = pd.get_dummies(df['Gender'])
print("\n ➡️  Cinsiyet sütunu", "="*30)
print(gender_encoded.head())

gender_encoded = pd.get_dummies(df['Gender'],drop_first=True)
print("\n ➡️  Cinsiyet sütununun 0/1'e çevrilmiş hali:", "="*30)
print(gender_encoded.head())

# üretilen bu yeni 0/1 kolonunu ana veriye ekleyelim
df = pd.concat([df,gender_encoded],axis=1)
print("\n ➡️  Cinsiyet sütunu eklendikten sonra veri seti:", "="*30)
print(df.head())

# veriyi numpy verisine çevirelim
x = df[['Male','Age','Salary','Price']].to_numpy()
print("\n ➡️  Veri setinin numpy dizisi hali:", "="*30)
print("Başlıklar: 'Male','Age','Salary','Price'")
print(x[:5])

# satın alma sonuçları ayrı bir dizide,
y = df['Purchased'].to_numpy()
print("\n ➡️  Satın alma sonuçları 'Purchased' dizisi:", "="*30)
print(y[:5])

# ana veriyi train ve test verilerine ayır (%80 train %20 test)
# x_train: eğitim için kullanılacak ana veri
# x_test: test için kullanılacak ana veri
# y_train: eğitim için hedef değişken = purchased
# y_test: test için hedef değişken = purchased
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n ➡️  Eğitim ve test verilerinin boyutları:")
print("x_train boyutu: ",len(x_train))
print("x_test boyutu : ",len(x_test),"\n")

print("Train Verisi - ilk 5 Satır:\n",x_train[:5],"\n")
print("Test Verisi  - ilk 5 Satır:\n",x_test[:5])

# verileri ölçeklendirme (standardizasyon)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print("\n ➡️  Ölçeklendirilmiş x_train verisi:\n",x_train[:5])

# KNN Modelinin Oluşturulması ve Eğitilmesi
k= 5   # 5 uzaklıktaki komşuya bak
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train,y_train)
print("\n ✨  KNN Modeli eğitildi.")

y_pred = knn.predict(x_test)
print("\n ➡️  Test verisi için gerçek değerler:\n",y_test[:15])
print("\n ➡️  Test verisi için tahminler:\n",y_pred[:15])


accuracy = accuracy_score(y_test, y_pred)
print("\n ➡️  Test verisi için doğruluk skoru:", accuracy)

# Modeli kaydetme

model_file = os.path.join(CURRENT_DIR, 'knn_model.pickle')
with open(model_file,'wb') as f:
  pickle.dump(knn,f)

scaler_file = os.path.join(CURRENT_DIR, 'knn_scaler.pickle')
with open(scaler_file,'wb') as f:
  pickle.dump(scaler,f)

print("\n ➡️  Model ve scaler verileri 'knn_model.pickle' ve 'knn_scaler.pickle' dosyalarına kaydedildi.")  


