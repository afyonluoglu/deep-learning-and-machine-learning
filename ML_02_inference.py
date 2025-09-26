import pickle
import pandas as pd
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*80)
print("Bu program ile kaydedilmiş KNN modeli kullanılarak yeni veriler üzerinde tahmin yapılacaktır.")
print("Yeni veriler 'new_customers.csv' dosyasından okunacaktır.")
print("Tahminler 'knn_model_predictions.csv' dosyasına kaydedilecektir.")
print("Program için knn_model.pickle ve knn_scaler.pickle dosyalarının 'ML_knn_classification.py' programı ile üretilmiş olması gerekmektedir.")
print("="*80)

try:
  with open(os.path.join(CURRENT_DIR, 'knn_model.pickle'), 'rb') as f:
    knn_new = pickle.load(f)

  with open(os.path.join(CURRENT_DIR, 'knn_scaler.pickle'), 'rb') as f:
    scaler_new = pickle.load(f)
except FileNotFoundError as e:
    print(f"❌ Model dosyası bulunamadı: {e}")
    exit()

CSV_File_Path = os.path.join(CURRENT_DIR, "knn_new_customers.csv")
new_df = pd.read_csv(CSV_File_Path)

print("\n ➡️  Yeni müşteri verisi:")
# print(new_df.head())
print(new_df)

# "one-hot encoding" uygula
# drop_first=True:ortaya çıkan ilk kategoriyi siler
#     pandas alfabetik olarak ilk gelen kategoriyi ('Female') atar ve sadece tek bir sütun oluşturur
# "drop_first=false" olursa sonuç çıktıda 2 sütun olur, (biri erkek diğeri kadın için)
gender_encoded_new = pd.get_dummies(new_df['Gender'], drop_first=True)
print("\n ➡️  Yeni müşteri verisindeki cinsiyet sütununun 0/1'e çevrilmiş hali:")
print(gender_encoded_new.head())

df_new_2 = pd.concat([new_df, gender_encoded_new], axis=1)
print("\n ➡️  Yeni müşteri verisi (cinsiyet sütunu eklendikten sonra):")
print(df_new_2)

# İşleme girecek sütunları içeren numpy dizisi
x_new = df_new_2[['Male','Age','Salary','Price']].to_numpy()
print("\n ➡️  Yeni müşteri verisinin numpy dizisi hali:")
# print(x_new[:5])
print(x_new)

# Ölçeklendirme işlemi
# DİKKAT: Aşağıdaki satırda fit_transform değil, sadece transform kullanılmalı
# fit_transform yeni veriye değil, eğitim verisine uygulanır
x_new_scale2 = scaler_new.transform(x_new)
print("\n ➡️  Yeni müşteri verisinin ölçeklendirilmiş hali:")
print(x_new_scale2)

# Ölçeklenmiş verilerin istatistiklerini kontrol edin
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# StandardScaler, verileri ortalaması 0 ve standart sapması 1 olacak şekilde ölçeklendirir.
print("\n🔍 Ölçeklenmiş verilerin istatistikleri:")
print(f"Ortalama: {np.mean(x_new_scale2, axis=0)}")
print(f"Standart sapma: {np.std(x_new_scale2, axis=0)}")

# Tahmin işlemi
y_new_pred = knn_new.predict(x_new_scale2)
print("\n ➡️  Yeni müşteri verisi için tahminler (0: satın almadı, 1: satın aldı):")
print(y_new_pred)

# Tahmin sonuçlarını orijinal dataframe'e ekle
df_new_2['will_purchase'] = y_new_pred
print("\n ➡️  Yeni müşteri verisi (tahminler eklendikten sonra):")
print(df_new_2)

# Beklenen sütunların kontrolü
expected_columns = ['Male', 'Age', 'Salary', 'Price']
if not all(col in df_new_2.columns for col in expected_columns):
    print("❌ Gerekli sütunlar eksik!")
    exit()

# Tahmin sonuçlarını CSV dosyasına kaydet
df_new_2.to_csv(os.path.join(CURRENT_DIR,"knn_model_predictions.csv") ,index=False)
print("\n ➡️  Tahminler 'knn_model_predictions.csv' dosyasına kaydedildi.")

print("\n 🟢 🟡 🔴  Tek bir yeni müşteri verisi üzerinde tahmin yapma örneği:")
# Yeni bir müşteri verisi üzerinde tahmin yapalım
row_values = [1, 32, 40000,5000]

# Yeni müşteri verisini içeren DataFrame oluştur
new_data = pd.DataFrame([row_values], columns=['Male', 'Age', 'Salary', 'Price'])

print("\n ➡️  Tek bir yeni müşteri verisi:")
print(new_data)

# Yeni müşteri verisini ölçeklendir
new_data_scaled = scaler_new.transform(new_data.to_numpy())
print("\n ➡️  Tek bir yeni müşteri verisinin ölçeklendirilmiş hali:")
print(new_data_scaled)

# Alternatif olarak, numpy dizisi oluşturup ölçeklendirebiliriz
# x_new = np.array(row_values).reshape(1,-1)
# print(f"\n {x_new=}")
# x_new_scale = scaler_new.transform(x_new)
# print(f"\n {x_new_scale=}")


# Yeni müşteri verisi üzerinde tahmin yap
new_prediction = knn_new.predict(new_data_scaled)
print("\n ➡️  Tek bir yeni müşteri verisi için tahmin (0: satın almadı, 1: satın aldı):")
prediction =str(new_prediction[0])
print("Purchase ? "+str(prediction))

print("\n ✅  İşlem tamamlandı.")
