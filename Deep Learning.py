"""
Deep Learning Eğitim Programı - Otomatik Kütüphane Yükleyici ile
Hazırlayan: Dr. Mustafa AFYONLUOĞLU - Eylül 2025 (https://afyonluoglu.org/)

Bu program gerekli kütüphaneleri otomatik olarak kontrol eder ve eksik olanları yükler.
Desteklenen kütüphaneler:
- numpy: Sayısal hesaplamalar için
- pandas: Veri analizi için
- tensorflow: Derin öğrenme modelleri için
- matplotlib: Grafik çizimleri için
- scikit-learn: Makine öğrenmesi yardımcıları için
- pygame: Sistem bilgileri için (opsiyonel)

Program çalışmadan önce tüm gerekli kütüphaneleri kontrol edip eksikleri otomatik yükler.
"""

import os
import subprocess
import sys
from stat import FILE_ATTRIBUTE_ARCHIVE
from time import time

# Kütüphane yükleme fonksiyonu
def install_and_import(package_name, import_name=None):
    """
    Kütüphane yüklü değilse otomatik olarak yükler
    Args:
        package_name (str): pip ile yüklenecek paket adı
        import_name (str): import ifadesinde kullanılacak modül adı
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} kütüphanesi zaten yüklü")
        return True
    except ImportError:
        print(f"⚠️  {package_name} kütüphanesi bulunamadı. Yükleniyor...")
        try:
            # pip güncellemesi ve yükleme
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} başarıyla yüklendi!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} yüklenirken hata oluştu: {e}")
            return False
        except Exception as e:
            print(f"❌ Beklenmedik hata: {e}")
            return False

# Gerekli kütüphaneleri kontrol et ve yükle
required_packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"), 
    ("tensorflow", "tensorflow"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("pygame", "pygame")
]

print("📦 Gerekli kütüphaneler kontrol ediliyor...")
print("-" * 50)

failed_packages = []
for package, import_name in required_packages:
    success = install_and_import(package, import_name)
    if not success:
        failed_packages.append(package)

if failed_packages:
    print(f"\n❌ Şu kütüphaneler yüklenemedi: {', '.join(failed_packages)}")
    print("Lütfen manuel olarak yüklemeyi deneyiniz:")
    for pkg in failed_packages:
        print(f"   pip install {pkg}")
    print("\nProgram devam edecek ancak bazı özellikler çalışmayabilir.")
    input("Devam etmek için ENTER tuşuna basınız...")
else:
    print("🚀 Tüm kütüphaneler hazır! Program başlatılıyor...")
    
print("-" * 50, "\n")

# Kütüphaneleri import et
try:
    from pygame import ver
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # pygame mesajlarını gizle
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # tensorflow uyarılarını gizle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping
import time

# Hazırlayan: Dr. Mustafa AFYONLUOĞLU - Eylül 2025  (https://afyonluoglu.org/)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ############################### California Housing veriseti ############################### 
# Seti internetten indirip CSV olarak kaydetmek için aşağıdaki kodu çalıştırın:
# from sklearn.datasets import fetch_california_housing
# print("📥  California Housing veriseti indiriliyor...")
# data = fetch_california_housing()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df["MedHouseVal"]=data.target

# df.to_csv(os.path.join(CURRENT_DIR, "california_housing.csv"), index=False, encoding="utf-8")
# print("California Housing veriseti kaydedildi.")
# ############################### California Housing veriseti ############################### 

#  pip install scikit-learn
#  pip install tensorflow

NEURON_COUNT = 10
ACTIVATION = 'relu'
#############  Diğer activation türleri:
# 'relu' (Rectified Linear Unit)
# 'sigmoid'
# 'tanh'
# 'softmax'
# 'softplus'
# 'softsign'
# 'selu'
# 'elu'
# 'exponential'
# 'linear'
EPOCHS = 130
HIDDEN_LAYERS = 3
OPTIMIZER = 'adam'  # 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'ftrl', 'nadam'
LOSS_ALGORITHM = 'sparse_categorical_crossentropy'
METRICS = 'accuracy'
VALIDATION_SPLIT = 0.5
TEST_SPLIT = 0.2
LEARNING_RATE = 0.001

OPTIMIZER = OPTIMIZER.lower()
optimizer_dict = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'adam': Adam,
    'adamax': Adamax,
    'nadam': Nadam,
    'ftrl': Ftrl
}

DATAFILE_CLASSIFICATION = "knn_purchase_history.csv"
DATAFILE_REGRESSION = "california_housing.csv"

train_duration = 0
Dataset_count = 0

TERMINAL_COLOR_GREEN = "\033[92m"
TERMINAL_COLOR_RED = "\033[91m"
TERMINAL_COLOR_YELLOW = "\033[93m"
TERMINAL_COLOR_BLUE = "\033[94m"
TERMINAL_COLOR_BOLD = "\033[1m"

TERMINAL_COLOR_RESET = "\033[0m"


def draw_graphs(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib kütüphanesi bulunamadı. Yükleniyor...")
        install_and_import("matplotlib", "matplotlib")
        import matplotlib.pyplot as plt
    
    # Tüm grafikleri aynı ekranda göster
    plt.figure(figsize=(15, 6))
    plt.suptitle('Model Eğitim Süreci', fontsize=16, fontweight='bold', color='blue')
    plt.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.2)

    actual_epochs = len(history.history['loss'])
    early_stopped = actual_epochs < EPOCHS
    
    # Grafiklerin altına bu eğitim sürecine ilişkin parametreleri yazdır
    params_text = (
        f"{pd.Timestamp.now().strftime('%d-%m-%Y  (%H:%M)')}  Process Type: {process_type} | "
        f"Train Duration: {train_duration:.2f} sec | Dataset Rec #: {Dataset_count:,}\n"
        f"Hidden Layers: {HIDDEN_LAYERS} | Neuron Count: {NEURON_COUNT} | Activation: {ACTIVATION} | "
        f"Epochs: {actual_epochs}/{EPOCHS} {'(Early Stopped)' if early_stopped else ''} | Learning Rate: {LEARNING_RATE}\n"
        f"Test Split: {TEST_SPLIT} | Validation Split: {VALIDATION_SPLIT} | "
        f"Optimizer: {OPTIMIZER} | Loss: {LOSS_ALGORITHM} | Metrics: {METRICS}\n"
        f"Early Stopping: Patience={EARLY_STOPPING_PATIENCE}, Min Delta={EARLY_STOPPING_MIN_DELTA}"
    )
    
    plt.figtext(0.01, 0.02, params_text, ha='left', va='bottom', 
                fontsize=11, color='black', 
                bbox=dict(facecolor="#A9E5FF", alpha=0.9))

    ################# İlk grafik - Metrik karşılaştırması
    ax = plt.subplot(1, 3, 1)
    
    if is_regression:
        # Regresyon için MAE grafiği
        if 'mean_absolute_error' in history.history:
            plt.plot(history.history['mean_absolute_error'], label='Train MAE')
            if 'val_mean_absolute_error' in history.history:
                plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
            plt.title('Model MAE - 1')
            plt.ylabel('Mean Absolute Error')
        elif 'mae' in history.history:
            plt.plot(history.history['mae'], label='Train MAE')
            if 'val_mae' in history.history:
                plt.plot(history.history['val_mae'], label='Val MAE')
            plt.title('Model MAE - 2')
            plt.ylabel('Mean Absolute Error')

    else:
        # Sınıflandırma için accuracy grafiği
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
    
    plt.xlabel('Epoch')
    plt.legend()

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("#61AFF8")

    ################# İkinci grafik - Detaylı metrik karşılaştırması
    bx = plt.subplot(1, 3, 2)
    
    if is_regression:
        # Regresyon için detaylı MAE/MSE grafiği
        if 'mean_squared_error' in history.history:
            plt.plot(history.history['mean_squared_error'], label='Train MSE')
            if 'val_mean_squared_error' in history.history:
                plt.plot(history.history['val_mean_squared_error'], label='Val MSE')
            plt.title('Model MSE')
            plt.ylabel('Mean Squared Error')
        elif 'mse' in history.history:
            plt.plot(history.history['mse'], label='Train MSE')
            if 'val_mse' in history.history:
                plt.plot(history.history['val_mse'], label='Val MSE')
            plt.title('Model MSE')
            plt.ylabel('Mean Squared Error')
        else:
            # MSE yoksa MAE'yi tekrar çiz
            if 'mean_absolute_error' in history.history:
                plt.plot(history.history['mean_absolute_error'], label='Train MAE (Eğitim)')
                if 'val_mean_absolute_error' in history.history:
                    plt.plot(history.history['val_mean_absolute_error'], label='Val MAE (Doğrulama)')
                plt.title('MAE Comparison')
                plt.ylabel('Mean Absolute Error')
            elif 'mae' in history.history:
                plt.plot(history.history['mae'], label='Train MAE (Eğitim)')
                if 'val_mae' in history.history:
                    plt.plot(history.history['val_mae'], label='Val MAE (Doğrulama)')
                plt.title('MAE Comparison')
                plt.ylabel('Mean Absolute Error')
    else:
        # Sınıflandırma için detaylı accuracy grafiği
        plt.plot(history.history['accuracy'], label='Train Accuracy (Eğitim)')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy (Doğrulama)')
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
    
    plt.xlabel('Epoch')
    plt.legend()

    for spine in bx.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("#186B0C")

    ################# Üçüncü grafik - Loss karşılaştırması
    cx = plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    for spine in cx.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("#C9801B")

    plt.show()

    # Eğitim notları:
    # Train ↑ ama Val ↓       → overfit.
    # Train ve Val birlikte ↑ → sağlıklı öğrenme

print(TERMINAL_COLOR_GREEN,"="*70)
print("          DEEP LEARNING EĞİTİM PROGRAMI  - TensorFlow Keras")
print("="*70,TERMINAL_COLOR_RESET)

print("\n", TERMINAL_COLOR_YELLOW,"TensorFlow sürümü:", tf.__version__,TERMINAL_COLOR_RESET)
if tf.__version__ > "2.11":
    print(TERMINAL_COLOR_RED,"⚠️  TensorFlow sürümünüz GPU'yu desteklemiyor. 2.11 veya altı bir sürüm yükleyin.",TERMINAL_COLOR_RESET)
sources = tf.config.list_logical_devices()
print("\n ✨  Kullanılabilir kaynaklar (CPU/GPU):", sources)
CPU_COUNT = os.cpu_count()
print("🟢 CPU çekirdek sayısı:", CPU_COUNT)
if not tf.config.list_physical_devices('GPU'):
    print(TERMINAL_COLOR_RED,"⚠️  GPU'ya erişilemedi. ",TERMINAL_COLOR_RESET)

# Veri Setinin Yüklenmesi

# internetten indirmek için:
# dataset = pd.read_csv("https://raw.githubusercontent.com/futurexskill/projects/refs/heads/main/knn-classification/purchase_history.csv")

print("\n", TERMINAL_COLOR_BLUE,"--- Veri seti seçenekleri:","-"*42,TERMINAL_COLOR_RESET)
print(f"   1 - Sınıflandırma - Classification      ({DATAFILE_CLASSIFICATION})")
print(f"   2 - Regresyon - Regression              ({DATAFILE_REGRESSION})")
print(f"Default: {DATAFILE_CLASSIFICATION} (1)")
print("="*70)
print("Seçiminize göre model oluşturulacak ve eğitilecektir.")
secim = input("Seçiminiz (E: exit): ")

if secim == "e" or secim == "E":
    exit()
elif secim == "2":
    FILENAME = DATAFILE_REGRESSION
else:
    FILENAME = DATAFILE_CLASSIFICATION
is_regression = FILENAME == DATAFILE_REGRESSION
CSV_File_Path = os.path.join(CURRENT_DIR, FILENAME)

if is_regression:
    LOSS_ALGORITHM = 'mean_squared_error'  # mse
    METRICS = 'mean_absolute_error'  # mae
    EPOCHS = 80
    NEURON_COUNT = 64
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    ACTIVATION = 'relu'
    HIDDEN_LAYERS = 2

print("📁  Veriseti yolu:", CSV_File_Path)
try:
    dataset = pd.read_csv(CSV_File_Path)
except Exception as e:
    print("❌  Veriseti yüklenemedi. Hata:", e)

Dataset_count = len(dataset)
process_type = "Regression" if is_regression else "Classification"
print(f"Veriseti: (İşlem Türü: {process_type})", "-"*40,"\n", dataset)
# Verisetinin ilk 5 satırını göster
X = dataset.iloc[:, :-1].values     # son sütun hariç tüm kolonlar   : Ana veri seti
y = dataset.iloc[:,-1].values       # sadece en son kolon            : Sonuç veri seti (train sonrasında AI'nın tahmin etmesi beklenen sütun)

print(f"\n 🔢  X shape: {X.shape} ({list(dataset.columns[:-1])})\n 🔢  y shape: {y.shape} ({dataset.columns[-1]})")
print("="*50)
print(dataset.describe().T)
print("="*50)

# X ve Y verilerinin train ve test olarak ayrılması (%80 train %20 test)
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("⚠️  scikit-learn kütüphanesi bulunamadı. Yükleniyor...")
    install_and_import("scikit-learn", "sklearn")
    from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =TEST_SPLIT, random_state=42)

# Kategorik ve sayısal sütunları işle
try:
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
except ImportError:
    print("⚠️  scikit-learn kütüphanesi bulunamadı. Yükleniyor...")
    install_and_import("scikit-learn", "sklearn")
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

if FILENAME == DATAFILE_CLASSIFICATION:
    # 'Gender' ve 'Product ID' sütunlarını one-hot encode et
    # 'Age', 'Salary', 'Price' sütunlarını ölçeklendir
    # 'Customer ID' sütununu dikkate alma (ilk sütun)
    ct = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(), [1, 4]),      # 'Gender' (index 1) ve 'Product ID' (index 4)
        ('scaler', StandardScaler(), [2, 3, 5])   # 'Age' (index 2), 'Salary' (index 3), 'Price' (index 5)
    ], remainder='drop') # Diğer sütunları (Customer ID gibi) bırak
else:
    # Tüm sütunları ölçeklendir
    # ct = ColumnTransformer(transformers=[
    #     ('scaler', StandardScaler(), list(range(X.shape[1]))) # tüm sütunlar
    # ], remainder='drop') # Diğer sütunları bırak
    ct = StandardScaler()

# Dönüşümleri uygula
######## fit_transform: Dönüşüm parametrelerini öğrenir ve uygular
######## transform    : Önceden öğrenilen parametreleri sadece uygular
X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

print(f"\n 🔢  X_train_transformed shape: {X_train_transformed.shape}")

# Dönüştürülmüş veriyi NumPy dizisine çevir (eğer sparse formatta ise)
if not isinstance(X_train_transformed, np.ndarray):
    X_train_transformed = X_train_transformed.toarray()
    print("🔢  X_train_transformed dönüştürüldü (sparse → dense)")
if not isinstance(X_test_transformed, np.ndarray):
    X_test_transformed = X_test_transformed.toarray()
    print("🔢  X_test_transformed dönüştürüldü (sparse → dense)")


# Sinir Ağı Modelinin Oluşturulması

# Rastgelelik için tohum belirleme (aynı sonuçları almak için)
np.random.seed(42)
tf.random.set_seed(42)

# modelde input katmanı özellikle belirtilmez. 
# Sistem input katmanını train serisinden öğrenir ve otomatik olarak oluşturur.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(X_train_transformed.shape[1],)))
for layer in range(HIDDEN_LAYERS):
    print(f"🔸  {layer+1}. katman nöron sayısı: {NEURON_COUNT}")
    if is_regression and layer > 0:
        neu = NEURON_COUNT // 2
        print(f"🔸  Regression - {layer+1}. katman nöron sayısı: {neu}")
        model.add(tf.keras.layers.Dense(neu, activation=ACTIVATION))

    model.add(tf.keras.layers.Dense(NEURON_COUNT, activation=ACTIVATION))
    model.add(tf.keras.layers.Dropout(0.2))  

if is_regression:
    # regresyon için tek nöronlu çıkış katmanı
    print("🔸  Regression - Çıkış katmanı nöron sayısı: 1")
    model.add(tf.keras.layers.Dense(1))  # Default olarak: activation='linear'
else:
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

print(f"\n 🧠  {HIDDEN_LAYERS} katmanlı model oluşturuldu.")

# EARLY_STOP PARAAMETERS - EXPLANATIONS
# monitor='val_loss': Validation loss'u izler (regression için ideal)
# patience=10: 10 epoch boyunca iyileşme yoksa durur
# min_delta=0.001: En az 0.001 iyileşme olmalı, yoksa sayılmaz
# restore_best_weights=True: Durduğunda en iyi epoch'taki ağırlıkları kullanır

# Early Stopping parametreleri 
EARLY_STOPPING_PATIENCE = 10           # Kaç epoch boyunca iyileşme yoksa durduracak
EARLY_STOPPING_MIN_DELTA = 0.001       # Minimum iyileşme miktarı
EARLY_STOPPING_RESTORE_BEST = True     # En iyi ağırlıkları geri yükle

# Modelin Eğitilmesi
print(f"\n 💫  Model ve Eğitim Bilgileri:"+
      f"\n      🔸   Dataset                    : {FILENAME} "+
      f"\n      🔸   Record Count               : {Dataset_count:,}   →  Train: {len(X_train):,}  Test: {len(X_test):,} "+
      f"\n      🔸   Process Type               : {process_type} "+
      f"\n      🔸   Hidden Layers              : {HIDDEN_LAYERS} "+
      f"\n      🔸   Neuron # in each Layer     : {NEURON_COUNT} "+
      f"\n      🔸   Activation                 : {ACTIVATION} "+
      f"\n      🔸   Epochs                     : {EPOCHS} " +
      f"\n      🔸   Optimization               : {OPTIMIZER} " +
      f"\n      🔸   Loss Function              : {LOSS_ALGORITHM} " +
      f"\n      🔸   Metric                     : {METRICS} " +
      f"\n      🔸   Test Split                 : {TEST_SPLIT} " +
      f"\n      🔸   Validation Split           : {VALIDATION_SPLIT} "
    )

if OPTIMIZER in optimizer_dict:
    optimizer_class = optimizer_dict[OPTIMIZER]
    try:
        optimizer_instance = optimizer_class(learning_rate=LEARNING_RATE)
        print(f"      🔸   Learning Rate              : {LEARNING_RATE} (atanan)")
    except TypeError:
        optimizer_instance = optimizer_class()
        print("      🔸   Seçilen optimizer learning_rate parametresini desteklemiyor, varsayılan kullanıldı.")
else:
    optimizer_instance = OPTIMIZER  # string olarak bırak (Keras default davranışı)
    print("      🔸   Bilinmeyen optimizer, string olarak kullanılacak.")

# Optimizer: Learning Rate'i adaptif şekilde ayarlayan algoritmalar
model.compile(optimizer=optimizer_instance, loss=LOSS_ALGORITHM, metrics=[METRICS])

# Early Stopping callback'ini oluştur
early_stopping = EarlyStopping(
    monitor='val_loss',                                # İzlenecek metrik (validation loss)
    patience=EARLY_STOPPING_PATIENCE,                  # Kaç epoch sabır gösterecek
    min_delta=EARLY_STOPPING_MIN_DELTA,                # Minimum iyileşme miktarı
    restore_best_weights=EARLY_STOPPING_RESTORE_BEST,  # En iyi ağırlıkları geri yükle
    verbose=1                                          # Durduğunda bilgi ver
)

print(f"      🔸   Early Stopping Patience    : {EARLY_STOPPING_PATIENCE} epochs")
print(f"      🔸   Early Stopping Min Delta   : {EARLY_STOPPING_MIN_DELTA}")
print(f"      🔸   Early Stopping Monitor     : val_loss")

# train_test_split ile verinin bir kısmını test için ayrılır (modelin hiç görmediği, nihai değerlendirme için).
# validation_split, eğitim verisinin bir kısmını eğitim sırasında modelin doğrulama (validation) performansını izlemek için ayırır.
print(f"\n💫  Model eğitiliyor... ")

if EPOCHS <= 20:
    verboseLevel = 1
else:
    verboseLevel = 0

basla = time.time()

# Batch_size= Stochastic Gradient Descent için alınan "mini_batches size"
history = model.fit(
    X_train_transformed, 
    y_train, 
    epochs=EPOCHS, 
    validation_split=VALIDATION_SPLIT, 
    batch_size=32,
    verbose=verboseLevel,
    callbacks=[early_stopping]  # Early stopping callback'ini ekle
) # verbose=0: epoch başına ilerleme çubuğu gösterilmez
bitis = time.time()
train_duration += (bitis - basla)
print(f"⏱️  Eğitim süresi: {train_duration:.2f} saniye")

if is_regression:
    val_metric = history.history['val_mean_absolute_error'][-1]
    print(f"🚩 {HIDDEN_LAYERS} katman → Val MAE: {val_metric:.5f}")
else:
    val_acc = history.history['val_accuracy'][-1]
    print(f"🚩 {HIDDEN_LAYERS} katman → Doğruluk: {val_acc:.5f}")
    
print("✅  Model eğitildi\n")

# Modelin test veri seti üzerinde değerlendirilmesi
print("➡️  Model test verisi üzerinde değerlendiriliyor...")
loss, metric = model.evaluate(X_test_transformed, y_test)

print(f"\n🔴  Test Seti Kaybı    : {loss}")
if is_regression:
    print(f"🟢  Test Seti MAE      : {metric}\n")
else:
    print(f"🟢  Test Seti Doğruluğu: {metric}\n")

model.summary()
print("_"*50,"\n")

# Eğitim sonrası bilgi ekleyin
actual_epochs = len(history.history['loss'])
if actual_epochs < EPOCHS:
    print(f"🛑  Early Stopping devreye girdi! Eğitim {actual_epochs}. epoch'ta durduruldu.")
    print(f"    (Planlanan: {EPOCHS}, Gerçekleşen: {actual_epochs})")
else:
    print(f"✅  Tüm {EPOCHS} epoch tamamlandı (Early Stopping devreye girmedi)")

# Örnek veriyle test et

if is_regression:
    # "California Housing" için örnek veri
    # Fields: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    print("\n➡️  Regression: Model ile tahmin yapılıyor...")
    new_data = np.array([[8.3252, 25.0, 6.0, 1.2, 600, 2.75, 39.00, -122.23]])
elif FILENAME == DATAFILE_CLASSIFICATION:
    # "knn_purchase_history.csv" için örnek veri
    # Fields: Customer ID, Gender, Age, Salary, Product ID, Price
    print("\n➡️  Classification: Model ile tahmin yapılıyor...")
    new_data = np.array([[1001, 'Male', 42, 50000, 'P01', 3000]])
else:
    print("❌  Bilinmeyen veri seti, tahmin yapılamıyor.")
    exit()

new_data_transformed = ct.transform(new_data)

print(f"{new_data=}")

if not isinstance(new_data_transformed, np.ndarray):
    new_data_transformed = new_data_transformed.toarray()

prediction = model.predict(new_data_transformed)

if is_regression:
    print(f"✨ Tahmin Edilen Ev Fiyatı: ${prediction[0][0]*100000:,.0f}")
else:
    print(f"✨ Tahmin Edilen Satın Alma Olasılığı: % {(prediction[0][1]*100):.2f}")

print("_"*50,"\n")

draw_graphs(history)

# input()  # grafik ekranı block=false ise açık kalmması için