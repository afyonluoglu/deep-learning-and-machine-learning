"""
Veri Analizi ve Ön İşleme Rehberi
Deep Learning için veri hazırlama süreçlerini öğretir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class DataAnalysisGuide:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.data = None
        self.analysis_results = {}
        
    def load_and_explore_data(self, file_path):
        """Veriyi yükle ve temel keşif yap"""
        print("📂 VERİ YÜKLENİYOR VE KEŞF EDİLİYOR...")
        print("="*60)
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"✅ Veri başarıyla yüklendi: {file_path}")
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
            
        # Temel bilgiler
        print(f"\n📊 VERİ SETİ GENEL BİLGİLERİ:")
        print("-"*40)
        print(f"🔢 Satır sayısı: {len(self.data):,}")
        print(f"🔢 Sütun sayısı: {len(self.data.columns)}")
        print(f"💾 Hafıza kullanımı: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Sütun türleri
        print(f"\n📋 SÜTUN TÜRLERİ:")
        print("-"*40)
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} sütun")
            
        # İlk 5 satır
        print(f"\n👀 VERİNİN İLK 5 SATIRI:")
        print("-"*40)
        print(self.data.head())
        
        # Eksik değerler
        missing_data = self.data.isnull().sum()
        if missing_data.any():
            print(f"\n⚠️  EKSİK DEĞERLER:")
            print("-"*40)
            for col, missing_count in missing_data[missing_data > 0].items():
                missing_percent = (missing_count / len(self.data)) * 100
                print(f"  {col}: {missing_count} ({missing_percent:.1f}%)")
        else:
            print(f"\n✅ Eksik değer bulunmuyor!")
            
        return True
    
    def analyze_data_quality(self):
        """Veri kalitesini analiz et"""
        if self.data is None:
            print("❌ Önce veri yüklemelisiniz!")
            return
            
        print("\n🔍 VERİ KALİTESİ ANALİZİ")
        print("="*60)
        
        quality_issues = []
        
        # 1. Duplicate kontrolü
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Tekrarlanan satırlar: {duplicates}")
            print(f"⚠️  {duplicates} tekrarlanan satır bulundu")
        else:
            print("✅ Tekrarlanan satır yok")
            
        # 2. Aykırı değer analizi (sayısal sütunlar için)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        if len(numeric_cols) > 0:
            print(f"\n📈 AYKIRI DEĞER ANALİZİ (Sayısal Sütunlar):")
            print("-"*50)
            
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                outlier_percent = (outliers / len(self.data)) * 100
                
                print(f"  {col}: {outliers} aykırı değer (%{outlier_percent:.1f})")
                if outlier_percent > 5:  # %5'ten fazla aykırı değer
                    outlier_cols.append(col)
                    quality_issues.append(f"{col} sütununda çok aykırı değer")
        
        # 3. Kategorik sütun analizi
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📊 KATEGORİK SÜTUN ANALİZİ:")
            print("-"*50)
            
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                unique_percent = (unique_count / len(self.data)) * 100
                
                print(f"  {col}: {unique_count} benzersiz değer (%{unique_percent:.1f})")
                
                if unique_percent > 50:
                    quality_issues.append(f"{col} sütununda çok fazla kategori")
                elif unique_count == 1:
                    quality_issues.append(f"{col} sütunu sabit (tek değer)")
        
        # Sonuç
        if quality_issues:
            print(f"\n⚠️  VERİ KALİTESİ UYARILARI:")
            print("-"*50)
            for i, issue in enumerate(quality_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print(f"\n✅ Veri kalitesi genel olarak iyi görünüyor!")
            
        return quality_issues
    
    def create_data_visualization_guide(self):
        """Veri görselleştirme rehberi"""
        if self.data is None:
            print("❌ Önce veri yüklemelisiniz!")
            return
            
        print("\n📊 VERİ GÖRSELLEŞTİRME REHBERİ")
        print("="*60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            print("❌ Görselleştirilebilecek sütun bulunamadı.")
            return
            
        # Subplot sayısını hesapla
        total_plots = min(len(numeric_cols), 4) + min(len(categorical_cols), 2)
        if total_plots == 0:
            return
            
        rows = (total_plots + 2) // 3  # 3 sütunlu layout
        cols = min(total_plots, 3)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if total_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Sayısal sütunlar için histogram
        for i, col in enumerate(numeric_cols[:4]):  # İlk 4 sayısal sütun
            axes[plot_idx].hist(self.data[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[plot_idx].set_title(f' {col} Dağılımı')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Frekans')
            
            # İstatistikler ekle
            mean_val = self.data[col].mean()
            median_val = self.data[col].median()
            axes[plot_idx].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Ortalama: {mean_val:.2f}')
            axes[plot_idx].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Medyan: {median_val:.2f}')
            axes[plot_idx].legend()
            
            plot_idx += 1
        
        # Kategorik sütunlar için bar chart
        for i, col in enumerate(categorical_cols[:2]):  # İlk 2 kategorik sütun
            if plot_idx >= total_plots:
                break
                
            value_counts = self.data[col].value_counts().head(10)  # En çok 10 kategori
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[plot_idx].set_title(f' {col} Kategorileri')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Sayı')
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            plot_idx += 1
        
        # Kullanılmayan subplot'ları gizle
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Korelasyon analizi (sayısal sütunlar için)
        if len(numeric_cols) > 1:
            print("\n🔗 KORELASYON ANALİZİ:")
            print("-"*50)
            
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Korelasyon Katsayısı'})
            plt.title(' Sütunlar Arası Korelasyon Matrisi')
            plt.tight_layout()
            plt.show()
            
            # Yüksek korelasyonları bul
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    if corr_val > 0.8:  # Yüksek korelasyon eşiği
                        high_corr_pairs.append((
                            correlation_matrix.columns[i], 
                            correlation_matrix.columns[j], 
                            correlation_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                print("⚠️  YÜKSEK KORELASYON UYARISI:")
                for col1, col2, corr in high_corr_pairs:
                    print(f"   {col1} ↔ {col2}: {corr:.3f}")
                print("   Not: Yüksek korelasyonlu sütunlardan birini çıkarmayı düşünün.")
    
    def preprocessing_recommendations(self):
        """Ön işleme önerilerini ver"""
        if self.data is None:
            print("❌ Önce veri yüklemelisiniz!")
            return
            
        print("\n🔧 VERİ ÖN İŞLEME ÖNERİLERİ")
        print("="*60)
        
        recommendations = []
        
        # 1. Eksik değer önerileri
        missing_data = self.data.isnull().sum()
        if missing_data.any():
            print("📋 EKSİK DEĞER ÖNERİLERİ:")
            for col, missing_count in missing_data[missing_data > 0].items():
                missing_percent = (missing_count / len(self.data)) * 100
                
                if missing_percent < 5:
                    recommendation = f"   {col}: Az eksik (%{missing_percent:.1f}) → Silme veya ortalama ile doldurma"
                elif missing_percent < 30:
                    if self.data[col].dtype in ['int64', 'float64']:
                        recommendation = f"   {col}: Orta eksik (%{missing_percent:.1f}) → Medyan ile doldurma önerilir"
                    else:
                        recommendation = f"   {col}: Orta eksik (%{missing_percent:.1f}) → Mod ile doldurma önerilir"
                else:
                    recommendation = f"   {col}: Çok eksik (%{missing_percent:.1f}) → Sütunu çıkarmayı düşünün"
                    
                print(recommendation)
                recommendations.append(recommendation)
        
        # 2. Normalizasyon önerileri
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📏 NORMALİZASYON ÖNERİLERİ:")
            
            for col in numeric_cols:
                col_min = self.data[col].min()
                col_max = self.data[col].max()
                col_range = col_max - col_min
                
                if col_range > 1000:
                    recommendation = f"   {col}: Geniş aralık ({col_min:.1f} - {col_max:.1f}) → StandardScaler önerilir"
                    print(recommendation)
                    recommendations.append(recommendation)
                elif col_min >= 0 and col_max <= 1:
                    print(f"   {col}: Zaten normalize ({col_min:.3f} - {col_max:.3f})")
                else:
                    recommendation = f"   {col}: Normal aralık → MinMaxScaler kullanabilirsiniz"
                    print(recommendation)
        
        # 3. Kategorik değişken önerileri  
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n🏷️  KATEGORİK DEĞİŞKEN ÖNERİLERİ:")
            
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                
                if unique_count == 2:
                    recommendation = f"   {col}: İkili kategori → LabelEncoder kullanın"
                    print(recommendation)
                elif unique_count <= 10:
                    recommendation = f"   {col}: Az kategori ({unique_count}) → OneHotEncoder kullanın"
                    print(recommendation)
                else:
                    recommendation = f"   {col}: Çok kategori ({unique_count}) → Target Encoding veya Embedding düşünün"
                    print(recommendation)
                    
                recommendations.append(recommendation)
        
        # 4. Örnek kod üretimi
        self.generate_preprocessing_code(recommendations)
        
        return recommendations
    
    def generate_preprocessing_code(self, recommendations):
        """Öneriler temelinde örnek kod üret"""
        print(f"\n💻 ÖNERİLER DOĞRULTUSUNDA ÖRNEK KOD:")
        print("="*60)
        
        # Hedef sütunu belirleme (son sütun varsayılan)
        target_column = self.data.columns[-1]
        
        code_lines = [
            "import pandas as pd",
            "import numpy as np", 
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder",
            "from sklearn.model_selection import train_test_split",
            "import os",
            "",
            "# Veriyi yükle",
            f"data = pd.read_csv(r'{self.dataset_path if self.dataset_path else 'your_dataset.csv'}')",
            "",
            "# Veri setinin temel bilgilerini göster",
            "print('📊 Veri Seti Bilgileri:')",
            "print(f'Satır sayısı: {len(data)}')",
            "print(f'Sütun sayısı: {len(data.columns)}')",
            "print(f'Sütunlar: {list(data.columns)}')",
            "print('\\nİlk 5 satır:')",
            "print(data.head())",
            ""
        ]
        
        # Eksik değer doldurma
        missing_data = self.data.isnull().sum()
        if missing_data.any():
            code_lines.extend([
                "# Eksik değerleri kontrol et",
                "missing_data = data.isnull().sum()",
                "if missing_data.any():",
                "    print('\\n⚠️ Eksik Değerler:')",
                "    for col, missing_count in missing_data[missing_data > 0].items():",
                "        print(f'  {col}: {missing_count}')",
                ""
            ])
            
            code_lines.append("    # Eksik değerleri doldur")
            for col, missing_count in missing_data[missing_data > 0].items():
                if self.data[col].dtype in ['int64', 'float64']:
                    code_lines.append(f"    data.loc[data['{col}'].isnull(), '{col}'] = data['{col}'].median()")
                else:
                    code_lines.append(f"    data.loc[data['{col}'].isnull(), '{col}'] = data['{col}'].mode()[0]")
            code_lines.extend(["", "else:", "    print('✅ Eksik değer yok')", ""])
        
        # Kategorik encoding
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            code_lines.extend([
                "# Kategorik sütunları encode et",
                "categorical_encoders = {}",
                ""
            ])
            
            for col in categorical_cols:
                # Hedef sütunu kategorik encoding'den hariç tut
                if col == target_column:
                    continue
                    
                unique_count = self.data[col].nunique()
                if unique_count == 2:
                    code_lines.extend([
                        f"# {col} sütunu için Label Encoding (2 kategori)",
                        f"categorical_encoders['{col}'] = LabelEncoder()",
                        f"data['{col}'] = categorical_encoders['{col}'].fit_transform(data['{col}'])"
                    ])
                elif unique_count <= 10:
                    code_lines.extend([
                        f"# {col} sütunu için One-Hot Encoding ({unique_count} kategori)",
                        f"data = pd.get_dummies(data, columns=['{col}'], prefix='{col}')"
                    ])
                else:
                    code_lines.extend([
                        f"# {col} sütunu çok fazla kategori içeriyor ({unique_count}), Target Encoding veya diğer yöntemler düşünülmeli",
                        f"# Şimdilik One-Hot Encoding kullanılıyor (dikkatli olun!)",
                        f"data = pd.get_dummies(data, columns=['{col}'], prefix='{col}')"
                    ])
                code_lines.append("")
        
        # Normalizasyon 
        numeric_cols = [col for col in self.data.select_dtypes(include=[np.number]).columns if col != target_column]
        if len(numeric_cols) > 0:
            code_lines.extend([
                "# Sayısal sütunları normalize et (hedef sütun hariç)",
                "scaler = StandardScaler()",
                f"numeric_features = {numeric_cols}",
                "print(f'\\n📏 Normalize edilecek sayısal sütunlar: {numeric_features}')",
                "",
                "# Normalizasyon öncesi istatistikler",
                "print('\\n📊 Normalizasyon Öncesi İstatistikler:')",
                "print(data[numeric_features].describe())",
                "",
                "# Normalizasyonu uygula",
                "data[numeric_features] = scaler.fit_transform(data[numeric_features])",
                "",
                "# Normalizasyon sonrası istatistikler",
                "print('\\n📊 Normalizasyon Sonrası İstatistikler:')",
                "print(data[numeric_features].describe())",
                ""
            ])
        
        # Hedef sütunu işleme
        if target_column in self.data.columns:
            if self.data[target_column].dtype == 'object':
                code_lines.extend([
                    f"# Hedef sütunu ({target_column}) kategorik - Label Encoding uygula",
                    f"target_encoder = LabelEncoder()",
                    f"data['{target_column}'] = target_encoder.fit_transform(data['{target_column}'])",
                    f"print(f'\\n🎯 Hedef sütunu encode edildi. Sınıflar: {{list(target_encoder.classes_)}}')",
                    ""
                ])
            else:
                code_lines.extend([
                    f"# Hedef sütunu ({target_column}) sayısal - ek işlem gerekmiyor",
                    f"print(f'\\n🎯 Hedef sütunu: {target_column} (sayısal)')",
                    ""
                ])
        
        # Final steps
        code_lines.extend([
            "# Son veri seti durumu",
            "print(f'\\n📋 İşlenmiş Veri Seti:')",
            "print(f'Boyut: {data.shape}')",
            "print(f'Sütunlar: {list(data.columns)}')",
            "",
            "# Özellik ve hedef değişkenleri ayır",
            f"if '{target_column}' in data.columns:",
            f"    X = data.drop('{target_column}', axis=1)",
            f"    y = data['{target_column}']",
            "else:",
            "    print('❌ Hedef sütunu bulunamadı!')",
            "    print('Mevcut sütunlar:', list(data.columns))",
            "    # En son sütunu hedef olarak kabul et",
            "    X = data.iloc[:, :-1]",
            "    y = data.iloc[:, -1]",
            "    print(f'Son sütun hedef olarak seçildi: {y.name}')",
            "",
            "# Eğitim ve test setlerine ayır", 
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "",
            "# Sonuçları göster",
            "print('\\n✅ Veri ön işleme tamamlandı!')",
            "print(f'📊 Eğitim seti boyutu: {X_train.shape}')",
            "print(f'📊 Test seti boyutu: {X_test.shape}')",
            "print(f'🎯 Hedef değişken dağılımı:')",
            "if hasattr(y, 'value_counts'):",
            "    print(y.value_counts())",
            "else:",
            "    print(f'Min: {y.min()}, Max: {y.max()}, Ortalama: {y.mean():.2f}')",
            "",
            "# Opsiyonel: Sonuçları kaydet",
            "save_choice = input('\\nİşlenmiş veriyi kaydetmek istiyor musunuz? (y/n): ')",
            "if save_choice.lower() == 'y':",
            "    processed_filename = 'processed_data.csv'",
            "    data.to_csv(processed_filename, index=False)",
            "    print(f'✅ İşlenmiş veri {processed_filename} dosyasına kaydedildi.')",
        ])
        
        # Kodu yazdır
        for line in code_lines:
            print(line)
        
        # Dosyaya kaydet
        out_file = os.path.join(CURRENT_DIR, "preprocessing_code.py")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(code_lines))

        print(f"\n✅ Kod '{out_file}' dosyasına kaydedildi!")
        print(f"💡 İpucu: Kodu çalıştırmadan önce hedef sütun adını kontrol edin!")
        
        # Oluşturulan kodu test et
        self.test_generated_code(out_file)
    
    def test_generated_code(self, code_file):
        """Oluşturulan preprocessing kodunu test et"""
        print(f"\n🧪 OLUŞTURULAN KOD TEST EDİLİYOR...")
        print("="*60)
        
        try:
            # Test amaçlı kodu çalıştır (güvenlik için sadece syntax check)
            with open(code_file, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Syntax kontrolü
            try:
                compile(code_content, code_file, 'exec')
                print("✅ Syntax kontrolü başarılı - kod hatalarız!")
            except SyntaxError as e:
                print(f"❌ Syntax hatası bulundu: {e}")
                return False
            
            # Import kontrolü
            required_imports = ['pandas', 'numpy', 'sklearn']
            for imp in required_imports:
                if imp not in code_content:
                    print(f"⚠️  {imp} import eksik olabilir")
            
            # Temel kod yapısı kontrolü
            essential_parts = [
                'pd.read_csv',
                'train_test_split', 
                'print',
                'shape'
            ]
            
            missing_parts = []
            for part in essential_parts:
                if part not in code_content:
                    missing_parts.append(part)
            
            if missing_parts:
                print(f"⚠️  Eksik kod parçaları: {missing_parts}")
            else:
                print("✅ Tüm temel kod parçaları mevcut")
            
            # Dosya boyutu kontrolü
            lines = len(code_content.split('\n'))
            print(f"📄 Oluşturulan kod: {lines} satır")
            
            if lines < 10:
                print("⚠️  Kod çok kısa görünüyor, eksiklik olabilir")
            elif lines > 200:
                print("⚠️  Kod çok uzun, optimizasyon gerekebilir") 
            else:
                print("✅ Kod boyutu uygun")
            
            print(f"🚀 Kodu çalıştırmak için: python {os.path.basename(code_file)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test sırasında hata: {e}")
            return False

# Demo fonksiyonu
def demo_data_analysis():
    """Demo veri analizi"""
    print("🎮 DEMO: Veri Analizi Rehberi")
    
    # Sahte veri oluştur
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'salary': np.random.normal(50000, 15000, n_samples),
        'experience': np.random.normal(8, 4, n_samples).clip(0, None),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'education': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n_samples),
        'score': np.random.normal(75, 10, n_samples).clip(0, 100)
    }
    
    # Hedef değişken oluştur (örnek: yüksek maaşlı = 1, düşük = 0)
    salary_threshold = np.median([s for s in data['salary'] if not np.isnan(s)])
    target = []
    for i in range(n_samples):
        if np.isnan(data['salary'][i]):
            # Eksik maaş değeri için, yaş ve deneyime göre tahmin et
            prob = 0.5 + (data['age'][i] - 35) * 0.01 + (data['experience'][i] - 8) * 0.02
            target.append(1 if np.random.random() < prob else 0)
        else:
            target.append(1 if data['salary'][i] > salary_threshold else 0)
    
    data['high_earner'] = target  # Hedef değişken
    
    # Bazı eksik değerler ekle
    missing_indices = np.random.choice(n_samples, 50, replace=False)
    data['salary'] = [val if i not in missing_indices else np.nan 
                     for i, val in enumerate(data['salary'])]
    
    df = pd.DataFrame(data)
    DEMOFILE = os.path.join(CURRENT_DIR, 'demo_dataset.csv')
    df.to_csv(DEMOFILE, index=False)

    # Analizi başlat
    analyzer = DataAnalysisGuide(DEMOFILE)
    analyzer.data = df  # Demo için direkt set et
    
    print("\n1. Veri keşfi...")
    analyzer.load_and_explore_data(DEMOFILE)

    input("\nDevam etmek için ENTER tuşuna basınız...")
    
    print("\n2. Veri kalitesi analizi...")
    analyzer.analyze_data_quality()
    
    input("\nGörselleştirme için ENTER tuşuna basınız...")
    
    print("\n3. Veri görselleştirme...")
    analyzer.create_data_visualization_guide()
    
    input("\nÖn işleme önerileri için ENTER tuşuna basınız...")
    
    print("\n4. Ön işleme önerileri...")
    analyzer.preprocessing_recommendations()

if __name__ == "__main__":
    choice = input("""
🔍 Veri Analizi Rehberi
1. Demo çalıştır
2. Kendi veri dosyanızı analiz edin
Seçiminiz: """)
    
    if choice == "1":
        demo_data_analysis()
    elif choice == "2":
        file_path = input("Veri dosyası yolunu girin (.csv): ")
        analyzer = DataAnalysisGuide(file_path)
        if analyzer.load_and_explore_data(file_path):
            analyzer.analyze_data_quality()
            analyzer.create_data_visualization_guide()
            analyzer.preprocessing_recommendations()