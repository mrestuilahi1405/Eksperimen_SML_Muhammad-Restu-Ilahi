import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

def run_preprocessing(input_path, output_path):
    print(f"🚀 Memulai preprocessing dari: {input_path}")
    
    # Load Data
    df = pd.read_csv(input_path)
    
    # 1. Handling Missing Values & Data Duplikat (Menjaga data tetap 'riil')
    df_clean = df.dropna().drop_duplicates().copy()
    
    # 2. Deteksi dan Penanganan Outlier (Logical Filtering)
    # Dilakukan sebelum scaling agar statistik tidak bias
    df_clean = df_clean[df_clean['person_age'] <= 90]
    df_clean = df_clean[df_clean['person_emp_length'] <= 50]
    df_clean = df_clean[df_clean['person_income'] <= 1000000]
    
    # 3. Feature Encoding
    # A. Ordinal Encoding untuk loan_grade (A=1, ..., G=7)
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df_clean['loan_grade'] = df_clean['loan_grade'].map(grade_mapping)
    
    # B. Binary Encoding untuk Default History
    df_clean['cb_person_default_on_file'] = df_clean['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
    
    # C. One-Hot Encoding untuk fitur Nominal (loan_intent, person_home_ownership)
    nominal_cols = ['loan_intent', 'person_home_ownership']
    df_clean = pd.get_dummies(df_clean, columns=nominal_cols)
    
    # 4. Feature Scaling (Standardization)
    # Memisahkan fitur dan target sebelum scaling
    X = df_clean.drop('loan_status', axis=1)
    y = df_clean['loan_status']
    
    scaler = RobustScaler()
    X_final = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Gabungkan kembali dengan target
    df_final = pd.concat([X_final, y.reset_index(drop=True)], axis=1)
    
    # 5. Simpan Hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai! Output: {output_path} | Total Data: {len(df_final)}")

if __name__ == "__main__":
    # Path disesuaikan dengan struktur repositori GitHub
    INPUT_FILE = "creditrisk_raw.csv" 
    OUTPUT_FILE = "preprocessing/creditrisk_preprocessing.csv"
    
    if os.path.exists(INPUT_FILE):
        run_preprocessing(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"❌ Error: File {INPUT_FILE} tidak ditemukan!")