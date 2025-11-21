# feature_extractor.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RobustFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extractor TF-IDF yang robust dan Pipeline-Friendly:
    - Handles NaN gracefully.
    - Bisa mengembalikan DataFrame (mempertahankan Index).
    - Bisa menggabungkan hasil dengan data asli (agar kolom lain tidak hilang).
    """
    
    def _init_(self, text_column='cleaned_lyrics',
                 max_features=1000,  # Dikurangi agar tidak terlalu berat saat concat
                 min_df=2, 
                 max_df=0.95, 
                 ngram_range=(1, 2),
                 output_format='dataframe'): # Options: 'numpy', 'dataframe', 'concat'
        
        self.text_column = text_column
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.output_format = output_format
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=ngram_range
        )
        
        self.fitted_ = False
        self.feature_names_ = []

    # =========================================================
    # FIT
    # =========================================================
    def fit(self, X, y=None):
        # Validasi input berupa DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("❌ Input X harus berupa pandas DataFrame")

        if self.text_column not in X.columns:
            raise ValueError(f"❌ Kolom '{self.text_column}' tidak ditemukan di DataFrame")

        # Ambil hanya teks valid untuk fitting
        # Copy safe slice
        text_series = X[self.text_column].fillna("").astype(str)
        mask_valid = text_series.str.strip() != ""
        valid_texts = text_series[mask_valid]

        if len(valid_texts) == 0:
            print("⚠ Warning: Tidak ada data valid untuk fit TF-IDF. Vocabulary akan kosong.")
            self.fitted_ = True
            return self

        print(f"🔧 Fitting TF-IDF on {len(valid_texts)} valid samples...")
        
        try:
            self.tfidf.fit(valid_texts)
            self.fitted_ = True
            
            # Simpan nama fitur untuk keperluan DataFrame column names
            if hasattr(self.tfidf, "get_feature_names_out"):
                self.feature_names_ = self.tfidf.get_feature_names_out()
            else:
                self.feature_names_ = self.tfidf.get_feature_names()
                
            print(f"✅ TF-IDF fitting completed! Vocab size: {len(self.feature_names_)}")
            
        except ValueError as e:
            print(f"⚠ TF-IDF fit failed (mungkin vocabulary kosong): {e}")
            self.fitted_ = False

        return self

    # =========================================================
    # TRANSFORM
    # =========================================================
    def transform(self, X):
        if not self.fitted_:
            print("⚠ Feature extractor belum di-fit atau vocab kosong. Returning empty/original data.")
            if self.output_format == 'concat':
                return X
            elif self.output_format == 'dataframe':
                return pd.DataFrame(index=X.index)
            else:
                return np.array([])

        X_tr = X.copy()
        
        # Isi NaN dengan string kosong agar jumlah baris TETAP SAMA
        texts = X_tr[self.text_column].fillna("").astype(str)

        print(f"🔧 Transforming {len(texts)} samples with TF-IDF...")

        # Transform
        tfidf_matrix = self.tfidf.transform(texts)
        
        # --- LOGIC HANDLING OUTPUT FORMAT ---
        
        # 1. Jika user minta Numpy Array (Standar Sklearn)
        if self.output_format == 'numpy':
            return tfidf_matrix.toarray()

        # Convert ke DataFrame dengan nama kolom yang benar & Index yang sesuai
        # Prefix 'tfidf_' agar tidak bentrok dengan nama kolom lain
        feature_names = [f"tfidf_{name}" for name in self.feature_names_]
        
        # Gunakan sparse dataframe jika memungkinkan untuk hemat memori (opsional), 
        # tapi untuk kompatibilitas kita pakai dense dulu.
        df_tfidf = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=X.index  # PENTING: Menjaga index tetap sinkron dengan data asli
        )
        
        # 2. Jika user minta DataFrame (hanya fitur TF-IDF)
        if self.output_format == 'dataframe':
            return df_tfidf
            
        # 3. Jika user minta Concat (Data Asli + Fitur TF-IDF) -> INI YANG COCOK UNTUK PIPELINE
        elif self.output_format == 'concat':
            print(f"   🔗 Concatenating original data ({X.shape[1]} cols) + TF-IDF ({len(feature_names)} cols)...")
            df_combined = pd.concat([X, df_tfidf], axis=1)
            return df_combined
            
        else:
            raise ValueError(f"Unknown output_format: {self.output_format}")

    # =========================================================
    def get_feature_names(self):
        return self.feature_names_