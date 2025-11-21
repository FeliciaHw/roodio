import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

class KMeansClusterer(BaseEstimator, TransformerMixin):
    """
    Quadrant-Based KMeans Clusterer (Fixed Version).
    Membagi data ke 4 Kuadran (Happy, Angry, Sad, Relaxed),
    lalu melakukan clustering di dalam setiap kuadran.
    """

    def __init__(
        self,
        sub_clusters=3,
        features=["valence", "arousal", "dominance"],
        text_column="cleaned_lyrics",
        random_state=42,
        center_point=0.0
    ):
        self.sub_clusters = sub_clusters
        self.features = features
        self.text_column = text_column
        self.random_state = random_state
        self.center_point = center_point
        
        # Internal state
        self.scalers = {} 
        self.models = {}
        self.cluster_names = {}
        self.quadrants = ["Q1_Happy", "Q2_Angry", "Q3_Sad", "Q4_Relaxed"]

    def _get_quadrant(self, v, a):
        """Menentukan kuadran berdasarkan threshold center_point"""
        cp = self.center_point
        if v >= cp and a >= cp:
            return "Q1_Happy"
        elif v < cp and a >= cp:
            return "Q2_Angry"
        elif v < cp and a < cp:
            return "Q3_Sad"
        else:
            return "Q4_Relaxed"

    def fit(self, X, y=None):
        print(f"\n🔧 Memulai Quadrant-Based Clustering (Center Point: {self.center_point})...")
        
        X_temp = X.copy()
        X_temp['temp_quadrant'] = X_temp.apply(
            lambda row: self._get_quadrant(row['valence'], row['arousal']), axis=1
        )

        for q in self.quadrants:
            subset = X_temp[X_temp['temp_quadrant'] == q]
            
            if len(subset) < self.sub_clusters:
                # print(f"⚠ Warning: Kuadran {q} data terlalu sedikit ({len(subset)}). Skip.")
                self.models[q] = None
                continue

            # Scale data (hanya fitur numeric)
            scaler = StandardScaler()
            subset_feat = subset[self.features].values
            subset_scaled = scaler.fit_transform(subset_feat)
            self.scalers[q] = scaler

            # Fit KMeans
            kmeans = KMeans(
                n_clusters=self.sub_clusters,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(subset_scaled)
            self.models[q] = kmeans
            
            # === PERBAIKAN ERROR ADA DI SINI ===
            # 1. Pakai underscore di depan (_generate...)
            # 2. Pakai underscore di belakang labels (labels_)
            self._generate_subcluster_names(q, subset, kmeans.labels_)

        print("✅ All quadrants processed.")
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['quadrant'] = df.apply(
            lambda row: self._get_quadrant(row['valence'], row['arousal']), axis=1
        )
        
        df['kmeans_cluster'] = "Unclassified"
        df['kmeans_cluster_name'] = "Unclassified"

        for q in self.quadrants:
            if q not in self.models or self.models[q] is None:
                continue
                
            indices = df[df['quadrant'] == q].index
            if len(indices) == 0: continue

            subset_data = df.loc[indices, self.features].values
            subset_scaled = self.scalers[q].transform(subset_data)
            labels = self.models[q].predict(subset_scaled)
            
            # Assign Labels
            labeled_clusters = [f"{q}-{l}" for l in labels]
            named_clusters = [self.cluster_names.get(f"{q}-{l}", f"{q} Type {l+1}") for l in labels]

            df.loc[indices, 'kmeans_cluster'] = labeled_clusters
            df.loc[indices, 'kmeans_cluster_name'] = named_clusters

        print("\n📈 Final Cluster Distribution:")
        print(df['kmeans_cluster_name'].value_counts().sort_index())
        return df
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _generate_subcluster_names(self, quadrant, subset_df, labels):
        """Membuat nama unik untuk sub-cluster menggunakan TF-IDF"""
        try:
            tfidf = TfidfVectorizer(max_features=20, stop_words='english')
            text_data = subset_df[self.text_column].fillna("")
            
            if len(text_data) == 0:
                return

            tfidf_matrix = tfidf.fit_transform(text_data)
            feature_names = tfidf.get_feature_names_out()

            for i in range(self.sub_clusters):
                idx = np.where(labels == i)[0]
                if len(idx) == 0:
                    self.cluster_names[f"{quadrant}-{i}"] = f"{quadrant} Type {i+1}"
                    continue
                
                cluster_tfidf = tfidf_matrix[idx].mean(axis=0).A1
                top_idx = cluster_tfidf.argsort()[-1] 
                top_word = feature_names[top_idx]
                
                # Format nama: "Q1_Happy (Love)"
                name = f"{quadrant} ({top_word.title()})"
                self.cluster_names[f"{quadrant}-{i}"] = name
                
        except Exception:
            # Fallback name jika TF-IDF gagal
            for i in range(self.sub_clusters):
                self.cluster_names[f"{quadrant}-{i}"] = f"{quadrant} Type {i+1}"