from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    TF-IDF feature extractor for sklearn pipelines.
    Handles missing values and provides multiple output formats.
    """
    
    def __init__(self, text_column='cleaned_lyrics',
                 max_features=1000,
                 min_df=2, 
                 max_df=0.95, 
                 ngram_range=(1, 2),
                 output_format='dataframe'):
        
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

    def fit(self, X, y=None):
        """Fit TF-IDF vectorizer on valid text data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be pandas DataFrame")

        if self.text_column not in X.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame")

        # Get valid texts for fitting
        text_series = X[self.text_column].fillna("").astype(str)
        mask_valid = text_series.str.strip() != ""
        valid_texts = text_series[mask_valid]

        if len(valid_texts) == 0:
            print("Warning: No valid data for TF-IDF fitting.")
            self.fitted_ = True
            return self

        # Fit TF-IDF vectorizer
        self.tfidf.fit(valid_texts)
        self.fitted_ = True
        
        # Store feature names
        if hasattr(self.tfidf, "get_feature_names_out"):
            self.feature_names_ = self.tfidf.get_feature_names_out()
        else:
            self.feature_names_ = self.tfidf.get_feature_names()
                
        return self

    def transform(self, X):
        """Transform text data to TF-IDF features."""
        if not self.fitted_:
            if self.output_format == 'concat':
                return X
            elif self.output_format == 'dataframe':
                return pd.DataFrame(index=X.index)
            else:
                return np.array([])

        X_tr = X.copy()
        
        # Handle missing values
        texts = X_tr[self.text_column].fillna("").astype(str)

        # Transform texts to TF-IDF matrix
        tfidf_matrix = self.tfidf.transform(texts)
        
        # Handle different output formats
        if self.output_format == 'numpy':
            return tfidf_matrix.toarray()

        # Create feature names with prefix
        feature_names = [f"tfidf_{name}" for name in self.feature_names_]
        
        # Create DataFrame with original index
        df_tfidf = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=X.index
        )
        
        if self.output_format == 'dataframe':
            return df_tfidf
            
        elif self.output_format == 'concat':
            df_combined = pd.concat([X, df_tfidf], axis=1)
            return df_combined
            
        else:
            raise ValueError(f"Unknown output_format: {self.output_format}")

    def get_feature_names(self):
        """Return TF-IDF feature names."""
        return self.feature_names_