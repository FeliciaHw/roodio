# ========================================================================
# valenceArousal.py — FINAL VERSION (NO ROW MISMATCH, COMPLETE FEATURES)
# ========================================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial


# ================================================================
# GLOBAL FUNCTION (WAJIB untuk multiprocessing)
# ================================================================
def extract_vad_single(text, vad_dict):
    """Extractor NON-OOP, aman untuk multiprocessing."""
    if not text or not isinstance(text, str) or text.strip() == "":
        return [0.5, 0.5, 0.5, 0.0, 0]

    words = text.split()
    total_words = len(words)

    v_scores = []
    a_scores = []
    d_scores = []
    n_matched = 0

    for word in words:
        if word in vad_dict:
            v, a, d = vad_dict[word]
            v_scores.append(v)
            a_scores.append(a)
            d_scores.append(d)
            n_matched += 1

    if n_matched > 0:
        return [
            float(np.mean(v_scores)),
            float(np.mean(a_scores)),
            float(np.mean(d_scores)),
            n_matched / total_words,
            n_matched
        ]
    else:
        return [0.5, 0.5, 0.5, 0.0, 0]


# ================================================================
# MAIN CLASS
# ================================================================
class VADFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vad_lexicon_path, lyrics_column='cleaned_lyrics',
                 n_jobs=-1, batch_size=1000):

        self.vad_lexicon_path = vad_lexicon_path
        self.lyrics_column = lyrics_column
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.batch_size = batch_size
        self.vad_dict = self._load_vad_lexicon()

    # ------------------------------------------------------------
    def _load_vad_lexicon(self):
        print("📚 Loading VAD lexicon...")

        df_vad = pd.read_csv(
            self.vad_lexicon_path,
            delimiter='\t',
            usecols=['term', 'valence', 'arousal', 'dominance'],
            dtype={
                'term': 'string',
                'valence': 'float32',
                'arousal': 'float32',
                'dominance': 'float32'
            }
        )

        vad_dict = {
            row['term']: (row['valence'], row['arousal'], row['dominance'])
            for _, row in df_vad.iterrows()
        }

        print(f"✅ VAD lexicon loaded: {len(vad_dict)} terms")
        return vad_dict

    # ------------------------------------------------------------
    def fit(self, X, y=None):
        return self

    # ------------------------------------------------------------
    def transform(self, X, y=None):
        df = X.copy()

        print("🎯 Extracting VAD features...")

        series = df[self.lyrics_column]

        # Parallel mode
        if len(series) > self.batch_size and self.n_jobs > 1:
            features_list = self._parallel_extract(series)
        else:
            features_list = [
                extract_vad_single(text, self.vad_dict)
                for text in series
            ]

        # Buat dataframe fitur
        feature_df = pd.DataFrame(
            features_list,
            columns=['valence', 'arousal', 'dominance', 'coverage_ratio', 'n_matched_words']
        )

        feature_df = feature_df.astype({
            'valence': 'float32',
            'arousal': 'float32',
            'dominance': 'float32',
            'coverage_ratio': 'float32',
            'n_matched_words': 'int32'
        })

        # Gabung ke DF original
        df = pd.concat([df.reset_index(drop=True),
                        feature_df.reset_index(drop=True)],
                       axis=1)

        self._print_stats(feature_df)

        return df

    # ------------------------------------------------------------
    def _parallel_extract(self, series):
        print(f"⚡ Using parallel processing with {self.n_jobs} workers...")

        chunks = [
            series[i:i + self.batch_size]
            for i in range(0, len(series), self.batch_size)
        ]

        func = partial(extract_vad_single, vad_dict=self.vad_dict)
        results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for chunk_result in executor.map(
                    lambda chunk: [func(text) for text in chunk],
                    chunks):
                results.extend(chunk_result)

        return results

    # ------------------------------------------------------------
    def _print_stats(self, fdf):
        print("✅ VAD features extracted")
        print(f"   Valence range: [{fdf['valence'].min():.3f}, {fdf['valence'].max():.3f}]")
        print(f"   Arousal range: [{fdf['arousal'].min():.3f}, {fdf['arousal'].max():.3f}]")
        print(f"   Dominance range: [{fdf['dominance'].min():.3f}, {fdf['dominance'].max():.3f}]")
        print(f"   Avg coverage: {fdf['coverage_ratio'].mean():.3f}")
        print(f"   Avg matched words: {fdf['n_matched_words'].mean():.2f}")

        nom = (fdf['n_matched_words'] == 0).sum()
        if nom > 0:
            print(f"⚠️ {nom} texts ({nom/len(fdf)*100:.1f}%) have no VAD matches")