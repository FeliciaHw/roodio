import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial


def extract_vad_single(text, vad_dict):
    """Extract VAD features for a single text."""
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


class VADFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract Valence-Arousal-Dominance features from text using lexicon."""

    def __init__(self, vad_lexicon_path, lyrics_column='cleaned_lyrics',
                 n_jobs=-1, batch_size=1000):
        self.vad_lexicon_path = vad_lexicon_path
        self.lyrics_column = lyrics_column
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.batch_size = batch_size
        self.vad_dict = self._load_vad_lexicon()

    def _load_vad_lexicon(self):
        """Load VAD lexicon from TSV file."""
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

        return vad_dict

    def fit(self, X, y=None):
        """Fit method for sklearn compatibility."""
        return self

    def transform(self, X, y=None):
        """Transform text data to VAD features."""
        df = X.copy()
        series = df[self.lyrics_column]

        # Extract features
        if len(series) > self.batch_size and self.n_jobs > 1:
            features_list = self._parallel_extract(series)
        else:
            features_list = [
                extract_vad_single(text, self.vad_dict)
                for text in series
            ]

        # Create feature DataFrame
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

        # Combine with original data
        df = pd.concat([df.reset_index(drop=True),
                        feature_df.reset_index(drop=True)],
                       axis=1)

        return df

    def _parallel_extract(self, series):
        """Extract features in parallel for large datasets."""
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