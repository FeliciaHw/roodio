# preprocessText.py
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import nltk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial

# Download NLTK resources sekali saja di level module
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class LyricsPreprocessor(BaseEstimator, TransformerMixin):
    """
    Class untuk preprocessing lyrics yang dioptimasi untuk performa dengan FIX NaN handling
    """
    
    def __init__(self, text_column='lyrics', cleaned_column='cleaned_lyrics', 
                 language='english', min_word_length=2, remove_sections=True,
                 n_jobs=-1, batch_size=1000, remove_nan=True):
        # Parameters untuk sklearn Pipeline
        self.text_column = text_column
        self.cleaned_column = cleaned_column
        
        # Parameters untuk preprocessing
        self.language = language
        self.min_word_length = min_word_length
        self.remove_sections = remove_sections
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.batch_size = batch_size
        self.remove_nan = remove_nan
        
        # Setup NLP components
        self._setup_nltk_resources()
        
        # Pre-compile regex patterns untuk performa
        self._compile_patterns()
        
        print(f"✅ LyricsPreprocessor initialized! (n_jobs: {self.n_jobs}, remove_nan: {self.remove_nan})")

    def _setup_nltk_resources(self):
        """Setup NLTK resources dengan caching"""
        try:
            self.stop_words = set(stopwords.words(self.language))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"⚠️ Warning: NLTK setup failed: {e}")
            self.stop_words = set()
            self.lemmatizer = WordNetLemmatizer()

    def _compile_patterns(self):
        """Pre-compile semua regex patterns untuk performa"""
        # Non-lyric sections to remove
        self.non_lyric_sections = [
            "intro", "outro", "chorus", "refrain", "verse", "bridge",
            "pre-chorus", "post-chorus", "instrumental", "coda", 
            "interlude", "reff", "hook", "solo", "break", "drop"
        ]
        
        # Pre-compile regex patterns
        self.patterns_to_remove = [
            re.compile(r'\[.*?\]'),      # [Chorus]
            re.compile(r'\(.*?\)'),      # (Verse 1)
            re.compile(r'\{.*?\}'),      # {Intro}
            re.compile(r'<.*?>'),        # <Bridge>
            re.compile(r'x\d+'),         # x2, x3 (repetition indicators)
            re.compile(r'\b\w\b'),       # single characters
            re.compile(r'\b\d+\b'),      # standalone numbers
            re.compile(r'\b\w*\d\w*\b'), # words with numbers
            re.compile(r'\.{2,}'),       # multiple dots
            re.compile(r'-{2,}'),        # multiple dashes
            re.compile(r'\*{2,}'),       # multiple asterisks
            re.compile(r'_{2,}'),        # multiple underscores
        ]
        
        # Additional compiled patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.special_char_pattern = re.compile(r'[^\w\s.,!?]')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.section_repeat_pattern = re.compile(r'(?i)(chorus|verse|bridge|intro|outro)\s*\d*\s*repeat')
        
        # Section patterns pre-compiled
        self.section_patterns = []
        for section in self.non_lyric_sections:
            self.section_patterns.extend([
                re.compile(f"\\b{section}\\b", re.IGNORECASE),
                re.compile(f"\\b{section}\\s*\\d*\\b", re.IGNORECASE),
                re.compile(f"\\b{section}\\s*:", re.IGNORECASE),
                re.compile(f"\\b{section}\\s*-", re.IGNORECASE),
            ])

    def _remove_nan_before_preprocessing(self, df):
        """Hapus rows dengan nilai NaN di text_column SEBELUM preprocessing"""
        if not self.remove_nan:
            return df
            
        initial_count = len(df)
        
        # Cek dan hapus NaN values di kolom teks
        nan_mask = df[self.text_column].isna()
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            df_clean = df[~nan_mask].copy()
            print(f"🗑️  REMOVED BEFORE PREPROCESSING: {nan_count} rows with NaN values in '{self.text_column}'")
            print(f"📊 Data after NaN removal: {len(df_clean):,} rows ({len(df_clean)/initial_count*100:.1f}% of original)")
            return df_clean
        else:
            print("✅ No NaN values found in the text column (before preprocessing)")
            return df

    def _remove_nan_after_preprocessing(self, df):
        """Hapus rows dengan nilai NaN di cleaned_column SETELAH preprocessing"""
        if not self.remove_nan:
            return df
            
        initial_count = len(df)
        
        # Cek dan hapus NaN values di kolom cleaned
        nan_mask = df[self.cleaned_column].isna()
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            df_clean = df[~nan_mask].copy()
            print(f"🗑️  REMOVED AFTER PREPROCESSING: {nan_count} rows with NaN values in '{self.cleaned_column}'")
            print(f"📊 Final data after cleaning: {len(df_clean):,} rows ({len(df_clean)/initial_count*100:.1f}% of preprocessed)")
            return df_clean
        else:
            print("✅ No NaN values found in the cleaned column (after preprocessing)")
            return df

    def _final_nan_cleanup(self, df):
        """Final cleanup untuk memastikan tidak ada NaN"""
        initial_count = len(df)
        
        # Hapus semua row yang memiliki NaN di kolom cleaned_lyrics
        nan_mask = df[self.cleaned_column].isna()
        df_clean = df[~nan_mask].copy()
        
        removed_count = nan_mask.sum()
        if removed_count > 0:
            print(f"🔧 FINAL CLEANUP: Removed {removed_count} rows with NaN in '{self.cleaned_column}'")
            print(f"📊 Final dataset: {len(df_clean):,} rows ({len(df_clean)/initial_count*100:.1f}% of original)")
        
        return df_clean

    def _simple_tokenize(self, text):
        """Optimized simple tokenizer"""
        return text.split()

    def fit(self, X, y=None):
        """Required untuk sklearn Transformer"""
        return self

    def transform(self, X, y=None):
        """Transform method yang dioptimasi untuk sklearn Pipeline"""
        df = X.copy()
        
        # STEP 0: HAPUS NILAI NaN SEBELUM PREPROCESSING
        print("🔍 Checking for NaN values BEFORE preprocessing...")
        df = self._remove_nan_before_preprocessing(df)
        if len(df) == 0:
            print("❌ No data remaining after NaN removal!")
            return df
        
        print("🔄 Cleaning lyrics...")
        
        # Gunakan parallel processing untuk dataset besar
        if len(df) > self.batch_size and self.n_jobs > 1:
            df[self.cleaned_column] = self._parallel_preprocess(df[self.text_column])
        else:
            df[self.cleaned_column] = df[self.text_column].apply(self.preprocess_lyrics)
        
        # STEP 1: HAPUS NILAI NaN SETELAH PREPROCESSING
        print("🔍 Checking for NaN values AFTER preprocessing...")
        df = self._remove_nan_after_preprocessing(df)
        
        # STEP 2: FINAL CLEANUP - PASTIKAN TIDAK ADA NAN
        print("🔍 Final NaN cleanup...")
        df = self._final_nan_cleanup(df)
        
        if len(df) == 0:
            print("❌ No data remaining after final cleanup!")
            return df
        
        # Hitung panjang lirik
        df['lyrics_length'] = df[self.cleaned_column].apply(lambda x: len(x.split()) if x else 0)
        
        # Filter lirik yang terlalu pendek DAN yang kosong
        initial_count = len(df)
        mask = (df['lyrics_length'] >= 5) & (df[self.cleaned_column].str.len() > 0)
        df = df[mask]
        final_count = len(df)
        
        print(f"✅ Lyrics cleaned: {final_count}/{initial_count} songs remaining")
        print(f"🗑️  Removed {initial_count - final_count} empty/short lyrics")
        
        # FINAL VALIDATION: Pastikan benar-benar tidak ada NaN
        final_nan_count = df[self.cleaned_column].isna().sum()
        if final_nan_count > 0:
            print(f"⚠️  CRITICAL: Still found {final_nan_count} NaN values, forcing removal...")
            df = df[~df[self.cleaned_column].isna()]
        
        print(f"🎯 FINAL DATASET: {len(df):,} clean rows ready for processing")
        print(f"🔍 FINAL NaN CHECK: {df[self.cleaned_column].isna().sum()} NaN values")
        
        return df

    def _parallel_preprocess(self, series):
        """Parallel processing untuk preprocessing"""
        print(f"⚡ Using parallel processing ({self.n_jobs} workers)...")
        
        # Bagi data menjadi chunks
        chunks = np.array_split(series, len(series) // self.batch_size + 1)
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(self._process_chunk, chunks))
        
        return pd.concat(results)

    def _process_chunk(self, chunk):
        """Process a chunk of data"""
        return chunk.apply(self.preprocess_lyrics)

    def clean_text(self, text):
        """Optimized text cleaning dengan handling NaN yang lebih robust"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        
        # Gunakan pre-compiled patterns
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = self.special_char_pattern.sub(' ', text)
        text = self.number_pattern.sub('', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text

    def tokenize_and_lemmatize(self, text):
        """Optimized tokenize and lemmatize dengan NaN handling"""
        if not text or text.strip() == "" or pd.isna(text):
            return []
            
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = self._simple_tokenize(text)
        
        processed_tokens = []
        for token in tokens:
            # Fast filtering
            if (len(token) <= self.min_word_length or 
                token in self.stop_words or
                token in string.punctuation or
                not any(c.isalpha() for c in token)):
                continue
            
            try:
                # Combined lemmatization
                lemma = self.lemmatizer.lemmatize(token, pos='n')
                lemma = self.lemmatizer.lemmatize(lemma, pos='v')
                processed_tokens.append(lemma)
            except:
                processed_tokens.append(token)
        
        return processed_tokens

    def clean_lyrics(self, lyrics):
        """Optimized remove non-lyric sections dengan NaN handling"""
        if not isinstance(lyrics, str) or pd.isna(lyrics):
            return ""
        
        lyrics = lyrics.lower()
        
        # Gunakan pre-compiled section patterns
        for pattern in self.section_patterns:
            lyrics = pattern.sub('', lyrics)
        
        # Remove lines that are just section markers
        lines = lyrics.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                # Cek cepat apakah line mengandung section markers
                is_section_line = any(
                    section in line_stripped for section in self.non_lyric_sections
                )
                if not is_section_line:
                    cleaned_lines.append(line_stripped)
        
        return ' '.join(cleaned_lines)

    def remove_special_patterns(self, text):
        """Optimized remove special patterns dengan NaN handling"""
        if pd.isna(text):
            return ""
            
        for pattern in self.patterns_to_remove:
            text = pattern.sub('', text)
        
        text = self.section_repeat_pattern.sub('', text)
        return text

    def remove_repeated_words(self, text, max_repeat=2):
        """Optimized remove repeated words dengan NaN handling"""
        if pd.isna(text) or not text:
            return text
            
        words = text.split()
        if len(words) < 2:
            return text
            
        cleaned_words = []
        prev_word = None
        repeat_count = 0
        
        for word in words:
            if word == prev_word:
                repeat_count += 1
                if repeat_count <= max_repeat:
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)
                prev_word = word
                repeat_count = 0
        
        return ' '.join(cleaned_words)

    def remove_extra_spaces(self, text):
        """Optimized remove extra spaces dengan NaN handling"""
        if pd.isna(text):
            return ""
            
        text = self.whitespace_pattern.sub(' ', text)
        text = re.sub(r'\s+\.\s+', '. ', text)
        text = re.sub(r'\s+,\s+', ', ', text)
        text = re.sub(r'\s+!\s+', '! ', text)
        text = re.sub(r'\s+\?\s+', '? ', text)
        return text.strip()

    def clean_contractions(self, text):
        """Optimized expand contractions dengan NaN handling"""
        if pd.isna(text) or len(text) <= 10:
            return text
            
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "can't": "cannot", "couldn't": "could not",
            "shouldn't": "should not", "wouldn't": "would not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
            "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "that's": "that is", "what's": "what is", "who's": "who is", "where's": "where is",
            "when's": "when is", "why's": "why is", "how's": "how is", "let's": "let us"
        }
        
        for contraction, expansion in contractions.items():
            if contraction in text:
                text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        return text

    def normalize_text(self, text):
        """Optimized normalize text dengan NaN handling"""
        if pd.isna(text):
            return ""
            
        text = text.lower()
        text = self.html_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
        return text

    def preprocess_lyrics(self, lyrics):
        """Optimized main preprocessing pipeline dengan robust NaN handling"""
        if pd.isna(lyrics) or lyrics == "":
            return ""
        
        try:
            # Step 1: Normalize text
            text = self.normalize_text(lyrics)
            
            # Step 2: Expand contractions (skip jika text pendek)
            if len(text) > 10:
                text = self.clean_contractions(text)
            
            # Step 3: Remove special patterns
            text = self.remove_special_patterns(text)
            
            # Step 4: Remove non-lyric sections (skip jika text pendek)
            if len(text) > 20 and self.remove_sections:
                text = self.clean_lyrics(text)
            
            # Step 5: Basic cleaning
            text = self.clean_text(text)
            
            # Step 6: Remove repeated words (skip jika text pendek)
            if len(text) > 30:
                text = self.remove_repeated_words(text)
            
            # Step 7: Remove extra spaces
            text = self.remove_extra_spaces(text)
            
            # Step 8: Tokenize and lemmatize (skip jika text sangat pendek)
            if len(text) > 3:
                tokens = self.tokenize_and_lemmatize(text)
                result = ' '.join(tokens)
            else:
                result = text
            
            # Skip jika hasilnya kosong atau NaN
            if not result or result.strip() == "" or pd.isna(result):
                return ""
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error preprocessing lyrics: {e}")
            return ""

    def preprocess_dataframe(self, df, lyrics_column='lyrics', new_column='cleaned_lyrics', 
                           save_path=None, save_format='parquet'):
        """Optimized Preprocess entire dataframe dengan NaN removal"""
        print("🔄 Memulai preprocessing lyrics...")
        
        df_processed = df.copy()
        
        # Hapus NaN sebelum preprocessing
        print("🔍 Checking for NaN values BEFORE preprocessing...")
        initial_total = len(df_processed)
        nan_mask = df_processed[lyrics_column].isna()
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            df_processed = df_processed[~nan_mask].copy()
            print(f"🗑️  Removed {nan_count} rows with NaN values before preprocessing")
        
        # Gunakan parallel processing untuk dataset besar
        if len(df_processed) > self.batch_size and self.n_jobs > 1:
            df_processed[new_column] = self._parallel_preprocess(df_processed[lyrics_column])
        else:
            df_processed[new_column] = df_processed[lyrics_column].apply(self.preprocess_lyrics)
        
        # Hapus NaN setelah preprocessing
        print("🔍 Checking for NaN values AFTER preprocessing...")
        post_nan_mask = df_processed[new_column].isna()
        post_nan_count = post_nan_mask.sum()
        
        if post_nan_count > 0:
            df_processed = df_processed[~post_nan_mask].copy()
            print(f"🗑️  Removed {post_nan_count} rows with NaN values after preprocessing")
        
        df_processed['lyrics_length'] = df_processed[new_column].apply(lambda x: len(x.split()))
        
        # Filter
        initial_count = len(df_processed)
        mask = (df_processed['lyrics_length'] >= 5) & (df_processed[new_column].str.len() > 0)
        df_processed = df_processed[mask]
        final_count = len(df_processed)
        
        print("✅ Preprocessing selesai!")
        self._print_preprocessing_stats(df_processed, new_column, initial_total, final_count)
        
        if save_path:
            self._save_preprocessing_results(df_processed, save_path, save_format)
        
        return df_processed

    def _print_preprocessing_stats(self, df, cleaned_column, initial_count, final_count):
        """Print statistics"""
        lyrics_lengths = df[cleaned_column].apply(lambda x: len(x.split()))
        
        print(f"\n📊 Statistik Preprocessing:")
        print(f"Jumlah lagu awal: {initial_count}")
        print(f"Jumlah lagu akhir: {final_count}")
        print(f"Lagu dihapus: {initial_count - final_count}")
        print(f"Rata-rata panjang lirik: {lyrics_lengths.mean():.1f} kata")
        print(f"Total kata: {lyrics_lengths.sum():,} kata")
        
        if len(df) > 0:
            print(f"\n🔍 Contoh Preprocessing:")
            original_sample = str(df.iloc[0]['lyrics'])[:100] + "..." 
            cleaned_sample = df.iloc[0][cleaned_column][:100] + "..."
            print(f"Sebelum: {original_sample}")
            print(f"Sesudah: {cleaned_sample}")

    def _save_preprocessing_results(self, df, save_path, save_format='parquet'):
        """Save hasil preprocessing"""
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_format == 'parquet':
            df.to_parquet(save_path, index=False)
        elif save_format == 'csv':
            df.to_csv(save_path, index=False)
        elif save_format == 'json':
            df.to_json(save_path, orient='records', indent=2)
        else:
            print(f"❌ Format {save_format} tidak didukung")
            return
            
        print(f"💾 Hasil preprocessing disimpan sebagai: {save_path}")

    def save_preprocessor(self, filepath):
        """Save preprocessor object"""
        joblib.dump(self, filepath)
        print(f"✅ Preprocessor disimpan sebagai: {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath):
        """Load preprocessor object"""
        preprocessor = joblib.load(filepath)
        print(f"✅ Preprocessor loaded dari: {filepath}")
        return preprocessor