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

# Download NLTK resources
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
    """Lyrics preprocessing pipeline for sklearn compatibility."""
    
    def __init__(self, text_column='lyrics', cleaned_column='cleaned_lyrics', 
                 language='english', min_word_length=2, remove_sections=True):
        self.text_column = text_column
        self.cleaned_column = cleaned_column
        self.language = language
        self.min_word_length = min_word_length
        self.remove_sections = remove_sections
        
        self._setup_nltk_resources()
        self._compile_patterns()

    def _setup_nltk_resources(self):
        """Initialize NLTK resources."""
        try:
            self.stop_words = set(stopwords.words(self.language))
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.stop_words = set()
            self.lemmatizer = WordNetLemmatizer()

    def _compile_patterns(self):
        """Pre-compile regex patterns."""
        self.non_lyric_sections = [
            "intro", "outro", "chorus", "refrain", "verse", "bridge",
            "pre-chorus", "post-chorus", "instrumental"
        ]
        
        self.patterns_to_remove = [
            re.compile(r'\[.*?\]'),
            re.compile(r'\(.*?\)'),
            re.compile(r'\{.*?\}'),
            re.compile(r'<.*?>'),
            re.compile(r'x\d+'),
            re.compile(r'\b\w\b'),
            re.compile(r'\b\d+\b'),
        ]
        
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.special_char_pattern = re.compile(r'[^\w\s.,!?]')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        self.section_patterns = []
        for section in self.non_lyric_sections:
            self.section_patterns.extend([
                re.compile(f"\\b{section}\\b", re.IGNORECASE),
                re.compile(f"\\b{section}\\s*\\d*\\b", re.IGNORECASE),
            ])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        
        # Remove NaN values
        df = df[~df[self.text_column].isna()]
        if len(df) == 0:
            return df
        
        # Clean lyrics
        df[self.cleaned_column] = df[self.text_column].apply(self.preprocess_lyrics)
        
        # Remove empty results
        df = df[~df[self.cleaned_column].isna()]
        df = df[df[self.cleaned_column].str.len() > 0]
        
        return df

    def clean_text(self, text):
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = self.url_pattern.sub('', text)
        text = self.special_char_pattern.sub(' ', text)
        text = self.number_pattern.sub('', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text."""
        if not text or pd.isna(text):
            return []
            
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            if (len(token) <= self.min_word_length or 
                token in self.stop_words or
                token in string.punctuation):
                continue
            
            try:
                lemma = self.lemmatizer.lemmatize(token, pos='n')
                lemma = self.lemmatizer.lemmatize(lemma, pos='v')
                processed_tokens.append(lemma)
            except:
                processed_tokens.append(token)
        
        return processed_tokens

    def clean_lyrics(self, lyrics):
        """Remove non-lyric sections."""
        if not isinstance(lyrics, str):
            return ""
        
        lyrics = lyrics.lower()
        
        for pattern in self.section_patterns:
            lyrics = pattern.sub('', lyrics)
        
        lines = lyrics.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                is_section_line = any(
                    section in line_stripped for section in self.non_lyric_sections
                )
                if not is_section_line:
                    cleaned_lines.append(line_stripped)
        
        return ' '.join(cleaned_lines)

    def remove_special_patterns(self, text):
        """Remove special patterns."""
        if pd.isna(text):
            return ""
            
        for pattern in self.patterns_to_remove:
            text = pattern.sub('', text)
        
        return text

    def remove_extra_spaces(self, text):
        """Clean up extra whitespace."""
        if pd.isna(text):
            return ""
            
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()

    def preprocess_lyrics(self, lyrics):
        """Main preprocessing pipeline."""
        if pd.isna(lyrics) or lyrics == "":
            return ""
        
        try:
            # Step 1: Basic normalization
            text = str(lyrics).lower()
            
            # Step 2: Remove special patterns
            text = self.remove_special_patterns(text)
            
            # Step 3: Remove non-lyric sections
            if self.remove_sections:
                text = self.clean_lyrics(text)
            
            # Step 4: Basic cleaning
            text = self.clean_text(text)
            
            # Step 5: Remove extra spaces
            text = self.remove_extra_spaces(text)
            
            # Step 6: Tokenize and lemmatize
            if len(text) > 3:
                tokens = self.tokenize_and_lemmatize(text)
                result = ' '.join(tokens)
            else:
                result = text
            
            return result if result and result.strip() != "" else ""
            
        except Exception:
            return ""

    def save_preprocessor(self, filepath):
        """Save preprocessor object."""
        joblib.dump(self, filepath)

    @classmethod
    def load_preprocessor(cls, filepath):
        """Load preprocessor object."""
        return joblib.load(filepath)