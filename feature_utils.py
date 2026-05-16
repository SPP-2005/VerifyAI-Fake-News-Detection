"""
Shared components between model_trainer.py and app.py
"""

import re
import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def clean_text(text):
    """Advanced text cleaning pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextStatisticsExtractor(BaseEstimator, TransformerMixin):
    """Extracts numerical features from text for better classification."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            if not isinstance(text, str):
                text = ""
            words = text.split()
            word_count = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            sentence_count = max(len(re.split(r'[.!?]+', text)) - 1, 1)
            exclamation_count = text.count('!')
            question_count = text.count('?')
            alpha_chars = [c for c in text if c.isalpha()]
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
            punct_count = sum(1 for c in text if c in string.punctuation)
            punct_ratio = punct_count / max(len(text), 1)
            all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            char_count = len(text)
            unique_ratio = len(set(words)) / max(word_count, 1)
            has_quotes = 1 if '"' in text or "'" in text else 0
            number_count = len(re.findall(r'\d+', text))

            features.append([
                word_count, avg_word_len, sentence_count,
                exclamation_count, question_count, caps_ratio,
                punct_ratio, all_caps_words, char_count,
                unique_ratio, has_quotes, number_count
            ])
        return np.array(features)
