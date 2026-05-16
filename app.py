"""
Enhanced Fake News Detection API
---------------------------------
Multi-signal analysis backend with calibrated ensemble model,
text statistics, sensationalism scoring, and detailed breakdowns.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import re
import string
import os
import numpy as np
from datetime import datetime
from feature_utils import clean_text, TextStatisticsExtractor

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

model = joblib.load("fake_news_model.pkl")
print("Model loaded successfully")

SENSATIONAL_KEYWORDS = {
    "high": [
        "shocking", "bombshell", "exposed", "leaked",
        "conspiracy", "cover-up", "coverup", "scandal", "explosive",
        "devastating", "unbelievable", "hoax", "plandemic",
        "they don't want you to know", "mainstream media won't tell",
        "wake up", "sheeple"
    ],
    "medium": [
        "miracle", "secret", "banned", "forbidden", "elite",
        "urgent", "alert", "exclusive",
        "outrageous", "disgusting", "terrifying",
        "deadly", "destroy", "destroyed"
    ],
    "low": [
        "you won't believe", "must see", "share before deleted",
        "going viral", "spread the word", "media silent",
        "big pharma", "deep state", "new world order"
    ]
}

CREDIBILITY_PHRASES = [
    "according to", "sources say", "reported by", "study shows",
    "research indicates", "data suggests", "officials said",
    "peer-reviewed", "published in", "press release",
    "spokesperson confirmed", "investigation found",
    "ministry of", "government of", "addressed the nation",
    "announced", "inaugurated", "launched"
]


def compute_sensationalism_score(text):
    text_lower = text.lower()
    score = 0
    matched_keywords = []
    for keyword in SENSATIONAL_KEYWORDS["high"]:
        if keyword in text_lower:
            score += 15
            matched_keywords.append({"word": keyword, "severity": "high"})
    for keyword in SENSATIONAL_KEYWORDS["medium"]:
        if keyword in text_lower:
            score += 8
            matched_keywords.append({"word": keyword, "severity": "medium"})
    for keyword in SENSATIONAL_KEYWORDS["low"]:
        if keyword in text_lower:
            score += 5
            matched_keywords.append({"word": keyword, "severity": "low"})
    return min(score, 100), matched_keywords


def count_syllables(word):
    word = word.lower()
    if len(word) <= 3:
        return 1
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in "aeiou"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e'):
        count -= 1
    return max(count, 1)


def compute_text_statistics(text):
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = round(sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1) * 100, 1)
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    unique_words = len(set(w.lower() for w in words))
    unique_ratio = round(unique_words / max(word_count, 1) * 100, 1)
    avg_word_length = round(np.mean([len(w) for w in words]), 1) if words else 0
    number_count = len(re.findall(r'\d+', text))
    syllable_count = sum(count_syllables(w) for w in words)
    if word_count > 0 and sentence_count > 0:
        reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        reading_ease = max(0, min(100, reading_ease))
    else:
        reading_ease = 50
    return {
        "word_count": word_count, "char_count": char_count,
        "sentence_count": sentence_count, "avg_word_length": avg_word_length,
        "exclamation_marks": exclamation_count, "question_marks": question_count,
        "caps_ratio": caps_ratio, "all_caps_words": all_caps_words,
        "vocabulary_richness": unique_ratio, "number_count": number_count,
        "readability_score": round(reading_ease, 1)
    }


def compute_credibility_indicators(text):
    text_lower = text.lower()
    return [p for p in CREDIBILITY_PHRASES if p in text_lower]


def compute_risk_factors(text, stats, sensationalism_score):
    factors = []
    if stats["caps_ratio"] > 30:
        factors.append({"factor": "Excessive capitalization",
                        "detail": "%s%% of letters are uppercase" % stats['caps_ratio'], "severity": "high"})
    elif stats["caps_ratio"] > 15:
        factors.append({"factor": "Above-average capitalization",
                        "detail": "%s%% of letters are uppercase" % stats['caps_ratio'], "severity": "medium"})
    if stats["exclamation_marks"] > 3:
        factors.append({"factor": "Multiple exclamation marks",
                        "detail": "%d exclamation marks found" % stats['exclamation_marks'], "severity": "high"})
    elif stats["exclamation_marks"] > 1:
        factors.append({"factor": "Exclamation marks present",
                        "detail": "%d exclamation marks found" % stats['exclamation_marks'], "severity": "medium"})
    if stats["word_count"] < 15:
        factors.append({"factor": "Short text",
                        "detail": "Only %d words - short texts have lower analysis reliability" % stats['word_count'],
                        "severity": "medium"})
    if sensationalism_score > 30:
        factors.append({"factor": "High sensationalism",
                        "detail": "Sensationalism score: %d/100" % sensationalism_score, "severity": "high"})
    elif sensationalism_score > 10:
        factors.append({"factor": "Moderate sensationalism",
                        "detail": "Sensationalism score: %d/100" % sensationalism_score, "severity": "medium"})
    if stats["vocabulary_richness"] < 40 and stats["word_count"] > 20:
        factors.append({"factor": "Low vocabulary diversity",
                        "detail": "Only %s%% unique words" % stats['vocabulary_richness'], "severity": "low"})
    if stats["all_caps_words"] > 3:
        factors.append({"factor": "Multiple ALL-CAPS words",
                        "detail": "%d fully capitalized words detected" % stats['all_caps_words'], "severity": "medium"})
    return factors


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 10:
        return jsonify({"error": "Text too short. Please provide at least 10 characters."}), 400

    cleaned = clean_text(text)

    proba = model.predict_proba([cleaned])[0]
    confidence_fake = proba[0]
    confidence_real = proba[1]

    stats = compute_text_statistics(text)
    sensationalism_score, sensational_keywords = compute_sensationalism_score(text)
    credibility_phrases = compute_credibility_indicators(text)

    word_count = stats["word_count"]
    if word_count < 30:
        damping = max(0.1, (word_count / 30.0) ** 0.7)
        confidence_fake = 0.5 + (confidence_fake - 0.5) * damping
        confidence_real = 0.5 + (confidence_real - 0.5) * damping

    cred_count = len(credibility_phrases)
    if cred_count > 0:
        cred_boost = min(cred_count * 0.08, 0.30)
        confidence_real += cred_boost
        confidence_fake -= cred_boost

    formal_score = 0
    if sensationalism_score == 0:
        formal_score += 1
    if stats["number_count"] > 0:
        formal_score += 1
    if stats["caps_ratio"] < 10:
        formal_score += 1
    if stats["exclamation_marks"] == 0:
        formal_score += 1
    words = text.split()
    proper_nouns = sum(1 for i, w in enumerate(words) if i > 0 and w[0].isupper() and not w.isupper()) if len(words) > 1 else 0
    if proper_nouns >= 2:
        formal_score += 1

    if formal_score >= 3:
        formal_boost = formal_score * 0.03
        confidence_real += formal_boost
        confidence_fake -= formal_boost

    if sensationalism_score > 0:
        sens_adj = sensationalism_score * 0.004
        confidence_fake += sens_adj
        confidence_real -= sens_adj

    if stats["caps_ratio"] > 25:
        confidence_fake += 0.06
        confidence_real -= 0.06

    confidence_fake = max(0.01, confidence_fake)
    confidence_real = max(0.01, confidence_real)
    total = confidence_fake + confidence_real
    confidence_fake /= total
    confidence_real /= total

    prediction = "real" if confidence_real >= 0.5 else "fake"
    confidence = confidence_real if prediction == "real" else confidence_fake

    risk_factors = compute_risk_factors(text, stats, sensationalism_score)

    credibility_score = round(
        (confidence_real * 60) +
        (min(cred_count, 3) * 5) +
        (max(0, 100 - sensationalism_score) * 0.2) +
        (min(stats["vocabulary_richness"], 80) * 0.1) +
        (5 if stats["number_count"] > 0 else 0),
        1
    )
    credibility_score = max(0, min(100, credibility_score))

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "credibility_score": credibility_score,
        "sensationalism_score": sensationalism_score,
        "text_statistics": stats,
        "risk_factors": risk_factors,
        "sensational_keywords": sensational_keywords,
        "credibility_phrases": credibility_phrases,
        "analysis_timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(debug=True)