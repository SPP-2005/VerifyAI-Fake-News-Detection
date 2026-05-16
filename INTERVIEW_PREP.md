# Fake News Detector (VerifyAI) — Interview Prep Guide

---

## Resume Description

**Fake News Detection Web App (VerifyAI)** | RepositoryLink
Personal Project · Python, Flask, Scikit-learn, HTML/CSS/JS, NLP · Machine Learning

- Built a calibrated ensemble classifier (LR + SVM + SGD) with TF-IDF and custom feature extraction, achieving 95.9% accuracy on a 37K+ sample balanced dataset with 5-fold cross-validation.
- Developed a multi-signal analysis API combining model predictions with sensationalism scoring, credibility detection, and short-text uncertainty damping for explainable fake news classification.

---

## Interview Q&A Cheat-Sheet

### "What does your project do?"
It takes a news article as input and tells you if it's likely real or fake, along with a confidence score and reasons why.

### "How does the ML model work?"
I convert text into numbers using TF-IDF — it counts how important each word is relative to the whole dataset. Then I feed those numbers into 3 classifiers (Logistic Regression, SVM, SGD) and they vote on the answer. This voting approach is called an ensemble.

### "What is TF-IDF?"
Term Frequency–Inverse Document Frequency. Words that appear a lot in one article but rarely across all articles get a high score. Common words like "the" get low scores. It's a way to find what makes each article unique.

### "Why 3 models instead of 1?"
Each model has different strengths. Combining them through soft voting reduces the chance of any single model's weakness affecting the result. It's like asking 3 doctors instead of 1.

### "What's the multi-signal analysis?"
The ML model alone can be wrong on short text. So I also check: Does it use ALL CAPS? Exclamation marks? Sensational words like "shocking"? Or credible phrases like "according to"? These signals adjust the final score.

### "What was your biggest challenge?"
Short political news was getting misclassified because words like "demonetization" appeared equally in fake and real news. I fixed it by damping confidence on short texts and boosting formal-tone signals like proper nouns and factual language.

### "What is an ensemble classifier?"
It's a technique where multiple ML models are combined to make a final prediction. In my case, I use soft voting — each model gives a probability, and the weighted average decides the outcome. This generally outperforms any single model.

### "What is calibration?"
Some classifiers (like SVM) don't naturally output well-calibrated probabilities. CalibratedClassifierCV wraps them so their confidence scores are meaningful — a 70% prediction should actually be correct 70% of the time.

### "How did you handle class imbalance?"
The original dataset had 37,800 real and 18,914 fake articles. I undersampled the real news to match the fake news count, creating a perfectly balanced 37,828-sample dataset. I also used stratified splits to maintain balance in train/test sets.

### "What features does the model use?"
Three types: (1) Word-level TF-IDF (unigrams + bigrams), (2) Character-level TF-IDF (3-5 char n-grams, captures writing style), and (3) 12 custom statistical features like word count, caps ratio, punctuation density, vocabulary richness, etc.

### "What's the tech stack?"
- Backend: Python, Flask
- ML: Scikit-learn (TF-IDF, LogisticRegression, LinearSVC, SGDClassifier, VotingClassifier)
- Frontend: HTML, CSS, JavaScript
- Data: Pandas, NumPy
- Dataset: IFND (Indian Fake News Dataset) — 56,714 articles

---

## Key Terms to Know

| Term | Simple Meaning |
|------|---------------|
| TF-IDF | Converts text to numbers based on word importance |
| Ensemble | Combining multiple models for better accuracy |
| Soft Voting | Each model gives a probability, averaged for final answer |
| Calibration | Making probability scores trustworthy |
| Precision | Of all items predicted as X, how many were actually X |
| Recall | Of all actual X items, how many did we correctly find |
| F1 Score | Balance between precision and recall |
| Cross-Validation | Testing model on different data splits to check consistency |
| Undersampling | Reducing the bigger class to match the smaller one |
| n-gram | A sequence of n words (bigram = 2 words together) |
