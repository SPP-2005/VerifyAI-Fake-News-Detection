# 🛡️ VerifyAI — Fake News Detection Web App

🔗 **Live Demo:** [https://verifyai-newscheck.vercel.app](https://verifyai-newscheck.vercel.app)

An AI-powered fake news detection system that uses a calibrated ensemble ML pipeline with multi-signal analysis to classify news articles as real or fake — with explainable confidence scores.

---

## ✅ Features

- 🤖 Calibrated ensemble classifier (Logistic Regression + SVM + SGD) with soft voting
- 📊 Word-level and character-level TF-IDF with 12 custom statistical features
- 🎯 95.9% accuracy on 37K+ balanced dataset with 5-fold cross-validation
- 📈 Multi-signal analysis: sensationalism scoring, credibility detection, formal tone analysis
- 🔍 Short-text uncertainty damping to prevent overconfident predictions
- 🎨 Modern glassmorphism UI with animated gauges and real-time analytics
- ☁️ Deployed on Vercel with auto-deploy from GitHub

---

## 🧠 How It Works

1. User enters a news headline or article
2. Text is cleaned and vectorized using dual TF-IDF (word + character n-grams)
3. Three calibrated classifiers vote on the prediction via soft voting
4. Multi-signal post-processing adjusts confidence based on:
   - Sensationalism keywords
   - Credibility phrases (e.g., "according to", "officials said")
   - Formal tone indicators (proper nouns, numbers, low caps)
   - Short-text uncertainty damping
5. Final verdict with confidence %, credibility score, and risk factors

---

## 📁 Project Structure

```
├── app.py                  # Flask API with multi-signal analysis
├── model_trainer.py        # Ensemble model training pipeline
├── feature_utils.py        # Shared text cleaning & feature extraction
├── vercel.json             # Vercel deployment config
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   ├── style.css           # Glassmorphism dark theme
│   ├── favicon.png         # Shield icon
│   └── wallpaper.jpg       # Background
```

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **ML:** Scikit-learn (TF-IDF, VotingClassifier, CalibratedClassifierCV)
- **Data:** Pandas, NumPy
- **Deployment:** Vercel

---

## 📦 Local Setup

```bash
pip install -r requirements.txt
python model_trainer.py
python app.py
```
Then open `http://localhost:5000`

> **Note:** You need `IFND.csv` (Indian Fake News Dataset) in the project root to train the model.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95.9% |
| F1 Score | 0.959 |
| CV Accuracy | 95.8% ± 0.5% |
| Precision (Fake) | 98% |
| Recall (Real) | 98% |

---

## ⚠️ Limitations

- Static model — not updated in real-time
- Pattern-based detection, not full fact verification
- May struggle with sarcasm, satire, or formal-sounding falsehoods
- Training dataset is primarily Indian political news

---

## 👋 Author

Made with ❤️ by [SPP-2005](https://github.com/SPP-2005)
