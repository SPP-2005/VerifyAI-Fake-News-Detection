# 📰 Fake News Detection Web App

This is a basic full-stack fake news detection web application that allows users to input news content and get instant predictions on whether the article is **real** or **fake**. The model uses traditional NLP techniques and logistic regression for classification, presented through a clean, interactive frontend.

---

## ✅ Features

- ✅ Logistic Regression model trained on a curated fake news dataset
- ✅ TF-IDF vectorization of input text
- ✅ Flask-based backend with a REST API for predictions
- ✅ Frontend built using HTML, CSS, and JavaScript
- ✅ Animated confidence bar showing model certainty
- ✅ Optional keyword-based confidence booster for suspicious words
- ✅ Responsive design with styled user interface
- ✅ Clean, modular structure (separate model, API, and UI layers)

---

## 🧠 How It Works

1. The user enters a news headline or article into the frontend form.
2. A JavaScript function sends this data to the Flask API using a POST request.
3. The backend loads a trained ML model (`fake_news_model.pkl`) and processes the input.
4. The model returns a **"real"** or **"fake"** label with a **confidence percentage**.
5. The frontend displays the result along with a smooth animated confidence bar and color-coded label.

---

## 📁 Folder Structure

```
project/
├── app.py                  # Flask backend API
├── model_trainer.py        # Training script for the ML model
├── IFND.csv                # Dataset file
├── fake_news_model.pkl     # Trained ML model
│
├── templates/
│   └── index.html          # Frontend HTML
│
├── static/
│   ├── style.css           # Custom CSS styles
│   ├── favicon.png         # Tab icon (optional)
│   └── wallpaper.jpg       # Background image
```

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn (TF-IDF + Logistic Regression)
- **Data Handling**: Pandas, NumPy

---

## 📦 Setup Instructions

### Step 1: Install Dependencies
```bash
pip install flask pandas scikit-learn joblib
```

### Step 2: Train the Model (Optional if `fake_news_model.pkl` already exists)
```bash
python model_trainer.py
```

### Step 3: Run the Flask App
```bash
python app.py
```

### Step 4: Open in Browser
```
http://localhost:5000
```

---

## 🧪 Dataset Info

- Dataset: `IFND.csv` (Indian Fake News Dataset)
- Used columns: `"Statement"` as input, `"Label"` as target (`TRUE` = 1, `Fake` = 0)
- Model trained using `TF-IDF` features and balanced sampling

---

## 📊 Limitations

- The model is **static**, trained once and not updated in real-time
- It does **not perform full fact verification**, only pattern detection for the tone of fake news in the dataset provided in the files
- May fail on sarcasm, satire, or formal-sounding falsehoods
- May not distinguish between fake, misleading, or out-of-distribution content beyond what it has seen in training

---

## 🚀 Future Improvements

- 🔍 Integrate transformer models like BERT for better semantic understanding
- 🧠 Add claim verification using natural language inference (e.g., FEVER dataset)
- 🌐 Deploy to the web via Render/Heroku
- 🔄 Add feedback loop for user correction and retraining
- 🛡️ Include more complex misinformation types (e.g., satire, propaganda)

---

## 👋 Author

Made with ❤️ by a passionate developer exploring the intersection of **AI** and **media credibility**.

Feel free to connect, suggest improvements, or collaborate on future upgrades!
