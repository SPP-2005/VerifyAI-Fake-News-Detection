"""
Enhanced Fake News Detection Model Trainer
-------------------------------------------
Uses calibrated classifiers with advanced text preprocessing
and feature engineering for improved fake news detection accuracy.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from feature_utils import clean_text, TextStatisticsExtractor

warnings.filterwarnings('ignore')

print("=" * 60)
print("  Enhanced Fake News Detection Model Trainer")
print("=" * 60)

print("\nLoading dataset...")
df = pd.read_csv('IFND.csv', encoding='ISO-8859-1')
print("   Total samples: %d" % len(df))

df = df[['Statement', 'Label']].copy()
df['Label'] = df['Label'].map({'Fake': 0, 'TRUE': 1})
df.dropna(inplace=True)

print("\nClass distribution (before balancing):")
print("   Real (TRUE): %d" % (df['Label'] == 1).sum())
print("   Fake:         %d" % (df['Label'] == 0).sum())

fake_df = df[df['Label'] == 0]
real_df = df[df['Label'] == 1]
real_sampled = real_df.sample(n=len(fake_df), random_state=42)
df_balanced = pd.concat([real_sampled, fake_df])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced dataset: %d total" % len(df_balanced))

print("Cleaning text data...")
df_balanced['Statement_clean'] = df_balanced['Statement'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['Statement_clean'], df_balanced['Label'],
    test_size=0.2, random_state=42, stratify=df_balanced['Label']
)
print("Training: %d, Testing: %d" % (len(X_train), len(X_test)))

print("\nBuilding enhanced pipeline...")

tfidf_word = TfidfVectorizer(
    stop_words='english', max_df=0.5, min_df=5,
    ngram_range=(1, 2), max_features=40000,
    sublinear_tf=True, strip_accents='unicode'
)
tfidf_char = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5),
    max_features=20000, sublinear_tf=True, strip_accents='unicode'
)

feature_union = FeatureUnion([
    ('tfidf_word', tfidf_word),
    ('tfidf_char', tfidf_char),
    ('text_stats', Pipeline([
        ('stats', TextStatisticsExtractor()),
        ('scaler', StandardScaler())
    ]))
])

svc_calibrated = CalibratedClassifierCV(
    LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'),
    cv=3, method='sigmoid'
)
sgd_calibrated = CalibratedClassifierCV(
    SGDClassifier(loss='modified_huber', max_iter=1000,
                  class_weight='balanced', random_state=42),
    cv=3, method='sigmoid'
)

ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=0.5, max_iter=2000, solver='lbfgs', class_weight='balanced')),
        ('svc', svc_calibrated),
        ('sgd', sgd_calibrated),
    ],
    voting='soft',
    weights=[2, 1, 1]
)

pipeline = Pipeline([('features', feature_union), ('clf', ensemble)])

print("\nTraining calibrated ensemble model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n   Accuracy: %.4f (%.2f%%)" % (accuracy, accuracy*100))
print("   F1 Score: %.4f" % f1)
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

cm = confusion_matrix(y_test, y_pred)
print("   Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d" % (cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, df_balanced['Statement_clean'], df_balanced['Label'], cv=cv, scoring='accuracy')
print("   Mean CV Accuracy: %.4f (+/- %.4f)" % (cv_scores.mean(), cv_scores.std()*2))

joblib.dump(pipeline, 'fake_news_model.pkl')
print("\n[OK] Model saved! Ensemble=%.1f%%" % (accuracy*100))