import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib
import os

def train_baseline():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # TF-IDF: convert text into numeric features based on word importance
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_val = vectorizer.transform(val_df['sentence'])

    y_train = train_df['sentiment_label']
    y_val = val_df['sentiment_label']

    # class_weight='balanced' compensates for your 12% negative class
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    print(classification_report(y_val, preds))
    print(f"Macro F1: {f1_score(y_val, preds, average='macro'):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/baseline_logreg.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_baseline()