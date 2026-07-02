import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

def train_baseline():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("stock-sentiment-classifier")

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    with mlflow.start_run(run_name="tfidf_logreg_baseline"):

        # --- Params ---
        max_features = 5000
        ngram_range = (1, 2)

        mlflow.log_param("model_type", "TFIDF_LogisticRegression")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("class_weight", "balanced")

        # --- Train ---
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = vectorizer.fit_transform(train_df['sentence'])
        X_val = vectorizer.transform(val_df['sentence'])

        y_train = train_df['sentiment_label']
        y_val = val_df['sentiment_label']

        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        # --- Evaluate ---
        preds = model.predict(X_val)
        report = classification_report(y_val, preds, output_dict=True)
        macro_f1 = f1_score(y_val, preds, average='macro')
        accuracy = accuracy_score(y_val, preds)

        # --- Log metrics ---
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("negative_f1", report['negative']['f1-score'])
        mlflow.log_metric("neutral_f1", report['neutral']['f1-score'])
        mlflow.log_metric("positive_f1", report['positive']['f1-score'])
        mlflow.log_metric("negative_recall", report['negative']['recall'])

        # --- Log model artifact to MLflow ---
        mlflow.sklearn.log_model(model, "model")

        print(classification_report(y_val, preds))
        print(f"Macro F1: {macro_f1:.4f}")

        # Still save locally too, for the FastAPI app to load directly
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/baseline_logreg.pkl")
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_baseline()