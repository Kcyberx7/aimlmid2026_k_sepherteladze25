import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# =========================
# CONFIGURATION
# =========================
FEATURE_COLS = ["words", "links", "capital_words", "spam_word_count"]
TARGET_COL = "is_spam"

SPAM_KEYWORDS = [
    "free", "winner", "urgent", "money", "offer", "click",
    "prize", "bonus", "limited", "now"
]

# =========================
# DATA LOADING
# =========================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    for c in FEATURE_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df

# =========================
# MODEL TRAINING & EVALUATION
# =========================
def train_and_evaluate(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 70% train / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["legitimate(0)", "spam(1)"])

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)

    print("=== Model Coefficients ===")
    for name, coef in zip(FEATURE_COLS, model.coef_[0]):
        print(f"{name:15s}: {coef:.6f}")
    print(f"intercept       : {model.intercept_[0]:.6f}")

    return model, cm, acc, df

# =========================
# VISUALIZATIONS
# =========================
def visualize(df, cm):
    # A) Class Distribution
    plt.figure()
    df[TARGET_COL].value_counts().plot(kind="bar")
    plt.title("Class Distribution (Spam vs Legitimate)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # B) Confusion Matrix Heatmap
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Spam"],
                yticklabels=["Legitimate", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

# =========================
# EMAIL TEXT PARSING
# =========================
def extract_features_from_text(text: str):
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    links = len(re.findall(r"http[s]?://|www\.", text))

    capital_words = sum(1 for w in words if w.isupper())

    spam_word_count = sum(
        1 for w in words if w.lower() in SPAM_KEYWORDS
    )

    return [[word_count, links, capital_words, spam_word_count]]

# =========================
# INTERACTIVE EMAIL CHECK
# =========================
def interactive_email_check(model):
    print("\n=== Email Text Classification ===")
    print("Paste email text below. Type 'q' to quit.\n")

    while True:
        text = input("Email text > ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break

        features = extract_features_from_text(text)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        label = "SPAM" if prediction == 1 else "LEGITIMATE"
        print(f"Prediction: {label} | P(spam) = {probability:.4f}\n")

# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) != 2:
        print("Usage: python spam_classifier.py <path_to_csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = load_data(csv_path)
    model, cm, acc, df = train_and_evaluate(df)
    visualize(df, cm)
    interactive_email_check(model)

if __name__ == "__main__":
    main()
