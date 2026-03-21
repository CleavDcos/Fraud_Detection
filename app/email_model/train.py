import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.text_features import (
    extract_keyword_features,
    extract_structural_features,
    get_tfidf_vectorizer
)


#load tghe data
def load_data():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data/processed/emails.csv")

    df = pd.read_csv(path)
    return df


#feature enginneering
def build_features(df):
    texts = df["text"].fillna("").tolist()

    # 🔹 TF-IDF
    tfidf = get_tfidf_vectorizer()
    X_tfidf = tfidf.fit_transform(texts)

    # 🔹 Manual features
    manual_features = []

    for text in texts:
        keyword_feats = extract_keyword_features(text)
        structural_feats = extract_structural_features(text)

        combined = {**keyword_feats, **structural_feats}
        manual_features.append(list(combined.values()))

    X_manual = pd.DataFrame(manual_features)

    # 🔹 Combine
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_manual.values])

    return X, tfidf

#balance dataset as fraud emails samples are way more then legit.
def balance_dataset(df):
    fraud = df[df["label"] == 1]
    legit = df[df["label"] == 0]

    legit_count = len(legit)

    fraud_sampled = fraud.sample(n=legit_count, random_state=42)

    df_balanced = pd.concat([fraud_sampled, legit])
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

    return df_balanced


def train():
    df = load_data()

    print("Original distribution:")
    print(df["label"].value_counts())

    #BALANCE DATASET
    df = balance_dataset(df)

    print("\nBalanced distribution:")
    print(df["label"].value_counts())

    X, tfidf = build_features(df)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining model...")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save
    base_dir = os.path.dirname(__file__)
    joblib.dump(model, os.path.join(base_dir, "models/email_model.pkl"))
    joblib.dump(tfidf, os.path.join(base_dir, "models/tfidf.pkl"))

    print("\nModel saved!")


if __name__ == "__main__":
    train()