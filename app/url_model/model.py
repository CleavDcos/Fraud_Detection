import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from app.url_model.feature_extractor import extract_features



def load_data():
    phishing = pd.read_csv("data/phishing_urls.csv")
    legit = pd.read_csv("data/legit_urls.csv")

    phishing['label'] = 1
    legit['label'] = 0

    data = pd.concat([phishing, legit])
    return data


def prepare_data(data):
    X = []
    y = data['label']

    for url in data['url']:
        X.append(extract_features(url))

    return X, y


def train_model():
    data = load_data()

    # balance dataset
    phishing = data[data['label'] == 1]
    legit = data[data['label'] == 0].sample(len(phishing), random_state=42)

    data = pd.concat([phishing, legit])

    print(f"\nDataset after balancing: {len(phishing)} phishing + {len(legit)} legit")

    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Feature length during training:", len(X[0]))
    print("Model expects after training:", model.n_features_in_)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, "app/url_model/url_model.pkl")
    print("\nModel saved!")





def predict_url(url, model):
    try:
        if not url or not isinstance(url, str):
            raise ValueError("Invalid URL input")

        features = extract_features(url)

        # Convert to Python float
        proba = float(model.predict_proba([features])[0][1])

        if proba >= 0.7:
            label = "phishing"
        elif proba >= 0.4:
            label = "suspicious"
        else:
            label = "legit"

        return {
            "url": str(url),                      # ensure string
            "label": str(label),                  # ensure string
            "confidence": float(proba),           # ensure float
            "is_phishing": bool(proba >= 0.7)     # ensure bool
        }

    except Exception as e:
        return {
            "error": str(e),
            "url": str(url)
        }


if __name__ == "__main__":
    train_model()