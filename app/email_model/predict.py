import os
from pydoc import text
import joblib
import numpy as np
import re

from app.url_model.model import predict_url
from scipy.sparse import hstack

from app.email_model.utils.text_features import (
    extract_keyword_features,
    extract_structural_features
)



def load_model():
    base_dir = os.path.dirname(__file__)

    model = joblib.load(os.path.join(base_dir, "models/email_model.pkl"))
    tfidf = joblib.load(os.path.join(base_dir, "models/tfidf.pkl"))

    return model, tfidf



def build_features(text, tfidf):
    text = text.lower()

    X_tfidf = tfidf.transform([text])

    keyword_feats = extract_keyword_features(text)
    structural_feats = extract_structural_features(text)

    manual_dict = {**keyword_feats, **structural_feats}
    manual_values = list(manual_dict.values())

    X_manual = np.array(manual_values).reshape(1, -1)

    X = hstack([X_tfidf, X_manual])

    return X, manual_dict


# give reasons
def generate_reasons(manual_features):
    reasons = []

    if manual_features["urgent_count"] > 0:
        reasons.append("Urgent language detected")

    if manual_features["threat_count"] > 0:
        reasons.append("Threatening language detected")

    if manual_features["suspicious_phrase_count"] > 0:
        reasons.append("Suspicious phrases like 'verify now' or 'click here'")

    if manual_features["uppercase_words"] > 5:
        reasons.append("Too many uppercase words (possible pressure tactic)")

    if manual_features["special_char_count"] > 10:
        reasons.append("Excessive special characters")

    if manual_features["email_length"] < 20:
        reasons.append("Very short message (common in scams)")

    return reasons


#make prediction
def predict_email(text):
    global email_model, tfidf_model

    X, manual_features = build_features(text, tfidf_model)

    prob = email_model.predict_proba(X)[0][1]
    pred = email_model.predict(X)[0]

    reasons = generate_reasons(manual_features)

    result = {
        "is_fraud": bool(pred),
        "confidence": float(prob),
        "reasons": reasons
    }

    urls = extract_urls(text)
    phishing_urls = []

    for url in urls:
        url_result = predict_url(url)

        if url_result["is_phishing"] or url_result["confidence"] >= 0.5:
            phishing_urls.append(url_result)

    if phishing_urls:
        reasons.append("Suspicious or phishing URL detected in email")

    result["phishing_urls"] = phishing_urls

    return result


#leveraging the url model here to see if url is present in the email
def extract_urls(text):
    return re.findall(r'https?://\S+', text)

#perform similarity search
def search_similar(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(data[idx])

    return results

if __name__ == "__main__":
    sample = "URGENT! Your account is suspended. Click here: http://fake-login.com to verify now!!!"

    result = predict_email(sample)

    print(result)