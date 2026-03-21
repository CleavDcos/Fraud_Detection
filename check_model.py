import joblib

print("Email model:", type(joblib.load("app/email_model/models/email_model.pkl")))
print("TFIDF model:", type(joblib.load("app/email_model/models/tfidf.pkl")))