import os
from fastapi import FastAPI
from pydantic import BaseModel

from app.utils.model_loader import download_file
from app.email_model.predict import predict_email
from app.url_model.model import predict_url
from services.llm_service import generate_explanation

app = FastAPI(title="Fraud Detection API")

# GLOBAL MODELS
email_model = None
tfidf_model = None
url_model = None



class EmailRequest(BaseModel):
    text: str


class URLRequest(BaseModel):
    url: str



@app.on_event("startup")
def load_resources():
    global email_model, tfidf_model, url_model

    try:
        print("Starting resource loading...")

        # Ensure directories exist
        os.makedirs("app/email_model/models", exist_ok=True)
        os.makedirs("app/url_model", exist_ok=True)

        # Get env variables
        email_link = os.getenv("EMAIL_MODEL_LINK")
        tfidf_link = os.getenv("TFIDF_LINK")
        url_link = os.getenv("URL_MODEL_LINK")

        if not email_link or not tfidf_link or not url_link:
            print("⚠️ Model links not found in environment variables")
            return

        # Download models
        download_file(email_link, "app/email_model/models/email_model.pkl")
        download_file(tfidf_link, "app/email_model/models/tfidf.pkl")
        download_file(url_link, "app/url_model/url_model.pkl")

        # Load email model
        from app.email_model.predict import load_model
        email_model, tfidf_model = load_model()

        # Load URL model
        import joblib
        url_model = joblib.load("app/url_model/url_model.pkl")

        print("All models loaded successfully")

    except Exception as e:
        print("STARTUP ERROR:", str(e))



@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


# ----------------------------
# EMAIL PREDICTION
# ----------------------------
@app.post("/predict/email")
def analyze_email(request: EmailRequest):

    if not email_model or not tfidf_model:
        return {"error": "Email model not loaded"}

    email = request.text

    try:
        prediction = predict_email(email, email_model, tfidf_model, url_model)

        # RAG (disabled for now)
        similar_cases = []
        reasons = []
        confidence = 0

        llm_explanation = generate_explanation(
            email,
            prediction,
            reasons,
            similar_cases
        )

        return {
            "is_phishing": prediction["is_fraud"],
            "model_confidence": prediction["confidence"],
            "rag_confidence": confidence,
            "final_confidence": round((prediction["confidence"] + confidence) / 2, 2),
            "llm_explanation": llm_explanation,
            "reasons": prediction["reasons"] + reasons,
            "similar_examples": similar_cases
        }

    except Exception as e:
        return {"error": str(e)}


# ----------------------------
# URL PREDICTION
# ----------------------------
@app.post("/predict/url")
def predict_url_route(request: URLRequest):

    if not url_model:
        return {"error": "URL model not loaded"}

    try:
        return predict_url(request.url, url_model)

    except Exception as e:
        return {"error": str(e)}