from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from app.email_model.predict import predict_email, load_model
from app.url_model.model import predict_url
from services.llm_service import generate_explanation

app = FastAPI(title="Fraud Detection API")

# GLOBAL MODELS
email_model = None
tfidf_model = None
url_model = None


# ----------------------------
# REQUEST MODELS
# ----------------------------
class EmailRequest(BaseModel):
    text: str


class URLRequest(BaseModel):
    url: str


# ----------------------------
# STARTUP: LOAD MODELS
# ----------------------------
@app.on_event("startup")
def load_resources():
    global email_model, tfidf_model, url_model

    try:
        print("🔄 Loading models locally...")

        # Load email model
        email_model, tfidf_model = load_model()

        # Load URL model
        url_model = joblib.load("app/url_model/url_model.pkl")

        # Debug check (VERY IMPORTANT)
        print("✅ URL Model expects:", url_model.n_features_in_)

        print("✅ All models loaded successfully")

    except Exception as e:
        print("❌ STARTUP ERROR:", str(e))
        raise e


# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


# ----------------------------
# EMAIL PREDICTION
# ----------------------------
@app.post("/predict/email")
def analyze_email(request: EmailRequest):

    if email_model is None or tfidf_model is None:
        return {"error": "Email model not loaded"}

    try:
        prediction = predict_email(
            request.text,
            email_model,
            tfidf_model,
            url_model
        )

        # Placeholder for RAG (future)
        similar_cases = []
        reasons = []
        confidence = 0

        llm_explanation = generate_explanation(
            request.text,
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

    if url_model is None:
        return {"error": "URL model not loaded"}

    try:
        return predict_url(request.url, url_model)

    except Exception as e:
        return {"error": str(e)}