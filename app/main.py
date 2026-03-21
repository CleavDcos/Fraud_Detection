
import os


from fastapi import FastAPI
from pydantic import BaseModel

from app.utils.model_loader import download_file

from app.email_model.predict import predict_email
from app.url_model.model import predict_url
#from services.vector_store import search_similar
from services.llm_service import generate_explanation

app = FastAPI(title="Fraud Detection API")
PORT = int(os.environ.get("PORT", 10000))

# request schemas
class EmailRequest(BaseModel):
    text: str


class URLRequest(BaseModel):
    url: str

@app.on_event("startup")
def load_resources():
    global email_model, tfidf_model, url_model

    # DOWNLOAD MODELS
    download_file(
        "https://drive.google.com/uc?export=download&id=1lscce0n4fCDStrPcnJyPZCBB2bhO8EMt",
        "app/email_model/models/email_model.pkl"
    )

    download_file(
        "https://drive.google.com/uc?export=download&id=1MKkHm1it6pk8Kg8cdhKPiXiWXTWD6IxJ",
        "app/email_model/models/tfidf.pkl"
    )

    download_file(
        "https://drive.google.com/uc?export=download&id=15xaauIq-wYnAKDFy2SUzdSz-8LbdjFwI",
        "app/url_model/url_model.pkl"
    )

    # LOAD MODELS
    from app.email_model.predict import load_model
    email_model, tfidf_model = load_model()

    import joblib
    url_model = joblib.load("app/url_model/url_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict/email")
def analyze_email(request: EmailRequest):

    email = request.text  
 
    #ml prediction
    prediction = predict_email(email,email_model, tfidf_model,url_model)  
   
    #rag retrieval
    #similar_cases = search_similar(email)
    similar_cases=[]

    

    phishing_count = sum(1 for case in similar_cases if case["label"] == "phishing")
    #confidence = phishing_count / len(similar_cases) if similar_cases else 0
    confidence = 0

    #reasons = [case["reason"] for case in similar_cases]
    reasons = []
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


@app.post("/predict/url")
def predict_url_route(request: URLRequest):
    return predict_url(request.url, url_model)