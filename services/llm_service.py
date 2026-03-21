from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_explanation(email, prediction, reasons, similar_cases):
    
    context = "\n".join([case["text"] for case in similar_cases])

    prompt = f"""
You are a cybersecurity expert.

Analyze this email:
"{email}"

Prediction: {prediction}

Known similar phishing examples:
{context}

Reasons detected:
{reasons}

Explain clearly why this email is safe or phishing in simple human language.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content