from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://router.huggingface.co/models/google/flan-t5-base"


headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    payload = {"inputs": req.message}

    response = requests.post(MODEL_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return {"reply": "Model loading or HF API error. Try again."}

    try:
        data = response.json()
    except:
        return {"reply": "Invalid response from model."}

    if isinstance(data, list) and len(data) > 0:
        return {"reply": data[0].get("generated_text", "")}

    return {"reply": str(data)}
