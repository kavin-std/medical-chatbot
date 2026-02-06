from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
HF_TOKEN = os.getenv("HF_TOKEN")
app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://router.huggingface.co/models/google/flan-t5-small"




headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    payload = {"inputs": req.message}

    response = requests.post(MODEL_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return {"reply": "Model temporarily unavailable"}

    try:
        data = response.json()
    except:
        return {"reply": "Invalid model response"}

    if isinstance(data, list) and len(data) > 0:
        return {"reply": data[0].get("generated_text", "")}

    return {"reply": str(data)}
