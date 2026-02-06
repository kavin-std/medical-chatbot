from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://router.huggingface.co/models/microsoft/BioGPT"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    payload = {"inputs": req.message}
    response = requests.post(MODEL_URL, headers=headers, json=payload)

    try:
        data = response.json()
    except:
        return {"reply": "Model temporarily unavailable. Try again."}

    if isinstance(data, list):
        return {"reply": data[0].get("generated_text", "")}

    return {"reply": str(data)}
