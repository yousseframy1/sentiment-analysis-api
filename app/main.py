from fastapi import FastAPI
from app.model_utils import load_model, predict_sentiment
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer, classifier = load_model()

class Text (BaseModel):
    text: str

@app.get("/")
def root():
    return {"message":"Sentiment Analysis API"}

@app.post("/predict")
def predict(input_data : Text):
    result = predict_sentiment(vectorizer, classifier, input_data.text)
    return result