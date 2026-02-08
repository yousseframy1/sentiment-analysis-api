from fastapi import FastAPI
from app.model_utils import load_model, predict_sentiment
from pydantic import BaseModel

app = FastAPI(title="Sentiment Analysis API")

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