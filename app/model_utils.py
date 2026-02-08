import pickle
from pathlib import Path

ModelPath = Path(__file__).parent.parent/"model"

def load_model():
    with open (ModelPath/"vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open (ModelPath/"classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    return vectorizer, classifier

def predict_sentiment(vectorizer, classifier, text:str):
    vector = vectorizer.transform([text])
    prediction = classifier.predict(vector)[0]
    pred_probability = classifier.predict_proba(vector)[0]
    return{
        "text" : text,
        "sentiment" : "positive" if prediction == 1 else "negative",
        "confidence" : max(pred_probability)
    }