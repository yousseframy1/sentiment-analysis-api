# Sentiment Analysis API

End-to-end machine learning project demonstrating training, evaluation, serialization, and serving of a sentiment classification model as a REST API.

## Features
- FastAPI backend
- Trained on Hugging Face IMDB dataset (50,000 movie reviews)
- **88.26% accuracy** using Logistic Regression with TF-IDF features
- Simple web interface for testing
- Confusion matrix evaluation

## Tech Stack
- Python 3.12.3
- FastAPI
- scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository:

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Train the model:
```bash
   jupyter notebook notebooks/train_model.ipynb
   # Run all cells to generate model files
```

4. Run the API:
```bash
   uvicorn app.main:app --reload
```

5. Open the web interface:
   - Open `frontend/index.html` in your browser
   - Or visit API docs at `http://localhost:8000/docs`

## API Documentation

### Endpoints

#### `GET /`
Returns API status message.

#### `POST /predict`
Predicts sentiment of input text.

**Request Body:**
```json
{
  "text": "This movie was amazing!"
}
```

**Response:**
```json
{
  "text": "This movie was amazing!",
  "sentiment": "positive",
  "confidence": 0.94
}
```

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF
- **Training Data**: Hugging Face IMDB Movie Reviews
- **Test Accuracy**: 88.26%
- **Evaluation**: Confusion matrix

## What I Learned

- Building end-to-end ML projects from training to deployment
- FastAPI framework
- Model serialization with pickle
- CORS configuration for web applications
- Linking the frontend with the API
- Git workflow and version control best practices

## Contact

youssef ramy - yousseframy200@gmail.com

Project Link: https://github.com/yousseframy1/sentiment-analysis-api