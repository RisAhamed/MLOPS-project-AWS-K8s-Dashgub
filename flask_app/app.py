from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import re
import string
import numpy as np
import dagshub
from dotenv import load_dotenv
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Text Preprocessing Functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    """Remove numbers from the text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    """Convert text to lowercase."""
    return text.lower()

def removing_punctuations(text):
    """Remove punctuations from the text."""
    return re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text).strip()

def removing_urls(text):
    """Remove URLs from the text."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    """Apply text normalization pipeline."""
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# Initialize MLflow & DagsHub
dagshub.init(repo_owner="RisAhamed", repo_name="MLOPS-project-AWS-K8s-Dashgub", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RisAhamed/MLOPS-project-AWS-K8s-Dashgub.mlflow")
mlflow.set_experiment("model_evaluation_dvc")

# Get latest model version dynamically
def get_latest_model_version(model_name):
    """Fetch the latest model version from MLflow."""
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["None"])
    return versions[0].version if versions else None

# Load MLflow model
model_name = "MLOPS-1"
model_version = get_latest_model_version(model_name)
if model_version is None:
    raise RuntimeError(f"No available version for model: {model_name}")

model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
try:
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    raise RuntimeError(f"Failed to load MLflow model: {e}")

# Load vectorizer
vectorizer_path = os.path.join("models", "vectorizer.pkl")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Initialize Flask App
app = Flask(__name__)

# Custom Prometheus Metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    cleaned_text = normalize_text(text)

    # Convert text to features
    features = vectorizer.transform([cleaned_text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    try:
        prediction = model.predict(features_df)[0]
    except Exception as e:
        return render_template("index.html", result=f"Prediction Error: {str(e)}")

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    
    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
