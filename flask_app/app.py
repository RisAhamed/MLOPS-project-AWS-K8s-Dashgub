from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import numpy as np
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Text Preprocessing Functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    """Apply text normalization pipeline."""
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# MLflow setup
dagshub_token = os.getenv("MLOPS_PROJECT")
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    dagshub_url = "https://dagshub.com"
    repo_owner = "RisAhamed"
    repo_name = "MLOPS-project-AWS-K8s-Dashgub"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
else:
    print("MLOPS_PROJECT environment variable not set, MLflow tracking disabled")

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry for Prometheus
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# Model and vectorizer setup
def get_latest_model_version(model_name):
    """Fetch the latest model version from MLflow."""
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        print(f"Error fetching model version: {e}")
        return None

# Load model with fallback options
try:
    model_name = "MLOPS-1"
    model_version = get_latest_model_version(model_name)
    
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Fetching model from MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        raise ValueError("No model version found")
except Exception as e:
    print(f"MLflow model loading failed: {e}")
    try:
        # Try Docker container path first
        if os.path.exists("/app/models/model.pkl"):
            model_path = "/app/models/model.pkl"
        else:
            model_path = "models/model.pkl"
        
        print(f"Loading model from local file: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e2:
        print(f"Local model loading failed: {e2}")
        raise RuntimeError(f"Failed to load model: {e2}")

# Load vectorizer
try:
    # Try Docker container path first
    if os.path.exists("/app/models/vectorizer.pkl"):
        vectorizer_path = "/app/models/vectorizer.pkl"
    else:
        vectorizer_path = "models/vectorizer.pkl"
    
    print(f"Loading vectorizer from: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    raise RuntimeError(f"Failed to load vectorizer: {e}")

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

    # Convert to features
    features = vectorizer.transform([cleaned_text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    try:
        prediction = model.predict(features_df)[0]
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    except Exception as e:
        return render_template("index.html", result=f"Prediction Error: {str(e)}")

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
