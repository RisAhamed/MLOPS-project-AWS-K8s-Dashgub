FROM python:3.10-slim

WORKDIR /app

# Copy application files
COPY flask_app/ /app/

# Create models directory and copy model files
RUN mkdir -p /app/models
COPY models/vectorizer.pkl /app/models/vectorizer.pkl
COPY models/model.pkl /app/models/model.pkl

# Install requirements
COPY flask_app/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# For production with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]