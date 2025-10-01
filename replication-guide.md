# Replication Guide - MLOps Sentiment Analysis Project

This guide provides step-by-step instructions to replicate, run, test, and deploy the MLOps Sentiment Analysis project from scratch.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] Docker installed (optional)
- [ ] Make installed (optional)
- [ ] Internet connection
- [ ] DagsHub account (free tier available)

## Step 1: Environment Setup

### 1.1 Clone Repository

```bash
# Clone the repository
git clone <<REPLACE_ME>>
cd MLOPS-project-AWS-K8s-Dashgub

# Verify repository structure
ls -la
```

**Expected Output:**
```
total 48
drwxr-xr-x 12 user user 4096 Jan 15 10:30 .
drwxr-xr-x  3 user user 4096 Jan 15 10:30 ..
-rw-r--r--  1 user user 1234 Jan 15 10:30 README.md
-rw-r--r--  1 user user  567 Jan 15 10:30 requirements.txt
-rw-r--r--  1 user user  234 Jan 15 10:30 Dockerfile
drwxr-xr-x  2 user user 4096 Jan 15 10:30 src
drwxr-xr-x  2 user user 4096 Jan 15 10:30 flask_app
drwxr-xr-x  2 user user 4096 Jan 15 10:30 tests
...
```

### 1.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Verify Python version
python --version
```

**Expected Output:**
```
Python 3.10.x
```

### 1.3 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python test_environment.py
```

**Expected Output:**
```
>>> Development environment passes all tests!
```

## Step 2: Configuration Setup

### 2.1 Create Environment File

```bash
# Create .env file
cat > .env << EOF
# MLflow/DagsHub Configuration
MLOPS_PROJECT=<<REPLACE_ME>>

# AWS Configuration (Optional)
AWS_ACCESS_KEY=<<REPLACE_ME>>
AWS_SECRET_KEY=<<REPLACE_ME>>
S3_BUCKET_NAME=<<REPLACE_ME>>
S3_DATA_NAME=<<REPLACE_ME>>

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
EOF
```

### 2.2 Verify Configuration Files

```bash
# Check params.yaml exists and is valid
python -c "import yaml; print(yaml.safe_load(open('params.yaml')))"
```

**Expected Output:**
```
{'data_ingestion': {'test_size': 0.3, 'data_path_url': 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv', 'data_path': './data'}, 'feature_engineering': {'max_features': 20}}
```

## Step 3: Data Pipeline Execution

### 3.1 Run Complete ML Pipeline

```bash
# Execute the complete pipeline
python app.py
```

**Expected Output:**
```
[ 2025-01-15 10:30:15,123 ] root - INFO - Starting pipeline execution
[ 2025-01-15 10:30:15,124 ] root - INFO - params loaded from: params.yaml
[ 2025-01-15 10:30:15,125 ] root - INFO - data loaded from: https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv
[ 2025-01-15 10:30:16,200 ] root - INFO - data preprocessing started
[ 2025-01-15 10:30:16,201 ] root - INFO - data preprocessed
[ 2025-01-15 10:30:16,202 ] root - INFO - data saving started
[ 2025-01-15 10:30:16,203 ] root - INFO - data saved to: ./data/raw
[ 2025-01-15 10:30:16,204 ] root - INFO - data ingestion completed
...
[ 2025-01-15 10:35:45,678 ] root - INFO - Model training and saving completed successfully
[ 2025-01-15 10:35:45,679 ] root - INFO - Model evaluation completed: {'accuracy': 0.85, 'precision': 0.87, 'recall': 0.83, 'roc_auc': 0.89}
```

### 3.2 Verify Pipeline Outputs

```bash
# Check data directories
ls -la data/
ls -la models/
ls -la reports/

# Verify model files exist
ls -la models/*.pkl
ls -la reports/*.json
```

**Expected Output:**
```
data/
├── raw/
│   ├── train.csv
│   └── test.csv
├── interim/
│   ├── train_processed.csv
│   └── test_processed.csv
└── processed/
    ├── train_bow.csv
    └── test_bow.csv

models/
├── model.pkl
└── vectorizer.pkl

reports/
├── metrics.json
└── model_info.json
```

### 3.3 Individual Pipeline Steps (Alternative)

If you prefer to run steps individually:

```bash
# Step 1: Data Ingestion
python src/data/data_ingestion.py

# Step 2: Data Preprocessing
python src/data/data_preprocessing.py

# Step 3: Feature Engineering
python src/features/feature_engineering.py

# Step 4: Model Building
python src/model/model_building.py

# Step 5: Model Evaluation
python src/model/model_evaluation.py

# Step 6: Model Registry
python src/model/model_registry.py
```

## Step 4: Web Application Testing

### 4.1 Start Flask Development Server

```bash
# Navigate to flask app directory
cd flask_app

# Start development server
python app.py
```

**Expected Output:**
```
MLflow tracking URI set to: https://dagshub.com/RisAhamed/MLOPS-project-AWS-K8s-Dashgub.mlflow
Loading vectorizer from: models/vectorizer.pkl
Loading model from local file: models/model.pkl
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[::1]:5000
 * Debug mode: on
```

### 4.2 Test Web Interface

Open browser and navigate to `http://localhost:5000`

**Expected Behavior:**
- Web page loads with sentiment analysis form
- Form has text area and "Predict" button
- No prediction result initially

### 4.3 Test API Endpoints

```bash
# Test home page
curl http://localhost:5000/

# Test prediction endpoint
curl -X POST http://localhost:5000/predict -d "text=I love this product"

# Test metrics endpoint
curl http://localhost:5000/metrics
```

**Expected Outputs:**

**Home Page Response:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
...
```

**Prediction Response:**
```html
<!DOCTYPE html>
<html lang="en">
...
<div class="result positive">
    😊 Positive Sentiment
</div>
...
```

**Metrics Response:**
```
# HELP app_request_count Total number of requests
# TYPE app_request_count counter
app_request_count{endpoint="/",method="GET"} 1.0
app_request_count{endpoint="/predict",method="POST"} 1.0
# HELP app_request_latency_seconds Latency of requests
# TYPE app_request_latency_seconds histogram
...
```

## Step 5: Testing Suite

### 5.1 Run Unit Tests

```bash
# Navigate back to project root
cd ..

# Run all tests
python -m pytest tests/ -v
```

**Expected Output:**
```
========================= test session starts =========================
tests/test_flask_app.py::FlaskAppTests::test_home_page PASSED
tests/test_flask_app.py::FlaskAppTests::test_predict_page PASSED
tests/test_model.py::TestModel::test_model_loading PASSED
========================= 3 passed in 2.34s =========================
```

### 5.2 Run Linting

```bash
# Run code linting
make lint
```

**Expected Output:**
```
src/data/data_ingestion.py:1:1: F401 'click' imported but unused
src/model/model_evaluation.py:89:1: F401 'dagshub' imported but unused
```

### 5.3 Run Environment Tests

```bash
# Test Python environment
python test_environment.py
```

**Expected Output:**
```
>>> Development environment passes all tests!
```

## Step 6: Docker Deployment

### 6.1 Build Docker Image

```bash
# Build Docker image
docker build -t sentiment-analysis-app .

# Verify image creation
docker images | grep sentiment-analysis-app
```

**Expected Output:**
```
sentiment-analysis-app   latest    abc123def456   2 minutes ago   1.2GB
```

### 6.2 Run Docker Container

```bash
# Run container
docker run -d -p 5000:5000 --name sentiment-app sentiment-analysis-app

# Check container status
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                   COMMAND                  CREATED         STATUS         PORTS                    NAMES
abc123def456   sentiment-analysis-app  "gunicorn --bind 0.0.…"   2 minutes ago   Up 2 minutes   0.0.0.0:5000->5000/tcp   sentiment-app
```

### 6.3 Test Docker Deployment

```bash
# Test containerized application
curl http://localhost:5000/

# Check container logs
docker logs sentiment-app
```

**Expected Output:**
```
MLflow tracking URI set to: https://dagshub.com/RisAhamed/MLOPS-project-AWS-K8s-Dashgub.mlflow
Loading vectorizer from: /app/models/vectorizer.pkl
Loading model from local file: /app/models/model.pkl
[2025-01-15 10:45:12 +0000] [1] [INFO] Starting gunicorn 21.2.0
[2025-01-15 10:45:12 +0000] [1] [INFO] Listening at: http://0.0.0.0:5000 (1)
[2025-01-15 10:45:12 +0000] [1] [INFO] Using worker: sync
[2025-01-15 10:45:12 +0000] [8] [INFO] Booting worker with pid: 8
```

## Step 7: Production Deployment

### 7.1 Local Production Server

```bash
# Stop development server (Ctrl+C)
# Start production server
cd flask_app
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
```

**Expected Output:**
```
[2025-01-15 10:50:00 +0000] [12345] [INFO] Starting gunicorn 21.2.0
[2025-01-15 10:50:00 +0000] [12345] [INFO] Listening at: http://0.0.0.0:5000 (12345)
[2025-01-15 10:50:00 +0000] [12345] [INFO] Using worker: sync
[2025-01-15 10:50:00 +0000] [12346] [INFO] Booting worker with pid: 12346
[2025-01-15 10:50:00 +0000] [12347] [INFO] Booting worker with pid: 12347
[2025-01-15 10:50:00 +0000] [12348] [INFO] Booting worker with pid: 12348
[2025-01-15 10:50:00 +0000] [12349] [INFO] Booting worker with pid: 12349
```

### 7.2 Load Testing

```bash
# Install Apache Bench (if available)
# Ubuntu/Debian: sudo apt-get install apache2-utils
# macOS: brew install httpd

# Run load test
ab -n 100 -c 10 http://localhost:5000/

# Test prediction endpoint
ab -n 50 -c 5 -p test_data.txt -T 'application/x-www-form-urlencoded' http://localhost:5000/predict
```

Create `test_data.txt`:
```
text=I love this amazing product!
```

## Step 8: Monitoring and Logging

### 8.1 Check Logs

```bash
# View application logs
ls -la logs/
tail -f logs/$(ls -t logs/ | head -1)
```

**Expected Output:**
```
[ 2025-01-15 10:30:15,123 ] root - INFO - Starting pipeline execution
[ 2025-01-15 10:30:15,124 ] root - INFO - params loaded from: params.yaml
[ 2025-01-15 10:30:16,200 ] root - INFO - data preprocessing started
...
```

### 8.2 Monitor Metrics

```bash
# Check Prometheus metrics
curl http://localhost:5000/metrics | head -20
```

**Expected Output:**
```
# HELP app_request_count Total number of requests
# TYPE app_request_count counter
app_request_count{endpoint="/",method="GET"} 15.0
app_request_count{endpoint="/predict",method="POST"} 8.0
# HELP app_request_latency_seconds Latency of requests
# TYPE app_request_latency_seconds histogram
app_request_latency_seconds_bucket{endpoint="/",le="0.005"} 10.0
app_request_latency_seconds_bucket{endpoint="/",le="0.01"} 12.0
...
```

## Step 9: CI/CD Pipeline Setup

### 9.1 Create GitHub Actions Workflow

```bash
# Create GitHub Actions directory
mkdir -p .github/workflows

# Create CI workflow
cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Run linting
      run: |
        pip install flake8
        flake8 src/
    
    - name: Test environment
      run: |
        python test_environment.py
    
    - name: Build Docker image
      run: |
        docker build -t sentiment-analysis-test .
    
    - name: Test Docker container
      run: |
        docker run -d -p 5000:5000 --name test-container sentiment-analysis-test
        sleep 10
        curl -f http://localhost:5000/ || exit 1
        docker stop test-container
        docker rm test-container
EOF
```

### 9.2 Manual CI Checklist

Before pushing to repository, run:

```bash
# Complete CI checklist
echo "Running CI checklist..."

# 1. Install dependencies
pip install -r requirements.txt
echo "✓ Dependencies installed"

# 2. Test environment
python test_environment.py
echo "✓ Environment test passed"

# 3. Run linting
make lint
echo "✓ Linting passed"

# 4. Run tests
python -m pytest tests/ -v
echo "✓ Tests passed"

# 5. Test Docker build
docker build -t sentiment-analysis-test .
echo "✓ Docker build successful"

# 6. Test Docker container
docker run -d -p 5000:5000 --name test-container sentiment-analysis-test
sleep 10
curl -f http://localhost:5000/ && echo "✓ Docker container test passed"
docker stop test-container && docker rm test-container

echo "All CI checks passed! Ready for deployment."
```

## Step 10: Cloud Deployment Examples

### 10.1 AWS ECS Deployment

```bash
# Build and tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <<REPLACE_ME>>
docker tag sentiment-analysis-app:latest <<REPLACE_ME>>/sentiment-analysis:latest
docker push <<REPLACE_ME>>/sentiment-analysis:latest

# Create ECS task definition
cat > task-definition.json << 'EOF'
{
  "family": "sentiment-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::<<REPLACE_ME>>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "sentiment-analysis",
      "image": "<<REPLACE_ME>>/sentiment-analysis:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MLOPS_PROJECT",
          "value": "<<REPLACE_ME>>"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analysis",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### 10.2 Google Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<<REPLACE_ME>>/sentiment-analysis

# Deploy to Cloud Run
gcloud run deploy sentiment-analysis \
  --image gcr.io/<<REPLACE_ME>>/sentiment-analysis \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MLOPS_PROJECT=<<REPLACE_ME>>
```

### 10.3 Heroku Deployment

```bash
# Create Heroku app
heroku create sentiment-analysis-app

# Set environment variables
heroku config:set MLOPS_PROJECT=<<REPLACE_ME>>

# Deploy
git push heroku main

# Check deployment
heroku logs --tail
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: MLflow Connection Error
```
Error: MLOPS_PROJECT environment variable is not set
```

**Solution:**
```bash
# Set environment variable
export MLOPS_PROJECT=your_dagshub_token

# Or add to .env file
echo "MLOPS_PROJECT=your_dagshub_token" >> .env
```

#### Issue 2: Model Files Not Found
```
Error: Failed to load model
```

**Solution:**
```bash
# Ensure pipeline has been run
python app.py

# Check if model files exist
ls -la models/
```

#### Issue 3: Port Already in Use
```
Error: Address already in use
```

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use different port
python flask_app/app.py --port 5001
```

#### Issue 4: NLTK Data Missing
```
Error: Resource stopwords not found
```

**Solution:**
```bash
# Download NLTK data
python -m nltk.downloader stopwords wordnet

# Or add to Dockerfile
RUN python -m nltk.downloader stopwords wordnet
```

#### Issue 5: Docker Build Fails
```
Error: COPY failed: file not found
```

**Solution:**
```bash
# Ensure models are built before Docker build
python app.py
docker build -t sentiment-analysis-app .
```

### Debug Commands

```bash
# Check Python environment
python test_environment.py

# Verify dependencies
pip list | grep -E "(flask|mlflow|pandas|numpy)"

# Check MLflow connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Test model loading
python -c "import pickle; model = pickle.load(open('models/model.pkl', 'rb')); print('Model loaded successfully')"

# Check logs
tail -f logs/$(ls -t logs/ | head -1)

# Monitor system resources
htop
# or
docker stats
```

### Performance Optimization

```bash
# Monitor memory usage
ps aux | grep python

# Check disk usage
du -sh models/ data/ logs/

# Monitor network connections
netstat -tulpn | grep :5000

# Profile application
python -m cProfile flask_app/app.py
```

## Verification Checklist

After completing all steps, verify:

- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] ML pipeline executed successfully
- [ ] Model files generated
- [ ] Flask application starts without errors
- [ ] Web interface accessible
- [ ] API endpoints respond correctly
- [ ] Tests pass
- [ ] Docker image builds successfully
- [ ] Docker container runs without errors
- [ ] Prometheus metrics available
- [ ] Logs are generated
- [ ] CI/CD pipeline configured (if applicable)

## Next Steps

1. **Customize Configuration**: Modify `params.yaml` for your specific use case
2. **Add Monitoring**: Set up Prometheus and Grafana dashboards
3. **Implement CI/CD**: Configure automated testing and deployment
4. **Scale Deployment**: Use Kubernetes or cloud services
5. **Enhance Security**: Add authentication and rate limiting
6. **Improve Performance**: Implement caching and optimization
7. **Add Features**: Extend the model with additional capabilities

## Support

If you encounter issues not covered in this guide:

1. Check the troubleshooting section above
2. Review the main README.md for additional information
3. Check the project's issue tracker
4. Contact the maintainer: <<REPLACE_ME>>

---

**Last Updated**: January 2025  
**Version**: 1.0  
**Maintainer**: Riswan Ahamed