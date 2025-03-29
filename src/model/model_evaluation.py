import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging
from dotenv import load_dotenv
load_dotenv()


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("MLOPS_PROJECT")
if not dagshub_token:
    raise EnvironmentError("MLOPS_PROJECT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "RisAhamed"
repo_name = "MLOPS-project-AWS-K8s-Dashgub"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


tracking_uri = os.getenv("MLFLOW_URL")
# mlflow.set_tracking_uri(tracking_uri)
dagshub.init(repo_owner='RisAhamed', repo_name='MLOPS-project-AWS-K8s-Dashgub', mlflow=True)

def load_model(model_path:str)->pickle:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.exception(f"Error loading model from {model_path}: {e}")
        raise e

def load_data(data_path:str)->pd.DataFrame:
    try:
        df =pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logging.exception(f"Error loading data from {data_path}: {e}")
        raise e

def evaluate_model(model,x_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred = model.predict(x_test)
        y_pred_probe= model.predict_proba(x_test)[:,1]
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred_probe)
        metrics = {"accuracy":accuracy,"precision":precision,"recall":recall,"roc_auc":roc_auc}
        logging.info(f"Model evaluation completed: {metrics}")
        return metrics
    except Exception as e:
        logging.exception(f"Error evaluating model: {e}")
        raise e

def save_metrics(metrics:dict,metrics_path:str)->None:
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        logging.info(f"Metrics saved successfully at {metrics_path}")
    except Exception as e:
        logging.exception(f"Error saving metrics at {metrics_path}: {e}")
        raise e

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise
def upload_model_to_dagshub(model_path: str) -> None:
    """Upload the model to DagsHub."""
    try:
        # Use the appropriate push method for uploading files
        dagshub.push(model_path, f'models/{os.path.basename(model_path)}')
        logging.info(f"Model uploaded successfully to DagsHub: {model_path}")
    except Exception as e:
        logging.error(f"Error uploading model to DagsHub: {e}")
        raise

def main():
    mlflow.set_experiment("model_evaluation_dvc")
    with mlflow.start_run():
        model_path = "models/model.pkl"  # Using forward slash for compatibility
        model = load_model(model_path)
        test_data = load_data("data/processed/test_bow.csv")
        x_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        metrics = evaluate_model(model,x_test,y_test)

        save_metrics(metrics,"reports/metrics.json")
        for metric_name,metric_value in metrics.items():
            mlflow.log_metric(metric_name,metric_value)
        mlflow.sklearn.log_model(model,"model")
        run_id = mlflow.active_run().info.run_id
        model_path = f"runs:/{run_id}/model"
        save_model_info(run_id, model_path, 'reports/model_info.json')
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("reports/model_info.json")
        # upload_model_to_dagshub(model_path)
        if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)


def model_evaluation_dvc():
    main()
if __name__ == "__main__":
    main()