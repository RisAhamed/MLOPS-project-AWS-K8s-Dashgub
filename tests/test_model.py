import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from tenacity import retry, stop_after_attempt, wait_exponential

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        # Set up Dagshub credentials
        dagshub_token = os.getenv("MLOPS_PROJECT")
        if not dagshub_token:
            raise EnvironmentError("MLOPS_PROJECT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Configure MLflow tracking URI
        dagshub_url = "https://dagshub.com"
        repo_owner = "RisAhamed"
        repo_name = "MLOPS-project-AWS-K8s-Dashgub"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Define model details
        cls.new_model_name = "MLOPS-1"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name,)
        if not cls.new_model_version:
            raise ValueError(f"No versions found for model {cls.new_model_name}")
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"

        

        try:
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except Exception as e:
            print(f"Failed to load model after retries: {e}")
            raise
        
        # Load vectorizer
# ... existing code ...

# Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        vectorizer_path = os.path.join(project_root, "models", "vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        with open(vectorizer_path, "rb") as f:
            cls.vectorizer = pickle.load(f)

        # vectorizer_path = os.path.join("models", "vectorizer.pkl")
        # if not os.path.exists(vectorizer_path):
        #     raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        # with open(vectorizer_path, "rb") as f:
        #     cls.vectorizer = pickle.load(f)

        # Load holdout data
        holdout_data_path = os.path.join("data", "processed", "test_bow.csv")
        if not os.path.exists(holdout_data_path):
            raise FileNotFoundError(f"Holdout data file not found at {holdout_data_path}")
        cls.holdout_data = pd.read_csv(holdout_data_path)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None
        # if not versions:
        #     return None
        # latest_version = max(versions, key=lambda v: int(v.version))
        # return latest_version.version
    

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model, "Model failed to load from MLflow")

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_names_out())
        prediction = self.new_model.predict(input_df)
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]
        y_pred_new = self.new_model.predict(X_holdout)
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
        recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)
        expected = 0.40
        self.assertGreaterEqual(accuracy_new, expected, f"Accuracy {accuracy_new:.2f} < {expected}")
        self.assertGreaterEqual(precision_new, expected, f"Precision {precision_new:.2f} < {expected}")
        self.assertGreaterEqual(recall_new, expected, f"Recall {recall_new:.2f} < {expected}")
        self.assertGreaterEqual(f1_new, expected, f"F1 score {f1_new:.2f} < {expected}")

if __name__ == "__main__":
    unittest.main()