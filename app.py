import os
import sys

# Add the project root directory to Python path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_ingestion import data_ingestion
from src.data.data_preprocessing import data_preprocessing
from src.features.feature_engineering import feature_engineering
from src.model.model_building import model_building
from src.model.model_evaluation import model_evaluation_dvc
from src.model.model_registry import model_registry
from src.logger import logging

if __name__ == "__main__":
    # Configure logger before running any steps
    
    logging.info("Starting pipeline execution")
    # Run pipeline steps
    data_ingestion()
    data_preprocessing()
    feature_engineering()
    model_building()
    model_evaluation_dvc()
    model_registry()

