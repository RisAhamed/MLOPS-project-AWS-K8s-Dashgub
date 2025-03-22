from src.data.data_ingestion import data_ingestion
from src.data.data_preprocessing import data_preprocessing
from src.features.feature_engineering import feature_engineering
if __name__ == "__main__":
    data_ingestion()
    data_preprocessing()
    feature_engineering()
