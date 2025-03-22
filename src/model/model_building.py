import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging


def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logging.exception(f"Error loading data from {data_path}: {e}")
        raise e

def train_model(x_train:np.ndarray,y_train:np.ndarray)->LogisticRegression:
    try:
        clf = LogisticRegression(C =2,solver = "liblinear",penalty="l1")
        clf.fit(x_train,y_train)
        logging.info("Model training completed successfully")
        return clf
    except Exception as e:
        logging.exception(f"Error training model: {e}")
        raise e

def save_model(model,file_path:str)->None:
    try:
        with open(file_path,"wb") as f:
            pickle.dump(model,f)
        logging.info(f"Model saved successfully at {file_path}")
    except Exception as e:
        logging.exception(f"Error saving model at {file_path}: {e}")
        raise e

def main():
    try:
        train_data = load_data("data/processed/train_bow.csv")
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train,y_train)
        save_model(clf,"models/model.pkl")
        logging.info("Model training and saving completed successfully")
    except Exception as e:
        logging.exception(f"Error in main function: {e}")
        raise e

def model_building():
    main()
    
if __name__ == "__main__":
    main()