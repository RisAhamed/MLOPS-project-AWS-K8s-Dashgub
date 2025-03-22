import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle

def load_params(params_path:str)->dict:
    try:
        with open(params_path) as yaml_file:
            params = yaml.safe_load(yaml_file)
            logging.info(f"loaded params successfully from {params_path}")
            return params
    except Exception as e:
        logging.exception(f"Error loading params from {params_path}: {e}")
        raise e

def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logging.info(f"loaded data successfully from {data_path}")
        df.fillna("",inplace=True)
        return df
    except Exception as e:
        logging.exception(f"Error loading data from {data_path}: {e}")
        raise e


def save_data(df: pd.DataFrame, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure dir exists
        df.to_csv(output_path, index=False)
        logging.info(f"Saved data successfully to {output_path}")
    except Exception as e:
        logging.exception(f"Error saving data to {output_path}: {e}")
        raise e

    

def apply_bow(train_df:pd.DataFrame,test_df:pd.DataFrame,max_features:int)->tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        x_train = train_df['review'].values
        x_test = test_df['review'].values
        y_train = train_df['sentiment'].values
        y_test =test_df['sentiment'].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        logging.info("Applied BOW successfully")
        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        logging.info('Bag of Words applied and data transformed')

        return train_df, test_df
    except Exception as e:
        logging.exception(f"Error applying BOW: {e}")
        raise e

def main():
    try:
        params =load_params("params.yaml")
        
        max_features = params["feature_engineering"]["max_features"]
        train_df =load_data("data/interim/train_processed.csv")
        test_df = load_data("data/interim/test_processed.csv")

        train_df,test_df = apply_bow(train_df,test_df,max_features)
        save_data(train_df,"data/processed/train_bow.csv")
        save_data(test_df,"data/processed/test_bow.csv")
    except Exception as e:
        logging.exception(f"Error in main function: {e}")
        raise e

def feature_engineering():
    main()

if __name__=="__main__":
    main()