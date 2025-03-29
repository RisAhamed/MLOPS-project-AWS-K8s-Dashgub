# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from dotenv import load_dotenv
load_dotenv()
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from src.connections import s3_connection


def load_params(params_path:str)->dict:
    try: 
        with open(params_path) as yaml_file:
            params = yaml.safe_load(yaml_file)
            logging.info(f"params loaded from: {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise e
    except yaml.YAMLError as e:
        logging.error(f"Error loading params: {e}")
        raise e

def read_data(data_path_url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path_url)
        logging.info(f"data loaded from: {data_path_url}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error loading data: {e}")
        raise e
    except pd.errors.ParserError as e:
        logging.error(f"Error loading data: {e}")
        raise e
    except pd.errors.DtypeWarning as e:
        logging.error(f"Error loading data: {e}")
        raise e

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        logging.info(f"data preprocessing started")
        df = df.drop_duplicates()
        df = df.dropna()
        df = df.reset_index(drop=True)
        final_df =df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].map({'positive': 1, 'negative': 0})
        logging.info(f"data preprocessed")
        return final_df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        logging.info(f"data saving started")
        data_path = os.path.join(data_path, "raw")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logging.info(f"data saved to: {data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error saving data: {e}")
        raise e

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed). """
    try:
        params = load_params("params.yaml")
        test_size = params['data_ingestion']['test_size']

        df = read_data(data_path_url=params['data_ingestion']['data_path_url'])
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        s3_bucket_name = os.getenv("S3_BUCKET_NAME")
        data_name = os.getenv("S3_DATA_NAME")
        # s3 = s3_connection.s3_operations(s3_bucket_name, aws_access_key, aws_secret_key)
        # df = s3.fetch_file_from_s3(data_name)
        df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path=params['data_ingestion']['data_path'])
        logging.info(f"data ingestion completed")
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e

def data_ingestion():
    try:
        main()
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e

if __name__ == "__main__":
    main()