
import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords as nltk_stopwords 
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_dataframe(df,col="tet"):
    """
    Preprocesses a DataFrame by removing special characters, stop words, and converting to lowercase
    """
    lemmatizer = WordNetLemmatizer()
    stopwords = set(nltk_stopwords.words('english'))
    def preprocess_text(text):

        text = re.sub(r'https?://\S+|www\.\S+[^a-zA-Z0-9\s]', '', text)
        text = "".join([char for char in text if not char.isdigit()])
        text = text.lower()

        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        text = " ".join([word for word in text.split() if word not in stopwords])
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])
    logging.info(f"Data preprocessed successfully")
    return df

def main():
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logging.info("Data loaded successfully; processing started")
        train_data = preprocess_dataframe(train_data,"review")
        test_data = preprocess_dataframe(test_data,"review")
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        train_data.to_csv(os.path.join(data_path,"train_processed.csv"),index = False)
        test_data.to_csv(os.path.join(data_path,"test_processed.csv"),index = False)
        logging.info("Data processed and saved successfully")
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e

def data_preprocessing():
    try:
        main()
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise e

if __name__ == "__main__":
    main()