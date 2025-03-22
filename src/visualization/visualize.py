from src.logger import logging
try:
    logging.info("Hello World")
except Exception as e:
    logging.error(f"Error in data ingestion: {e}")
    raise e