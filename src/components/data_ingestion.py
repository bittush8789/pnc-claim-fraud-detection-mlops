import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_path='insurance_claims.csv'):
        logging.info("Starting data ingestion component...")
        try:
            df = pd.read_csv(source_path)
            logging.info(f"Loaded source dataset '{source_path}' shape: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            # Perform train test split
            logging.info("Performing train test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraud_reported'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion completed. Raw, Train, and Test sets generated.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
