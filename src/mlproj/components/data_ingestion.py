import os
import sys
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from dataclasses import dataclass
from src.mlproj.utils import reading_SQL_data
from sklearn.model_selection import train_test_split

''' This class specifically store path for various data files involved
in data ingestion process '''


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'raw.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        ''' Reading data from sql db making paths and directories
            and returning the training path and 
            the test path from through this method '''

        try:
            df = reading_SQL_data()

            logging.info('Reading data Completes from mySql DB.')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(
                self.ingestion_config.raw_data_path, 
                index=False, header=True)

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            df.to_csv(
                self.ingestion_config.train_data_path, 
                index=False, header=True)
            
            df.to_csv(
                self.ingestion_config.test_data_path, 
                index=False, header=True)

            logging.info('Data ingestion is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
