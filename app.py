import sys
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from src.mlproj.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    logging.info('Execution has started')

    try:
        data_ingest = DataIngestion()
        data_ingest.initiate_data_ingestion()
        
    except Exception as e:
        logging.info('Custom exception')
        raise CustomException(e, sys)
