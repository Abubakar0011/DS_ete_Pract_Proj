import sys
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from src.mlproj.components.data_ingestion import DataIngestion
from src.mlproj.components.data_transformation import DataTransformation

if __name__ == '__main__':
    logging.info('Execution has started')

    try:
        data_ingest = DataIngestion()
        train_data_path, test_data_path = data_ingest.initiate_data_ingestion()

        data_trn = DataTransformation()
        data_trn.initiate_data_transformation(train_data_path, test_data_path)
        
    except Exception as e:
        logging.info('Custom exception')
        raise CustomException(e, sys)
