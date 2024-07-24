import sys
from mlproj.components.model_trainer import ModelTrainer
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from src.mlproj.components.data_ingestion import DataIngestion
from src.mlproj.components.data_transformation import DataTransformation
from src.mlproj.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    logging.info('Execution has started')

    try:

        data_ingestion = DataIngestion()
        trn_dta_pat, tst_dta_pat = data_ingestion.initiate_data_ingestion()

        # data_transformation_config=DataTransformationConfig()
        data_transformation = DataTransformation()
        train_ar, test_ar, _ = data_transformation.initiate_data_transormation(
            trn_dta_pat, tst_dta_pat)

        # Model Training

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_ar, test_ar))

    except Exception as e:
        logging.info('Custom exception')
        raise CustomException(e, sys)
