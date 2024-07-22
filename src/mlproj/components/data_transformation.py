import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from src.mlproj.utils import save_object_file_path


@dataclass
class DataTransfomationConfigPath:
    preprocessor_obj_fle_path = os.path.join(
        'artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transf_config = DataTransfomationConfigPath

    def get_data_transformer_object(self):
        '''This method will transform the input data before the passing 
        to model and avoid any inconsistency in the data'''
        try:

            numerical_col = ["writing_score", "reading_score"]
            categorical_col = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            '''these two pipelines are using for appropriate transformation'''
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{numerical_col}")
            logging.info(f"Numerical Columns:{categorical_col}")

            '''It combine both pipeline and implement 
            specific transformation'''
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_col),
                ('cat_pipeline', cat_pipeline, categorical_col)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading the training and the testing data files")

            preprocessor_obj = self.get_data_transformer_object()

            target_col_name = "math_score"

            '''split the train data into dependent and independent features'''
            inp_train_features_df = train_df.drop(
                columns=[target_col_name], axis=1)
            target_train_feature_df = train_df[target_col_name]

            '''split the test data into dependent and independent features'''
            inp_test_features_df = test_df.drop(
                columns=[target_col_name], axis=1)
            target_test_feature_df = test_df[target_col_name]

            logging.info(
                "Applying the preprocessing on the test and train dataframes")

            inp_fetur_train_arr = preprocessor_obj.fit_transform(
                inp_train_features_df)
            inp_fetur_test_arr = preprocessor_obj.transform(
                inp_test_features_df)

            '''Concatenating the train and test array with the target array'''

            trarin_arr = np.c_[
                inp_fetur_train_arr, np.array(target_train_feature_df)
            ]
            test_arr = np.c_[
                inp_fetur_test_arr, np.array(target_test_feature_df)
            ]

            save_object_file_path(
                file_path=self.data_transf_config.preprocessor_obj_fle_path,
                obj=preprocessor_obj
            )

            return (
                trarin_arr,
                test_arr,
                self.data_transf_config.preprocessor_obj_fle_path
            )
        except Exception as e:
            raise CustomException(e, sys)
