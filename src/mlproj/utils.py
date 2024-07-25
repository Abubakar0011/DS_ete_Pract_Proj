import os
import sys
import pandas as pd
import pymysql
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')

'''This method will connect to sql database and wil return the dataframe '''


def reading_SQL_data():
    logging.info('Reading SQL database statrted')
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info('Connection is Established', mydb)
        df = pd.read_sql_query('select * from students', mydb)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e, sys)


'''It is ageneric routine will take the preprocessor object file path and 
will dump that prpprocessor into pickle file'''


def save_object_file_path(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            logging.info(
                f"Model:{model}, Train R2 Score: {train_model_score}")

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
