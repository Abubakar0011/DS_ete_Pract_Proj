import os
import sys
import pandas as pd
import pymysql
from src.mlproj.logger import logging
from src.mlproj.exceptions import CustomException
from dotenv import load_dotenv
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