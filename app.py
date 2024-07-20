from src.mlproj.logger import logging  # type: ignore
from src.mlproj.exceptions import CustomException  # type: ignore
import sys

if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        div = 1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
