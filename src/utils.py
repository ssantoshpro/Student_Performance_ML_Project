import os
import sys

import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from src.logger import logging

def save_object(file_path, obj):
    logging.info("*** Function called - save_object() ***")
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info("---->> Saved the objects <<----")

    except Exception as e:
        raise CustomException(e, sys)