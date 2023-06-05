import os
import sys

import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.metrics import mean_squared_error, r2_score

def save_object(file_path, obj):
    logging.info("*** Function called - save_object() ***")
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info(f"---->> Saved the objects file_obj<<----")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models :dict):
    try:
        report = {}

        for key, model in models.items():
            model.fit(X_train, y_train)

            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            # print(test_model_score)   
            report[key] = test_model_score
            
        return report

    except Exception as e:
        raise CustomException(e, sys)