# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import re
import os
import sys
from dataclasses import dataclass

# Modelling

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","trained_model.pkl")
    logging.info("***  Object Initiallized of class- ModelTrainerConfig() ***")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("***  Object Initiallized of class- ModelTrainer() ***")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("*** Function called - initiate_model_trainer() ***")
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                # "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradiant Boosting":GradientBoostingRegressor(),
            }

            params={
                "Linear Regression":{},
                "Lasso": {
                    'alpha':[0.25, 0.5, 0.75, 1.0 , 1.25 , 1.5, 1.75, 2.0],
                    },
                "Ridge": {
                    'alpha':[0.25, 0.5, 0.75, 1.0 , 1.25 , 1.5, 1.75, 2.0],
                    },
                "K-Neighbors Regressor": {'n_neighbors': [2,3,4,5,6],
                                           'weights': ['uniform','distance']
                                           },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                        },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "XGBRegressor":{
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                # },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradiant Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
            }
            
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                models=models, param=params)
            
            logging.info("Evaluated Model -- Proceeding to select best model")
            
            ## To get the best score model from dict
            best_model_score = max(sorted(model_report.values()))

            ## To select the best score model from dict
            best_model = models[list(model_report.keys())[list(model_report.values()).index(best_model_score)]]
        
            if best_model_score <0.6:
                raise CustomException("All models scored <0.6 --> Aborting /n Try cleaning data and re-train the model")
            logging.info(f"Best Model selected --> {best_model}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logging.info(f"R2_Score from best_model : {best_model_score}")
            logging.info(f"R2_Score from best_model : {r2_square}")

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
