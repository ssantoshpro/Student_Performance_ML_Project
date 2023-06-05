import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipleline:
    def __init__(self) -> None:
        logging.info("***  Object Initiallized of class- PredictPipleline: ***")
        pass
    def predict(self,features):
        try:
            logging.info("*** Function called - predict() ***")
            model_path = r"artifacts\trained_model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
            gender:str,
            race_ethnicity:str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            math_score:int,
            reading_score:int,
            writing_score:int):
        logging.info("***  Object Initiallized of class- CustomData: ***")
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score
    

    def get_data_dataframe(self):
        try:
            logging.info("*** Function called - get_data_dataframe() ***")
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "math_score":[self.math_score],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score],
            }
            
            return pd.DataFrame(data=custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

