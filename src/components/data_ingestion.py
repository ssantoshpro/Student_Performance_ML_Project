import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import re

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    logging.info("***  Object Initiallized of class- DataIngestionConfig() ***")
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        logging.info("***  Object Initiallized of class- DataIngestion() ***")
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("*** Function called - initiate_data_ingestion() ***")
        logging.info("++++ Entered the data ingestion method/component ++++")
        try:
            df = pd.read_csv('Notebook\data\StudentsPerformance.csv')
            logging.info('Procured the data from the Source: *.CSV file')
            ###
            old_col = list(df.columns)
            # new_col = [str(col).replace(string.punctuation,"_").replace(" ","_") for col in list(df.columns)]
            new_col = [str(re.sub('[^a-zA-Z0-9\n\.]', '_',col)) for col in list(df.columns)]

            rename_col={}
            rename_col.update(zip(old_col,new_col))
            df.rename(columns = rename_col, inplace=True)
            #####
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train-Test -> Split : initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=41)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("---- Exiting the data ingestion method/component ----")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)