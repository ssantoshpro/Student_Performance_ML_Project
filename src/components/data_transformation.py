import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig():
    logging.info("***  Object Initiallized of class- DataTransformationConfig() ***")
    preprocessor_obj_file_pattern = os.path.join("artifact","preprocessor.pkl")
    

class DataTransformation():
    def __init__(self) -> None:
        logging.info("***  Object Initiallized of class- DataTransformation() ***")
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data tansformations.
        '''
        logging.info("*** Function called - get_data_transformer_obj() ***")
        try:
            logging.info("++++++++++++  Data Transformation Started  ++++++++++++")
            numeric_features = [ 'reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ], 
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    # ("scaler", StandardScaler())
                ],
            )

            logging.info("Numberical Column : {}".format(numeric_features))
            logging.info("Numberical Column : {}".format(categorical_features))

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipline", num_pipeline, numeric_features),
                    ("categorical_pipline", cat_pipeline, categorical_features)
                ]
            )

            logging.info("Numberical Column Standardization Completed Sucessfully")
            logging.info("Categorical Column Encoding Completed Sucessfully")
            logging.info("++++++++++++  Data Transformation Ended  ++++++++++++")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("*** Function called - initiate_data_transformation() ***")
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Train and test data is loaded to the DataFrame")
            logging.info("Obtaining preprocessor objects")

            preprocessing_obj = self.get_data_transformer_obj() 

            target_column = ["math_score"]
            numeric_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            input_feature_df_train = df_train.drop(columns=target_column, axis=1)
            output_feature_df_train = df_train[target_column]

            input_feature_df_test = df_test.drop(columns=target_column, axis=1)
            output_feature_df_test = df_test[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_arr_train = preprocessing_obj.fit_transform(input_feature_df_train)
            input_feature_arr_test = preprocessing_obj.transform(input_feature_df_test)

            train_arr = np.c_[input_feature_arr_train, np.array(output_feature_df_train)]
            test_arr = np.c_[input_feature_arr_test, np.array(output_feature_df_test)]
            logging.info("Applied Preprocessing")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_pattern,
                obj = preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_pattern,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    pass
