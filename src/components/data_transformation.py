import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomeException
from src.logger import logging
import os
from src.utils.features import get_features

from src.utils.filesManager import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.numerical_columns=[]
        self.categorical_columns=[]

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = self.numerical_columns
            categorical_columns = self.categorical_columns

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomeException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_column_name="IsCounselingNeeded"
            self.numerical_columns, self.categorical_columns = get_features(train_df, target_column_name)
            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # train_arr = np.c_[
            #     input_feature_train_arr, np.array(target_feature_train_df)
            # ]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
######### fixing the shape issue
            # Ensure dense arrays for safe concatenation
            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, 'toarray'):
                input_feature_test_arr = input_feature_test_arr.toarray()

            y_train = np.asarray(target_feature_train_df).reshape(-1, 1)
            y_test = np.asarray(target_feature_test_df).reshape(-1, 1)

            logging.info(f"X_train shape: {input_feature_train_arr.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {input_feature_test_arr.shape}, y_test shape: {y_test.shape}")

            train_arr = np.hstack([input_feature_train_arr, y_train])
            test_arr = np.hstack([input_feature_test_arr, y_test])
###########

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomeException(e,sys)