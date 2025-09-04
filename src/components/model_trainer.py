import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomeException
from src.logger import logging

from src.utils.filesManager import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            logging.info(f"train array: {train_array}")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            models = {
                "Random Forest": RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None),
                "XGBClassifier": XGBClassifier(eval_metric='logloss', n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
                "CatBoost Classifier": CatBoostClassifier(verbose=False, random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
            }
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 5, 10, 20]
                },
                "Random Forest":{
                    'n_estimators': [100,200,400],
                    'max_depth': [None, 5, 10]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.05,.01],
                    'n_estimators': [100,200,300]
                },
                "Logistic Regression":{
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.05,.01],
                    'n_estimators': [100,200,300]
                },
                "CatBoost Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.5,1.0],
                    'n_estimators': [50,100,200]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomeException("No best model found with sufficient F1 score")
            logging.info(f"Best classifier selected based on F1 score on test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            f1 = f1_score(y_test, predicted)
            return f1
            



            
        except Exception as e:
            raise CustomeException(e,sys)