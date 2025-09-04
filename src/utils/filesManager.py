import os
import sys
from src.logger import logging

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV

from src.exception import CustomeException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42))
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Confusion-matrix-derived metrics: use F1 as scalar selection metric
            f1 = f1_score(y_test, y_test_pred)
            # Optionally compute confusion matrix for future use/logging if needed
            cm = confusion_matrix(y_test, y_test_pred)

            logging.info(f"F1 score for {list(models.keys())[i]}: {f1}")
            logging.info(f"Confusion matrix for {list(models.keys())[i]}: {cm}")

            report[list(models.keys())[i]] = f1

        return report

    except Exception as e:
        raise CustomeException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomeException(e, sys)