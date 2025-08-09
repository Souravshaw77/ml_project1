import os
import sys

import numpy as np
import pandas as pd
import dill
import logging


from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Get parameters for the current model
            para = param.get(model_name, {})

            # Hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Set the best parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Save the test score
            report[model_name] = test_score

            logging.info(
                f"{model_name} — Best Params: {gs.best_params_}, Train R2: {train_score}, Test R2: {test_score}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
