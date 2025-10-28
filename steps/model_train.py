import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig
import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig = ModelNameConfig(),
) -> RegressorMixin:


    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
            
        else:
            logging.error(f"Model {config.model_name} is not supported.")
            raise ValueError(f"Model {config.model_name} is not supported.")
    
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e