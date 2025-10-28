import logging
import pandas as pd
from zenml import step
from src.evaluation import RMSE, Evaluation, MSE, R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                x_test:pd.DataFrame,
                y_test:pd.DataFrame,
                ) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse_score"],
              
                ]:
    
    """
    Evaluates the trained model using MSE and R2 metrics.
    
    Args:
        model: Trained regression model
        x_test: Test feature data
        y_test: Test target data
    """
    try:
        predictions = model.predict(x_test)

        mse = MSE()
        r2 = R2()
        rmse = RMSE()

        mse_score = mse.calculate_scores(y_test, predictions)
        r2_score = r2.calculate_scores(y_test, predictions)
        rmse_score = rmse.calculate_scores(y_test, predictions)

        
        mlflow.log_metric("MSE", mse_score)
        mlflow.log_metric("R2_Score", r2_score)
        mlflow.log_metric("RMSE", rmse_score)
        return r2_score, rmse_score
    
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
