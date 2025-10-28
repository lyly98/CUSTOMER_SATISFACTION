import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model performance.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            float: Evaluation score
        """
        pass


class MSE(Evaluation):
    """
    Evaluation metric for Mean Squared Error.
    
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) :
        """
        Calculates the Mean Squared Error between true and predicted values.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            float: Mean Squared Error score
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error calculated: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation metric for R-squared.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the R-squared score between true and predicted values.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            float: R-squared score
        """
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R-squared score calculated: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R-squared score: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation metric for Root Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the Root Mean Squared Error between true and predicted values.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            float: Root Mean Squared Error score
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error calculated: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating Root Mean Squared Error: {e}")
            raise e