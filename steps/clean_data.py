import logging
from typing import Tuple

import pandas as pd
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated

# from zenml.steps import Output, step
from zenml import step


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
        Returns:
            x_train: pd.DataFrame   
            x_test: pd.DataFrame
            y_train: pd.Series
            y_test: pd.Series
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed successfully.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
