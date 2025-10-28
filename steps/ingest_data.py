import logging
import pandas as pd
from zenml import step

class IngestData:
    '''Class to handle data ingestion from CSV files.'''
    def __init__(self,data_path: str):
        """
        Args:
            data_path (str): Path to the CSV file containing the data.
        """
        self.data_path = data_path


    def get_data(self):
        """
        Ingests data from the specified CSV file.

        """
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)       
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the ingested data.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error ingesting data: {e}')
        raise e