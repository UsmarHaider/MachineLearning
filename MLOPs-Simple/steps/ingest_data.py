import logging
import pandas as pd
from zenml import step
class IngestData:
    """Ingest data from a file path."""
    def __init__(self, data_path : str):
        self.data_path = data_path
    def get_data(self):
        """
        `get_data` method reads the data from the file path provided in the constructor.

        Args:
            None

        :return:
            Pandas DataFrame of the data read from the file path provided in the constructor.

        """
        logging.info(f'Getting data from {self.data_path}')
        df = pd.read_csv(self.data_path)
        return df

@step
def ingest_df(data_path : str) -> pd.DataFrame:
    """Ingest data from a file path."""
    try :
        ingest_data = IngestData(data_path)
        df= ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while ingesting data: {e}')
        raise e

