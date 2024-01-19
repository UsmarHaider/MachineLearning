import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",

                ],
                axis=1, )
            data['product_weight_g'] = data['product_weight_g'].fillna(data['product_weight_g'].mean())
            data['product_length_cm'] = data['product_length_cm'].fillna(data['product_length_cm'].mean())
            data['product_height_cm'] = data['product_height_cm'].fillna(data['product_height_cm'].mean())
            data['product_width_cm'] = data['product_width_cm'].fillna(data['product_width_cm'].mean())
            data['review_comment_message'] = data['review_comment_message'].fillna('No Comment', inplace=True)

            data = data.select_dtypes(include=[np.number])
            data = data.drop(['customer_zip_code_prefix', 'order_item_id'], axis=1)
            return data
        except Exception as e:
            logging.error("Error in DataPreProcessStrategy.handle_data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("Error in DataDivideStrategy.handle_data: {}".format(e))
            raise e


class DataCleaning:

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.__data = data
        self.__strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.__strategy.handle_data(self.__data)
        except Exception as e:
            logging.error("Error in DataCleaning.handle_data: {}".format(e))
            raise e
