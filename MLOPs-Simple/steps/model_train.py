import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from .config import ModelNameConfig
@step
def train_model(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: ModelNameConfig,
)->RegressorMixin:

    model=None
    try:
        if config.model_name == "LinearRegressionModel":
            model = LinearRegressionModel().train(x_train, y_train)
            return model
        else:
            raise Exception("Model not found")

    except Exception as e:
        logging.error("Error in train_model: {}".format(e))
        raise e