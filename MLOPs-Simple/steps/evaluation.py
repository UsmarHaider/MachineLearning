import logging

import pandas as pd
from zenml import step
from src.evaluation import R2, RMSE,MSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated


@step

def evaluate_model(model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Annotated[float, "r2"], Annotated[float, "rmse"], Annotated[float, "mse"]]:
    try:
        predictions = model.predict(x_test)
        rmse_class= RMSE()
        rmse= rmse_class.calculate_score(y_test, predictions)

        r2_class= R2()
        r2= r2_class.calculate_score(y_test, predictions)

        mse_class= MSE()
        mse= mse_class.calculate_score(y_test, predictions)
        print("R2: ", r2)
        print("RMSE: ", rmse)
        print("MSE: ", mse)


        return r2, rmse , mse
    except Exception as e:
        logging.error("Error in evaluate_model: {}".format(e))
        raise e



