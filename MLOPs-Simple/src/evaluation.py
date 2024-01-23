import logging

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):

    @abstractmethod

    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):

        pass

class MSE(Evaluation):
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray)-> float:
        try:
            logging.info("Calculating MSE score")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE score")
            raise e

class R2(Evaluation):

    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray)-> float:
        try:
            logging.info("Calculating R2 score")
            mean_y = np.mean(y_true)
            ss_tot = np.sum((y_true - mean_y)**2)
            ss_res = np.sum((y_true - y_pred)**2)
            r2 = 1 - (ss_res / ss_tot)
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score")
            raise e

class RMSE(Evaluation):

    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray)-> float:
        try:
            logging.info("Calculating RMSE score")
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE score")
            raise e

