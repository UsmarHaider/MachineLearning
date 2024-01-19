import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):

    @abstractmethod
    def train(self, x_train, y_train):
        pass


class LinearRegressionModel(Model):

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            logging.error("Error in LinearRegressionModel.train: {}".format(e))
            raise e
