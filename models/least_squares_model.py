import numpy as np

from .model import Model


class LeastSquaresModel(Model):

    def __init__(self, input_size=1, output_size=1):
        assert output_size == 1, 'Output size must be 1'
        super().__init__(input_size, output_size)
        self._w = np.random.randn(input_size, 1)
        self._b = np.random.randn(1, 1)

    def fit(self, x_train, y_train):
        corrcoef = np.corrcoef(x_train, y_train)[1][0]
        self._w = np.std(y_train, ddof=1) / np.std(x_train, ddof=1) * corrcoef
        self._w = self._w.reshape(self._input_size, 1)
        self._b = y_train.mean() - self._w * x_train.mean()

    def predict(self, x):
        x = np.array(x).reshape(-1, self._input_size)
        result = x @ self._w + self._b
        return result.reshape(-1, 1)

    def get_weights(self):
        return self._w, self._b
