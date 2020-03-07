import numpy as np

from .model import Model


class GradienDescentModel(Model):

    def __init__(self, input_size=1, output_size=1):
        super().__init__(input_size, output_size)

        self._w = np.random.randn(self._input_size, self._output_size)
        self._b = np.random.randn(1, self._output_size)

    def MSE(self, y_predicted, y_expected):
        return np.square(y_predicted - y_expected).mean()

    def _backward_pass(self, x, y_pred, y_expected, lr):
        x = np.array(x).reshape(self._input_size,)

        x = np.array(x).reshape(1, self._input_size)

        dloss_dy = 2.0 / self._output_size * (y_pred - y_expected)

        dy_dw = x.T
        dy_db = 1

        dloss_dw = dy_dw @ dloss_dy
        dloss_db = dy_db * dloss_dy

        self._w -= lr * dloss_dw
        self._b -= lr * dloss_db

    def fit(self, x_train, y_train, x_val, y_val, epochs_number, lr=0.001, early_stopping_rounds=2, verbose=10):
        train_loss_history = []
        val_loss_history = []

        best_loss = self.MSE(self.predict(x_val).reshape(-1), y_val)
        best_loss_epoch = 0

        for epoch in range(epochs_number):

            for i in range(len(x_train)):
                x = x_train[i]
                y_expected = y_train[i]
                y_pred = self.predict(x)

                self._backward_pass(x, y_pred, y_expected, lr)

            train_loss_history.append(
                self.MSE(self.predict(x_train).reshape(-1), y_train)
            )
            val_loss_history.append(
                self.MSE(self.predict(x_val).reshape(-1), y_val)
            )

            if val_loss_history[-1] < best_loss:
                best_loss = val_loss_history[-1]
                best_loss_epoch = epoch
            elif val_loss_history[-1] > best_loss and early_stopping_rounds > 0:
                if epoch - best_loss_epoch > early_stopping_rounds:
                    print(
                        f'Early stopping. Epoch {epoch + 1}. Current loss: {val_loss_history[-1]}'
                    )
                    break

            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(
                    f'{epoch+1}: train_loss: {train_loss_history[-1]}; val_loss: {val_loss_history[-1]}'
                )

        return train_loss_history, val_loss_history

    def predict(self, x):
        x = np.array(x).reshape(-1, self._input_size)
        result = x @ self._w + self._b
        return result.reshape(-1, self._output_size)

    def get_weights(self):
        return self._w, self._b
