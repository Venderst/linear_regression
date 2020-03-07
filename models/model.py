from abc import ABC, abstractmethod


class Model(ABC):

	def __init__(self, input_size=1, output_size=1):
		self._input_size = input_size
		self._output_size = output_size

	@abstractmethod
	def fit(self, x_train, y_train):
		pass

	@abstractmethod
	def predict(self, x):
		pass

	@abstractmethod
	def get_weights(self):
			pass
