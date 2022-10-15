import numpy as np


class MyLinearRegression:
	"""
	Description: My personal linear regression class to fit like a boss.
	"""

	def __init__(self, thetas: np.ndarray, alpha: float = 0.0001, max_iter: int = 200000):
		if not isinstance(thetas, np.ndarray) or thetas.size == 0:
			raise TypeError('Bad Thetas given to MyLinearRegression.__init__()')
		if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
			raise TypeError('Bad alpha given to MyLinearRegression.__init__()')
		if not isinstance(max_iter, int) or max_iter <= 0:
			raise TypeError('Bad max_iter given to MyLinearRegression.__init__()')
		self.thetas = thetas.copy()
		self.alpha = alpha
		self.max_iter = max_iter

	def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		# return (x.T.dot(self.__predict(x) - y)) / x.shape[0]
		first_step = x.dot(self.thetas)
		second_step = first_step - y
		third_step = x.T.dot(second_step)
		final_step = third_step / x.shape[0]
		return x.T.dot(x.dot(self.thetas) - y) / x.shape[0]

	def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
		if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or x.size == 0 or y.size == 0:
			return None
		ones = np.ones(shape=(x.shape[0], 1))
		x = np.column_stack((ones, x))
		print(f'new {x.shape = }')
		for idx in range(self.max_iter):
			self.thetas = self.thetas - (self.alpha * self.gradient_(x, y))
		return self.thetas

	def __predict(self, x: np.ndarray) -> np.ndarray:
		"""This function assumes you have a column of 1's and a column of X"""
		return x.dot(self.thetas)

	def predict_(self, x: np.ndarray) -> np.ndarray | None:
		"""And this function does not"""
		if not isinstance(x, np.ndarray) or x.size == 0:
			return None
		ones = np.ones(shape=(x.shape[0], 1))
		x = np.hstack((ones, x))
		result = x.dot(self.thetas)
		return result

	@staticmethod
	def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or y.size == 0 or y_hat.size == 0:
			return None
		return np.square(y_hat - y)

	@staticmethod
	def loss_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or y.size == 0 or y_hat.size == 0:
			return None
		return np.square(y_hat - y).sum() / (2 * y.shape[0])

	@staticmethod
	def mse_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or y.size == 0 or y_hat.size == 0:
			return None
		return np.square(y_hat - y).sum() / y.shape[0]

	@staticmethod
	def zscore(x: np.ndarray) -> np.ndarray | None:
		"""Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
		Args:
		x: has to be an numpy.ndarray, a vector.
		Returns:
		x’ as a numpy.ndarray.
		None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
		Raises:
		This function shouldn’t raise any Exception.
		"""
		if not isinstance(x, np.ndarray) or x.size == 0:
			return
		return (x - x.mean()) / x.std()
