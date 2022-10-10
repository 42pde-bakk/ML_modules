import numpy as np


def check_types(args, types) -> bool:
	if len(args) != len(types):
		return False
	return any(not isinstance(arg, t) for arg, t in zip(args, types))


class MyLinearRegression:
	"""
	Description: My personal linear regression class to fit like a boss.
	"""

	def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 1000):
		self.thetas = thetas.copy()
		self.alpha = alpha
		self.max_iter = max_iter

	def better_predict_(self, x: np.ndarray) -> np.ndarray:
		"""This function assumes you have a column of 1's and a column of X"""
		return x.dot(self.thetas)

	def gradient_(self, x: np.ndarray, y: np.ndarray):
		return (x.T.dot(self.better_predict_(x) - y)) / x.shape[0]

	def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		ones = np.ones_like(x)
		x = np.column_stack((ones, x))
		for _ in range(self.max_iter):
			self.thetas = self.thetas - (self.alpha * self.gradient_(x, y))
		return self.thetas

	def predict_(self, x: np.ndarray) -> np.ndarray | None:
		ones = np.ones(shape=(x.shape[0]))
		new = np.column_stack((ones, x))
		return new.dot(self.thetas)

	def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
		return np.square(y_hat - y)

	def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
		return np.square(y_hat - y).sum() / (2 * y.shape[0])
