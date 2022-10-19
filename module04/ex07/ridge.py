from __future__ import annotations

import numpy as np

from my_linear_regression import MyLinearRegression


class MyRidge(MyLinearRegression):
	"""
	Description: My personal linear regression class to fit like a boss.
	"""

	def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 1000, lambda_: float | int = 0.5):
		super().__init__(thetas, alpha, max_iter)
		self.lambda_ = lambda_
		self.polynomial = 1
		self.loss = 0

	def get_params(self) -> dict:
		"""Get parameters for this estimator."""
		return vars(self)

	def set_params(self, **params) -> MyRidge:
		"""Set the parameters of this estimator."""
		for key, value in params.items():
			# if getattr(self, key):
			if key in vars(self).keys():
				setattr(self, key, value)
		return self

	def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
		without any for-loop. The three arrays must have compatible shapes.
		Args:
			x: has to be a numpy.ndarray, a matrix of dimesion m * n.
			y: has to be a numpy.ndarray, a vector of shape m * 1.
		Return:
			A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
			None if y, x, or theta are empty numpy.ndarray.
			None if y, x or theta does not share compatibles shapes.
			None if y, x or theta or lambda_ is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""
		new_thetas = self.__copy_thetas_first0()

		ones = np.ones((y.shape[0], 1))
		x_ = np.hstack((ones, x))
		y_hat = x_.dot(self.thetas)

		return (np.dot(x_.T, (y_hat - y)) + (self.lambda_ * new_thetas)) / y.shape[0]

	def __gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		new_thetas = self.__copy_thetas_first0()
		y_hat = x.dot(self.thetas)
		return (np.dot(x.T, (y_hat - y)) + self.lambda_ * new_thetas) / y.shape[0]

	def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
		if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or x.size == 0 or y.size == 0:
			return None
		ones = np.ones(shape=(x.shape[0], 1))
		x_ = np.column_stack((ones, x))
		for _ in range(self.max_iter):
			self.thetas = self.thetas - (self.alpha * self.__gradient_(x_, y))
		return self.thetas

	def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
		"""
		:param y: numpy array of shape=(m, 1)
		:param y_hat: numpy array of shape(m, 1)
		:return: array of the squared differences with l2 regularization
		"""
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or y.size == 0 or y_hat.size == 0:
			return None
		return np.square(y_hat - y) + self.lambda_ * self.l2()

	def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
		"""Computes the regularized loss of a linear regression model from two non-empty numpy.array.
		Args:
			y: has to be an numpy.ndarray, a vector of shape m * 1.
			y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		Returns:
			The regularized loss as a float.
			None if y, y_hat, or theta are empty numpy.ndarray.
			None if y and y_hat do not share the same shapes.
		Raises:
			This function should not raise any Exception.
		"""
		diff = y_hat - y
		return np.sum(np.dot(diff.T, diff) + self.lambda_ * self.l2()) / (2 * y.shape[0])

	def mse_(self, y: np.ndarray, y_hat: np.ndarray) -> float | None:
		diff = y_hat - y
		return (np.dot(diff.T, diff) + self.lambda_ * self.l2()) / y.shape[0]

	def l2(self) -> float:
		"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
		Returns:
			The L2 regularization as a float.
			None if theta in an empty numpy.ndarray.
		Raises:
			This function should not raise any Exception.
		"""
		new_theta = self.__copy_thetas_first0()
		return new_theta.T.dot(new_theta).sum()

	def __copy_thetas_first0(self) -> np.ndarray:
		new_theta = self.thetas.copy()
		new_theta[0][0] = 0
		return new_theta

	@staticmethod
	def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
		"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power given
		Args:
			x: has to be an numpy.ndarray, a matrix of shape m * n.
			power: has to be an int, the power up to which the columns of matrix x are going to be raised.
		Returns:
			The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature values
			None if x is an empty numpy.ndarray.
		Raises:
			This function should not raise any Exception.
		"""
		x_new = x.copy()
		for p in range(2, power + 1):
			x_new = np.hstack((x_new, x ** p))
		return x_new

	def score(self, x: np.ndarray, y: np.ndarray) -> float:
		y_hat = self.predict_(x)
		u = np.square(y - y_hat).sum()
		v = np.square(y - y).sum()
		return 1 - u / v
