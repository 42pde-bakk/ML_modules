from __future__ import annotations

import numpy as np


def accepts(*types):
	def check_accepts(f):
		if len(types) != f.__code__.co_argcount:
			return None

		def new_f(*args, **kwargs):
			if any(not isinstance(arg, t) for arg, t in zip(args, types)):
				return None
			return f(*args, **kwargs)

		# new_f.__name__ = f.__name__
		return new_f

	return check_accepts


class MyLogisticRegression:
	"""
	Description: My personal logistic regression to classify things.
	If a function has the __ prefix, it means that it assumes the x value has a column of ones already...
	"""

	supported_penalities = ['l2']  # We consider l2 penality only. One may wants to implement other penalities.

	def __init__(self, theta: np.ndarray, alpha: float = 0.001, max_iter: int = 1000, penalty: str | None = 'l2', lambda_: float = 1.0):
		if not isinstance(theta, np.ndarray) or not isinstance(alpha, float) or not isinstance(max_iter, int) \
				or not (isinstance(penalty, str) or penalty is None) or not isinstance(lambda_, float):
			raise TypeError('Bad arguments given to MyLogisticRegression')
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.penalty = penalty
		self.lambda_ = lambda_ if penalty in self.supported_penalities else 0.0
		self.polynomial = 1
		self.zipcode: int = 0
		self.f1 = 0.0
		if self.penalty not in self.supported_penalities and self.penalty is not None:
			raise TypeError(f'Error. Please provide me with a valid penalty, or {None}')

	def get_params(self) -> dict:
		"""Get parameters for this estimator."""
		return vars(self)

	def set_params(self, **params) -> MyLogisticRegression:
		"""Set the parameters of this estimator."""
		for key, value in params.items():
			if key in vars(self).keys():
				setattr(self, key, value)
		return self

	@staticmethod
	@accepts(np.ndarray)
	def sigmoid_(x: np.ndarray) -> np.ndarray:
		"""
		Compute the sigmoid of a vector.
		Args:
		x: has to be a numpy.ndarray of shape (m, 1).
		Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
		Raises:
		This function should not raise any Exception.
		"""
		return 1 / (1 + np.exp(-x))

	def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
		Returns:
		The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
		Raises:
		This function should not raise any Exception.
		"""
		reg_thetas = self.__get_regularization_thetas()
		ones = np.ones(shape=(x.shape[0], 1))
		x = np.hstack((ones, x))
		y_hat = self.__predict_(x)
		return (x.T.dot(y_hat - y) + self.lambda_ * reg_thetas) / y.shape[0]

	def __gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		reg_thetas = self.__get_regularization_thetas()
		y_hat = self.__predict_(x)
		return (x.T.dot(y_hat - y) + self.lambda_ * reg_thetas) / y.shape[0]

	def __gradient_unregularized_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		y_hat = self.__predict_(x)
		return x.T.dot(y_hat - y) / y.shape[0]

	def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""
		:param x: np.ndarray
		:param y: np.ndarray
		:return: new theta values
		"""
		x = np.column_stack((np.ones(shape=(x.shape[0], 1)), x))
		if self.penalty == 'l2':
			for idx in range(self.max_iter):
				self.theta -= (self.alpha * self.__gradient_(x, y))
		else:
			for idx in range(self.max_iter):
				self.theta -= (self.alpha * self.__gradient_unregularized_(x, y))
		return self.theta

	@staticmethod
	@accepts(np.ndarray, np.ndarray, float)
	def loss_elem_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> np.ndarray | None:
		"""
		:param y: Actual values as an np.ndarray
		:param y_hat: Predicted values as an np.ndarray
		:param eps: very small value
		:return: np.ndarray of the losses
		"""
		if y.shape != y_hat.shape:
			return None
		return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

	@staticmethod
	@accepts(np.ndarray, np.ndarray, float)
	def loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float | None:
		"""
		Computes the logistic loss value.
		Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		eps: has to be a float, epsilon (default=1e-15)
		Returns:
		The logistic loss value as a float.
		None on any error.
		Raises:
		This function should not raise any Exception.
		"""
		loss_elem = MyLogisticRegression.loss_elem_(y, y_hat, eps)
		if loss_elem is None:
			return None
		return -np.sum(loss_elem) / y.shape[0]

	def __predict_(self, x: np.ndarray) -> np.ndarray:
		"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
		Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
		Raises:
		This function should not raise any Exception.
		"""
		return MyLogisticRegression.sigmoid_(x.dot(self.theta))

	def predict_(self, x: np.ndarray) -> np.ndarray | None:
		"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
		Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
		Raises:
		This function should not raise any Exception.
		"""
		if not isinstance(x, np.ndarray) or x.size == 0:
			return None
		ones = np.ones(shape=(x.shape[0], 1))
		x = np.hstack((ones, x))
		return MyLogisticRegression.sigmoid_(x.dot(self.theta))

	def __get_regularization_thetas(self) -> np.ndarray:
		new_theta = self.theta.copy()
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
		return (x - x.mean(axis=0)) / x.std(axis=0)

	@staticmethod
	def zscore_precomputed(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray | None:
		"""
		Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
		This version uses precomputed means and std_deviations so we can apply the same standardization to
		both the training and the test set.
		:param x: np.ndarray
		:param mean: np.ndarray (result of np.ndarray.mean(axis=0)
		:param std: np.ndarray (result of np.ndarray.std(axis=0)
		:return: normalized version of a non-empty numpy.ndarray using the z-score standardization.
		"""
		if not isinstance(x, np.ndarray) or x.size == 0:
			return
		return (x - mean) / std
