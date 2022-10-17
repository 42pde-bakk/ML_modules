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


@accepts(np.ndarray, np.ndarray)
def logistic_predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
	return sigmoid_(x.dot(theta))


@accepts(np.ndarray, np.ndarray, np.ndarray)
def log_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
	ones = np.ones(shape=(x.shape[0], 1))
	x = np.hstack((ones, x))
	y_hat = logistic_predict_(x, theta)
	result = []
	for j in range(x.shape[1]):
		total = 0
		for row_nb in range(x.shape[0]):
			total += (y_hat[row_nb][0] - y[row_nb][0]) * x[row_nb][j]
		result.append(total / y.shape[0])
	return np.array(result).reshape(-1, 1)
