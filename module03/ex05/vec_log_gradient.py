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


@accepts(np.ndarray, np.ndarray, np.ndarray)
def vec_log_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
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
	y_hat = sigmoid_(x.dot(theta))

	return x.T.dot(y_hat - y) / y.shape[0]
