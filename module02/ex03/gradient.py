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


@accepts(np.ndarray, np.ndarray)
def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
	"""Computes the prediction vector y_hat from two non-empty numpy.array.
	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
	Return:
	y_hat as a numpy.array, a vector of dimension m * 1.
	None if x or theta are empty numpy.array.
	None if x or theta dimensions are not matching.
	None if x or theta is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	return x.dot(theta)


@accepts(np.ndarray, np.ndarray, np.ndarray)
def gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
	"""Computes a gradient vector from three non-empty numpy.array, without any for-loop.
	The three arrays must have the compatible dimensions.
	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	y: has to be an numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector (n +1) * 1.
	Return:
	The gradient as a numpy.array, a vector of dimensions n * 1,
	containg the result of the formula for all j.
	None if x, y, or theta are empty numpy.array.
	None if x, y and theta do not have compatible dimensions.
	None if x, y or theta is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	ones = np.ones(shape=(x.shape[0], 1))
	x = np.hstack((ones, x))
	return x.T.dot(x.dot(theta) - y) / x.shape[0]
