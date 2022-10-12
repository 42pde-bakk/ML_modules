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
	# print(f'{x=}')
	# print(f'{theta=}')
	ones = np.ones(shape=(x.shape[0], 1))
	x = np.hstack((ones, x))
	return sigmoid_(x.dot(theta))
