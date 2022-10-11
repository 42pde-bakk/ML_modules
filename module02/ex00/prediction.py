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
	ones = np.ones(shape=(x.shape[0], 1))
	x = np.hstack((ones, x))
	result = x.dot(theta)
	return result
