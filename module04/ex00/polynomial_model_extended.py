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


@accepts(np.ndarray, int)
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
