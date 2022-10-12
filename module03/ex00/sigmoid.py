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


def real_sigmoid(x: np.ndarray, lvalue: int | float, k: int | float, x0: int | float = 0) -> np.ndarray:
	"""
	:param x: numpy array
	:param lvalue: the curve's maximum value
	:param k: the logistic growth rate or steepness of the curve
	:param x0: the midpoint of the curve, normally being 0
	:return: numpy array with all values transformed into [0,1] range
	"""
	return lvalue / (1 + np.exp(-k * (x - x0)))
