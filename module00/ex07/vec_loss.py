import numpy as np

def accepts(*types):
	def check_accepts(f):
		assert len(types) == f.__code__.co_argcount

		def new_f(*args, **kwargs):
			if any(not isinstance(arg, t) for arg, t in zip(args, types)):
				return None
			return f(*args, **kwargs)
		# new_f.__name__ = f.__name__
		return new_f
	return check_accepts


@accepts(np.ndarray, np.ndarray)
def loss_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
	"""Computes the half mean squared error of two non-empty numpy.array, without any for loop.
	The two arrays must have the same dimensions.
	Args:
	y: has to be a numpy.array, a vector.
	y_hat: has to be a numpy.array, a vector.
	Returns:
	The half mean squared error of the two vectors as a float.
	None if y or y_hat are empty numpy.array.
	None if y and y_hat does not share the same dimensions.
	Raises:
	This function should not raise any Exceptions.
	"""
	if y.size == 0 or y.shape != y_hat.shape:
		return None
	return ((y_hat - y) ** 2).sum() / (2 * y.size)
