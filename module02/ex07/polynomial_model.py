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
	"""Add polynomial features to vector x by raising its values up to the power given in argument.
	Args:
	x: has to be an numpy.array, a vector of dimension m * 1.
	power: has to be an int, the power up to which the components of vector x are going to be raised.
	Return:
	The matrix of polynomial features as a numpy.array, of dimension m * n,
	containing the polynomial feature values for all training examples.
	None if x is an empty numpy.array.
	None if x or power is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	for _ in range(power - 1):
		first, last = x[:, 0], x[:, -1]
		new_column = (first * last).reshape(-1, 1)
		x = np.hstack((x, new_column))
	return x
