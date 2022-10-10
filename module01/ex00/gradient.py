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
def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
	ones = np.ones(shape=(x.shape[0]))
	new = np.column_stack((ones, x))
	return new.dot(theta)


@accepts(np.ndarray, np.ndarray, np.ndarray)
def simple_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
	"""Computes a gradient vector from three non-empty numpy.array, with a for-loop.
	The three arrays must have compatible shapes.
	Args:
	x: has to be an numpy.array, a vector of shape m * 1.
	y: has to be an numpy.array, a vector of shape m * 1.
	theta: has to be an numpy.array, a 2 * 1 vector.
	Return:
	The gradient as a numpy.array, a vector of shape 2 * 1.
	None if x, y, or theta are empty numpy.array.
	None if x, y and theta do not have compatible shapes.
	None if x, y or theta is not of the expected type.
	Raises:
	This function should not raise any Exception.
	"""
	preds = predict_(x, theta)
	gradients = [
		(preds - y).sum() / y.size,
		((preds - y) * x).sum() / y.size
	]
	return np.array(gradients).reshape(theta.shape)
