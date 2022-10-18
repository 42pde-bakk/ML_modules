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


@accepts(np.ndarray, np.ndarray, float)
def log_loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> np.ndarray:
	"""
	Computes the logistic loss value.
	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	eps: has to be a float, epsilon (default=1e-15)
	Returns:
	The logistic loss value as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	m = y.shape[0]
	result = y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
	return -np.sum(result) / m
