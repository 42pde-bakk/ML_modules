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


@accepts(np.ndarray, np.ndarray, np.ndarray, float)
def reg_loss_(y: np.ndarray, y_hat: np.ndarray, theta: np.ndarray, lambda_: float) -> float:
	"""Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta are empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	new_theta = theta.copy()
	new_theta[0][0] = 0
	tmp = y_hat - y
	return (np.dot(tmp.T, tmp) + lambda_ * np.dot(new_theta.T, new_theta)).sum() / (2 * y.shape[0])
