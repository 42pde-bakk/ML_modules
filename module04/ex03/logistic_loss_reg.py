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
def reg_log_loss_(y: np.ndarray, y_hat: np.ndarray, theta: np.ndarray, lambda_: float):
	"""Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for lArgs:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta is empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	m = y.shape[0]
	eps = 1e-15
	y_hat += eps
	new_thetas = theta.copy()
	new_thetas[0][0] = 0

	inner = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
	theta_product = np.sum(np.dot(new_thetas.T, new_thetas))
	return (-1 / m) * np.sum(inner) + (lambda_ * theta_product / (2 * m))
