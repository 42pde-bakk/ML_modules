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
def l2(theta: np.ndarray) -> float:
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	new_theta = theta.copy()
	new_theta[0][0] = 0
	return new_theta.T.dot(new_theta).sum()


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

	inner = y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
	return (-1 / m) * np.sum(inner) + lambda_ * l2(theta) / (2 * m)


@accepts(np.ndarray, np.ndarray, np.ndarray, float | int)
def reg_linear_grad(y: np.ndarray, x: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
	with two for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	new_thetas = np.copy(theta)
	new_thetas[0][0] = 0.0

	ones = np.ones((y.shape[0], 1))
	x_ = np.hstack((ones, x))
	y_hat = x_.dot(theta)
	out = []

	for j in range(x_.shape[1]):
		total = 0.0
		for i in range(x_.shape[0]):
			real, pred = y[i][0], y_hat[i][0]
			total += (pred - real) * x_[i][j]
		out.append((total + lambda_ * new_thetas[j][0]) / y.shape[0])
	return np.array(out).reshape(-1, 1)


@accepts(np.ndarray, np.ndarray, np.ndarray, float | int)
def vec_reg_linear_grad(y: np.ndarray, x: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
	without any for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	new_thetas = np.copy(theta)
	new_thetas[0][0] = 0.0

	ones = np.ones((y.shape[0], 1))
	x_ = np.hstack((ones, x))
	y_hat = x_.dot(theta)

	return (np.dot(x_.T, (y_hat - y)) + lambda_ * new_thetas) / y.shape[0]
