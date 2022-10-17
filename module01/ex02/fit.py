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
def predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
	"""This version stacks the column of 1's before the x column"""
	ones = np.ones(shape=(x.shape[0]))
	new = np.column_stack((ones, x))
	return new.dot(theta)


@accepts(np.ndarray, np.ndarray)
def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
	return x.dot(theta)


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
	Raises:
	This function should not raise any Exception.
	"""
	if any(a.size == 0 for a in [x, y, theta]):
		return None
	return (x.T.dot(predict_(x, theta) - y)) / x.shape[0]


@accepts(np.ndarray, np.ndarray, np.ndarray, float, int)
def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
	"""
	Description:
	Fits the model to the training dataset contained in x and y.
	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient descent
	Returns:
	new_theta: numpy.ndarray, a vector of dimension 2 * 1.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exception.
	"""
	ones = np.ones_like(x)
	x = np.column_stack((ones, x))
	new_theta = theta.copy()
	for _ in range(max_iter):
		new_theta = new_theta - (alpha * simple_gradient(x, y, new_theta))
	return new_theta
