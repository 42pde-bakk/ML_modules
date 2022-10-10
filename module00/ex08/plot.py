import numpy as np
from matplotlib import pyplot as plt


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
def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
	ones = np.ones(shape=(x.shape[0]))
	new = np.column_stack((ones, x))
	return new.dot(theta)


@accepts(np.ndarray, np.ndarray, np.ndarray)
def plot_with_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1.
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exception.
	"""
	plt.scatter(x, y, color='b')
	y_hat = predict_(x, theta)
	plt.plot(x, y_hat, 'orange')
	plt.plot((x, x), [y, y_hat], 'r--')
	plt.show()
