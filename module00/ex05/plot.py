import numpy as np
from matplotlib import pyplot as plt


def pred(thetas: np.ndarray, x: int | float) -> int | float:
	return thetas[0] + x * thetas[1]


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
	"""Plot the data and prediction line from three non-empty numpy.arrays.
	Args:
	x: has to be a numpy.array, a vector of dimension m * 1.
	y: has to be a numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exceptions.
	"""
	if any(not isinstance(arg, np.ndarray) or arg.size == 0 for arg in (x, y, theta)):
		return None
	plt.plot(x, y, 'bo')
	xmin, xmax = np.min(x), np.max(x)
	predictions = np.array([pred(theta, xmin), pred(theta, xmax)])
	plt.plot([xmin, xmax], predictions, color='orange')
	plt.show()
