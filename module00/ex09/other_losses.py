import numpy as np
from math import sqrt


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
def mse_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
	"""
	Description:
	Calculate the MSE between the predicted output and the real output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	mse: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	return ((y_hat - y) ** 2).sum() / y.size


@accepts(np.ndarray, np.ndarray)
def rmse_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
	"""
	Description:
	Calculate the RMSE between the predicted output and the real output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	rmse: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	return sqrt(mse_(y_hat, y))


@accepts(np.ndarray, np.ndarray)
def mae_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
	"""
	Description:
	Calculate the MAE between the predicted output and the real output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	mae: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	return np.absolute(y_hat - y).sum() / y.size


@accepts(np.ndarray, np.ndarray)
def r2score_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
	"""
	Description:
	Calculate the R2score between the predicted output and the output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	r2score: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	numerator = ((y_hat - y) ** 2).sum()
	denominator = ((y - y.mean()) ** 2).sum()
	return 1 - (numerator / denominator)
