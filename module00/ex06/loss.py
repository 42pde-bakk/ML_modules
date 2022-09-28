import numpy
import numpy as np


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
def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
	"""
	Description:
	Calculates all the elements (y_pred - y)^2 of the loss function.
	Args:
	y: has to be an numpy.array, a vector.
	y_hat: has to be an numpy.array, a vector.
	Returns:
	J_elem: numpy.array, a vector of dimension (number of the training examples,1).
	None if there is a dimension matching problem between X, Y or theta.
	None if any argument is not of the expected type.
	Raises:
	This function should not raise any Exception.
	"""
	return np.array([(real - pred) ** 2 for real, pred in zip(y, y_hat)])


@accepts(np.ndarray, np.ndarray)
def loss_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
	"""
	Description:
	Calculates the value of loss function.
	Args:
	y: has to be a numpy.array, a vector.
	y_hat: has to be a numpy.array, a vector.
	Returns:
	J_value : has to be a float.
	None if there is a dimension matching problem between X, Y or theta.
	None if any argument is not of the expected type.
	Raises:
	This function should not raise any Exception.
	"""
	return loss_elem_(y, y_hat).sum() / (2 * y.size)
