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
def iterative_l2(theta: np.ndarray) -> float:
	"""Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	total = 0.0
	new_theta = theta.copy()
	new_theta[0][0] = 0
	for idx, th in np.ndenumerate(new_theta):
		total += th * th
	return total


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
