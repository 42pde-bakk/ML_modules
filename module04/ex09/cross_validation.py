import sys

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
def build_cross_validation_sets(x: np.ndarray, y: np.ndarray, proportion: float) -> list | None:
	"""
	:param x: np.ndarray of shape m * n
	:param y: np.ndarray of shape m * 1
	:param proportion: non-null postive float lower than 1.0
	:return: list of dictionaries containing the training and testing datasets for x and y
	"""
	if x.shape[0] != y.shape[0]:
		print(f'Why are you giving me arrays of differing sizes?', file=sys.stderr)
		return
	test_proportion: float = 1.0 - proportion
	cross_validation_sets = []
	for start in np.arange(0.0, 1.0, step=test_proportion):
		end = start + test_proportion
		if end > 1.0:
			print(f'end is bigger than 1.0')
			break
		x_train_set_0, x_test_set, x_train_set_1 = np.split(x, [int(start * x.shape[0]), int(end * x.shape[0])])
		y_train_set_0, y_test_set, y_train_set_1 = np.split(y, [int(start * y.shape[0]), int(end * y.shape[0])])

		entry = {
			'x': {
				'training': np.vstack((x_train_set_0, x_train_set_1)),
				'testing': x_test_set
			},
			'y': {
				'training': np.vstack((y_train_set_0, y_train_set_1)),
				'testing': y_test_set
			}
		}
		cross_validation_sets.append(entry)

	return cross_validation_sets
