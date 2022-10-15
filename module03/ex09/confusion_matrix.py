import numpy as np
from typing import Any
import pandas as pd


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


@accepts(np.ndarray, np.ndarray, (list, None), bool)
def confusion_matrix_(y_true: np.ndarray, y_hat: np.ndarray, labels=None, df_option: bool = False) -> np.ndarray | pd.DataFrame | None:
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
	y:a numpy.array for the correct labels
	y_hat:a numpy.array for the predicted labels
	labels: optional, a list of labels to index the matrix.
	This may be used to reorder or select a subset of labels. (default=None)
	df_option: optional, if set to True the function will return a pandas DataFrame
	instead of a numpy array. (default=False)
	Return:
	The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
	None if any error.
	Raises:
	This function should not raise any Exception.
	"""
	if y_true.shape != y_hat.shape:
		return None
	if labels:
		len_labels = len(labels)
	else:
		labels = np.unique(np.vstack((y_true, y_hat)))
		len_labels = labels.shape[0]
	matrix = np.zeros(shape=(len_labels, len_labels))

	for i, predicted_label in enumerate(labels):
		for j, true_label in enumerate(labels):
			view = (y_hat == predicted_label) & (y_true == true_label)
			matrix[j][i] += np.sum(view)

	if df_option:
		return pd.DataFrame(data=matrix, index=labels, columns=labels, dtype=int)
	return matrix
