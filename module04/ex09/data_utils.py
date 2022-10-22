from typing import Tuple

import numpy as np

from my_logistic_regression import MyLogisticRegression as MyLogR


def add_polynomials_and_normalize(model: MyLogR, x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	x_train_new = MyLogR.add_polynomial_features(x_train, model.polynomial)
	x_test_new = MyLogR.add_polynomial_features(x_test, model.polynomial)

	return normalize_data(x_train_new, x_test_new)


def normalize_data(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	mean, std = x_train.mean(axis=0), x_train.std(axis=0)
	normalized_x_train = MyLogR.zscore_precomputed(x_train, mean, std)
	normalized_x_test = MyLogR.zscore_precomputed(x_test, mean, std)
	return normalized_x_train, normalized_x_test
