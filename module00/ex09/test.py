import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from other_losses import mse_, rmse_, mae_, r2score_


def main() -> None:
	# Example 1:
	x = np.array([0, 15, -9, 7, 12, 3, -21])
	y = np.array([2, 14, -13, 5, 12, 4, -19])

	# Mean squared error
	assert mse_(x, y) == mean_squared_error(x, y), f'I return {mse_(x, y)}'
	# 4.285714285714286

	# Root mean squared error
	assert rmse_(x, y) == sqrt(mean_squared_error(x, y)), f'I return {rmse_(x, y)}'
	# 2.0701966780270626

	# Mean absolute error
	assert mae_(x, y) == mean_absolute_error(x, y), f'I return {mae_(x, y)}'
	# 1.7142857142857142

	# R2-score
	assert r2score_(x, y) == r2_score(x, y), f'I return {r2score_(x, y)}'
	# 0.9681721733858745


if __name__ == '__main__':
	main()
