import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from other_losses import mse_, rmse_, mae_, r2score_


def main() -> None:
	# Example 1:
	x = np.array([0, 15, -9, 7, 12, 3, -21])
	y = np.array([2, 14, -13, 5, 12, 4, -19])

	# Mean squared error
	mse = mse_(x, y)
	print(f'{mse = }')
	assert mse == mean_squared_error(x, y), f'I return {mse}'
	# 4.285714285714286

	# Root mean squared error
	rmse = rmse_(x, y)
	print(f'{rmse = }')
	assert rmse == sqrt(mean_squared_error(x, y)), f'I return {rmse}'
	# 2.0701966780270626

	# Mean absolute error
	mae = mae_(x, y)
	print(f'{mae = }')
	assert mae == mean_absolute_error(x, y), f'I return {mae}'
	# 1.7142857142857142

	# R2-score
	r2 = r2score_(x, y)
	print(f'R2-score = {r2}')
	assert r2 == r2_score(x, y), f'I return {r2}'
	# 0.9681721733858745


if __name__ == '__main__':
	main()
