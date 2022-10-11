import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
from matplotlib import pyplot as plt


def univariate(data: np.ndarray, x_feature: str, y_feature: str, xlabel: str, col: tuple, alpha: float, max_iter: int) -> None:
	initial_thetas = np.array([[1.0], [1.0]], dtype=np.float64)
	x = data[x_feature].to_numpy().reshape(-1, 1)
	y = data[y_feature].to_numpy().reshape(-1, 1)
	lr = MyLR(initial_thetas, alpha=alpha, max_iter=max_iter)
	lr.fit_(x, y)
	y_hat = lr.predict_(x)

	plt.scatter(x, y, color=col[0], label='Sell price')
	plt.scatter(x, y_hat, color=col[1], label='Predicted sell price')
	plt.xlabel(xlabel)
	plt.ylabel('y: sell price (in keuros)')
	plt.show()


def main() -> None:
	data = pd.read_csv('spacecraft_data.csv')
	univariate(
		data,
		'Age',
		'Sell_price',
		'$x_{1}: age (in years)',
		('midnightblue', 'dodgerblue'),
		alpha=0.001,
		max_iter=25000
	)
	univariate(
		data,
		'Thrust_power',
		'Sell_price',
		'$x_{2}: thrust power (in 10 KM/s)',
		('forestgreen', 'lime'),
		alpha=0.0001,
		max_iter=1000
	)
	univariate(
		data,
		'Terameters',
		'Sell_price',
		'$x_{3}: distance totalizer value of spacecraft (in Tmeters)',
		('darkviolet', 'violet'),
		alpha=0.0001,
		max_iter=100000
	)


if __name__ == '__main__':
	main()
