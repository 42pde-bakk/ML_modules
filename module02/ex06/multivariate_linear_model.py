import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
from matplotlib import pyplot as plt


MYDICT = {
	'Age': {
		'colours': ('midnightblue', 'dodgerblue'),
		'label': '$x_{1}$: age (in years)'
	},
	'Thrust_power': {
		'colours': ('forestgreen', 'lime'),
		'label': '$x_{2}$: thrust power (in 10 KM/s)'
	},
	'Terameters': {
		'colours': ('darkviolet', 'violet'),
		'label': '$x_{3}$: distance totalizer value of spacecraft (in Tmeters)'
	}
}


def plot_univariate(x: np.ndarray, y: np.ndarray, feature: str, alpha: float, max_iter: int) -> None:
	colours = MYDICT[feature]['colours']
	initial_thetas = np.ones((2, 1), dtype=float)
	lr = MyLR(initial_thetas, alpha=alpha, max_iter=max_iter)
	lr.fit_(x, y)
	print(f'Univariate thetas = {lr.thetas}')
	y_hat = lr.predict_(x)

	plt.scatter(x, y, color=colours[0], label='Sell price')
	plt.scatter(x, y_hat, color=colours[1], label='Predicted sell price')
	plt.xlabel(MYDICT[feature]['label'])
	plt.ylabel('y: sell price (in keuros)')

	plt.legend(loc='best')
	plt.show()


def plot_multivariate(x, y, y_hat, feature: str):
	colours = MYDICT[feature]['colours']
	plt.scatter(x, y, color=colours[0], label='Sell price')
	plt.scatter(x, y_hat, color=colours[1], label='Predicted sell price')
	plt.xlabel(MYDICT[feature]['label'])
	plt.ylabel('y: sell price (in keuros)')

	plt.legend(loc='best')
	plt.show()


def multivariate(data: np.ndarray) -> None:
	features = ['Age', 'Thrust_power', 'Terameters']
	x = data[features].to_numpy().reshape(-1, 3)
	y = data['Sell_price'].to_numpy().reshape(-1, 1)
	lr = MyLR(np.ones((4, 1), dtype=float), alpha=0.00001, max_iter=1_500_000)
	lr.fit_(x, y)

	print(f'Multivariate thetas = {lr.thetas}')

	for idx, feature in enumerate(features):
		plot_multivariate(x[:, idx], y, lr.predict_(x), feature)


def test_univariate(data: pd.DataFrame) -> None:
	plot_univariate(
		data['Age'].to_numpy().reshape(-1, 1),
		data['Sell_price'].to_numpy().reshape(-1, 1),
		feature='Age',
		alpha=0.001,
		max_iter=25000
	)
	plot_univariate(
		data['Thrust_power'].to_numpy().reshape(-1, 1),
		data['Sell_price'].to_numpy().reshape(-1, 1),
		feature='Thrust_power',
		alpha=0.0001,
		max_iter=1000
	)
	plot_univariate(
		data['Terameters'].to_numpy().reshape(-1, 1),
		data['Sell_price'].to_numpy().reshape(-1, 1),
		feature='Terameters',
		alpha=0.0001,
		max_iter=100000
	)


if __name__ == '__main__':
	csv_path = '../resources/spacecraft_data.csv'
	spacecraft_data = pd.read_csv(csv_path)
	test_univariate(spacecraft_data)
	multivariate(spacecraft_data)
