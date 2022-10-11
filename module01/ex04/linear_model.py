import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR
from matplotlib import pyplot as plt
import matplotlib.pylab as pl


def plot(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
	plt.scatter(x, y, color='b', label='true(pills)')
	plt.scatter(x, y_hat, color='limegreen', label='predict(pills)')
	plt.plot(x, y_hat, 'limegreen', linestyle='--')
	plt.grid(visible=True)
	plt.xlabel('Quantity of blue pill (in micrograms)')
	plt.ylabel('Space driving score')
	plt.show()


def plot_loss(x: np.ndarray, y_hat: np.ndarray, thetas: np.ndarray) -> None:
	plt.ylim(0, 150)
	theta0 = np.linspace(thetas[0] - 20, thetas[0] + 20, 6).reshape(-1, 1)

	plt.grid(visible=True)
	plt.show()


def main() -> None:
	data = pd.read_csv('are_blue_pills_magic.csv')
	Xpill = np.array(data['Micrograms']).reshape(-1, 1)
	Yscore = np.array(data['Score']).reshape(-1, 1)
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	# y_hat1 = linear_model1.predict_(Xpill)

	linear_model1.fit_(Xpill, Yscore)
	y_hat2 = linear_model1.predict_(Xpill)

	# plot(Xpill, Yscore, y_hat2)
	plot_loss(Xpill, Yscore, linear_model1.thetas)


if __name__ == '__main__':
	main()
