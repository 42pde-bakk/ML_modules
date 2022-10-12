import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR
from matplotlib import pyplot as plt


def plot(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
	plt.scatter(x, y, color='b', label='$S_{true}$(pills)')
	plt.scatter(x, y_hat, color='limegreen', label='$S_{predict}$(pills)')
	plt.plot(x, y_hat, 'limegreen', linestyle='--')

	plt.grid(visible=True)
	plt.xlabel('Quantity of blue pill (in micrograms)')
	plt.ylabel('Space driving score')
	plt.legend(loc='upper center')
	plt.show()


def plot_loss(x: np.ndarray, y: np.ndarray, thetas: np.ndarray) -> None:
	lr = MyLR(thetas)
	plt.ylim(0, 150)
	plt.xlim(-14.4, -3.6)
	theta0_list = np.linspace(thetas[0] - 15, thetas[0] + 15, 6).reshape(-1, 1)
	theta1_list = np.arange(-18, -2, step=0.01).reshape(-1, 1)

	cmap = plt.get_cmap('Greys_r')
	colors = cmap(np.linspace(0, 1, 10))
	for idx, (theta0, color) in enumerate(zip(theta0_list, colors)):
		losses = []
		for _, theta1 in np.ndenumerate(theta1_list):
			lr.thetas = np.array([[theta0], [theta1]], dtype=object)
			loss = lr.mse_(y, lr.predict_(x))
			losses.append(loss)
		plt.plot(theta1_list, losses, color=color, label=f'J((θ0=c{idx},θ1)')
	plt.grid(visible=True)
	plt.ylabel('cost function J($θ_{0}$,$θ_{1}$)')
	plt.xlabel('$θ_{1}$')
	plt.legend(loc='lower right')
	plt.show()


def main() -> None:
	try:
		data = pd.read_csv('../resources/are_blue_pills_magic.csv')
	except (FileNotFoundError, pd.errors.EmptyDataError) as e:
		print('Error. Please supply a valid path to the csv file')
		exit(1)
	Xpill = np.array(data['Micrograms']).reshape(-1, 1)
	Yscore = np.array(data['Score']).reshape(-1, 1)
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	y_hat1 = linear_model1.predict_(Xpill)

	linear_model1.fit_(Xpill, Yscore)
	y_hat2 = linear_model1.predict_(Xpill)

	plot(Xpill, Yscore, y_hat2)
	plot_loss(Xpill, Yscore, linear_model1.thetas)


if __name__ == '__main__':
	main()
