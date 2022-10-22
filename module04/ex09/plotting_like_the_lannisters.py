import numpy as np
from matplotlib import pyplot as plt

FEATURES = ['height', 'weight', 'bone_density']


def plot_f1_scores(f1_scores: list, title: str, lambdas: np.ndarray) -> None:
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.title(title)
	plt.ylim(0.0, 1.0)
	plt.bar([f'{lamb:.1f}' for lamb in lambdas], f1_scores)

	plt.xlabel('Lambda values')
	plt.ylabel('F1 score')
	plt.show()


def plot_true_price(x: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> None:
	fig, axs = plt.subplots(nrows=2, ncols=2)
	for i, feature in enumerate(FEATURES):
		plot = axs[i // 2, i % 2]
		plot.set_title(feature)
		x_col = x[:, i]
		plot.scatter(x_col, y, label='True zipcodes')

		# y_hat = model.predict_(x_norm)
		plot.scatter(x_col, y_hat, label=f'Predicted zipcode')
		plot.legend(loc='best')
	plt.show()
