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
		size = 15
		plot.scatter(x_col, y, label='True zipcodes', s=5*size)
		plot.scatter(x_col, y_hat, label=f'Predicted zipcode', s=2*size)
		plot.legend(loc='best')

	axs[1, 1].text(0.1, 0.8, 'github.com/', fontsize=20)
	axs[1, 1].text(0.1, 0.5, '42pde-bakk/', fontsize=20)
	axs[1, 1].text(0.1, 0.2, 'ml_modules', fontsize=20)
	plt.show()
