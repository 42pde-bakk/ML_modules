from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLogR


FEATURES = ['weight', 'prod_distance', 'time_delivery']


def plot_f1_scores(f1_scores: list | np.ndarray, title: str, lambdas: list) -> None:
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.title(title)
	plt.plot(range(1, len(f1_scores) + 1), f1_scores)
	print(f1_scores)

	plt.xticks(range(len(lambdas)), [f'{l:.1f}' for l in lambdas])

	plt.xlabel('Lambda values')
	plt.ylabel('F1 score')
	plt.show()


def plot_evaluation_curve(models: list[MyLogR]) -> None:
	xticks = [f'Pol {model.polynomial} & λ = {model.lambda_:.1f}' for model in models]
	losses = [model.f1 for model in models]
	plt.title(f'Evaluation metrics of models')

	plt.xticks(range(len(models)), xticks, rotation=270)
	plt.ylabel('Loss score')
	plt.xlabel('Polynomials + Lambda (λ) value')
	plt.plot(range(len(models)), losses)

	plt.show()


def setup_quad_plot(title: str, x_true: np.ndarray, y_true: np.ndarray):
	dim_axs: np.ndarray[plt.Axes]
	_, dim_axs = plt.subplots(nrows=2, ncols=2)
	axes = dim_axs
	plt.title(title)
	for idx, feature in enumerate(FEATURES):
		axes[idx].scatter(x_true[:, idx], y_true, label='True target prices')
		axes[idx].set_xlabel(feature)
		axes[idx].set_ylabel('Target price')
	plt.legend(loc='best')
	return axes


def plot(axes: np.ndarray[plt.Axes], model: MyLogR, x: np.ndarray) -> None:
	x_pol = MyLogR.add_polynomial_features(x, model.polynomial)
	x_norm = MyLogR.zscore(x_pol)
	y_hat = model.predict_(x_norm)

	for idx, feature in enumerate(FEATURES):
		axes[idx].scatter(x[:, idx], y_hat, alpha=0.3, label=f'Pol{model.polynomial}-λ{model.lambda_}')
	plt.legend(loc='best')


def plot_true_price(models: list[MyLogR], x: np.ndarray, x_norm, y: np.ndarray) -> None:
	for i, feature in enumerate(FEATURES):
		plt.title(f'{feature} vs the true and predicted prices')
		x_col = x[:, i]
		plt.scatter(x_col, y, label='True prices')
		for model in models:
			y_hat = model.predict_(x_norm)
			print(f'feature {feature}, lambda_={model.lambda_:.1f}, y_hat[0] = {y_hat[0]}')
			plt.scatter(x_col, y_hat, label=f'Predicted price with lambda_={model.lambda_:.1f}')
		plt.legend(loc='best')
		plt.show()
		break
