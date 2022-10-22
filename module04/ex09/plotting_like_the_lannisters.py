from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLogR


FEATURES = ['weight', 'prod_distance', 'time_delivery']


def plot_f1_scores(f1_scores: list, title: str, lambdas: np.ndarray) -> None:
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.title(title)
	plt.plot([f'{l:.1f}' for l in lambdas], f1_scores)
	print(f1_scores)

	# plt.xticks(range(len(lambdas)), )

	plt.xlabel('Lambda values')
	plt.ylabel('F1 score')
	plt.show()


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
