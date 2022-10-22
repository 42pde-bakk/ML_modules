import numpy as np
from matplotlib import pyplot as plt

from ridge import MyRidge

FEATURES = ['weight', 'prod_distance', 'time_delivery']


def plot_mse_scores(losses: list[float], title: str):
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.title(title)
	print(len(losses))
	print(losses)
	plt.plot(range(1, len(losses) + 1), losses)
	plt.xlabel('Model nb')
	plt.ylabel('Loss score')
	plt.show()


def plot_evaluation_curve(models: list[MyRidge]) -> None:
	xticks = [f'Pol {model.polynomial} & λ = {model.lambda_:.1f}' for model in models]
	losses = [model.loss for model in models]
	plt.title(f'Evaluation metrics of models')

	plt.xticks(range(len(models)), xticks, rotation=270)
	plt.ylabel('Loss score')
	plt.xlabel('Polynomials + Lambda (λ) value')
	plt.plot(range(len(models)), losses)

	plt.show()


def setup_triple_plot(title: str, x_true: np.ndarray, y_true: np.ndarray):
	dim_axs: np.ndarray[plt.Axes]
	_, dim_axs = plt.subplots(ncols=len(FEATURES))
	axes = dim_axs.flatten()
	plt.title(title)
	for idx, feature in enumerate(FEATURES):
		axes[idx].scatter(x_true[:, idx], y_true, label='True target prices')
		axes[idx].set_xlabel(feature)
		axes[idx].set_ylabel('Target price')
	plt.legend(loc='best')
	return axes


def plot(axes: np.ndarray[plt.Axes], model: MyRidge, x: np.ndarray) -> None:
	x_pol = MyRidge.add_polynomial_features(x, model.polynomial)
	x_norm = MyRidge.zscore(x_pol)
	y_hat = model.predict_(x_norm)

	for idx, feature in enumerate(FEATURES):
		axes[idx].scatter(x[:, idx], y_hat, alpha=0.3, label=f'Pol{model.polynomial}-λ{model.lambda_}')
	plt.legend(loc='best')


def plot_true_price(models: list[MyRidge], x: np.ndarray, x_norm, y: np.ndarray) -> None:
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
