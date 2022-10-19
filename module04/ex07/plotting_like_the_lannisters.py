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
	# for polynomial in range(1, 5):
	xticks = [f'Pol {model.polynomial} & λ = {model.lambda_}' for model in models]
	losses = [model.loss for model in models]
	plt.title(f'Evaluation metrics of models')

	plt.xticks(range(len(models)), xticks, rotation=270)
	plt.ylabel('Loss score')
	plt.xlabel('Polynomials + Lambda (λ) value')
	plt.plot(range(len(models)), losses)

	plt.show()


def plot_true_price(models: list[MyRidge], x: np.ndarray, x_norm, y: np.ndarray):
	for i, feature in enumerate(FEATURES):
		plt.title(f'{feature} vs the true and predicted prices')
		x_col = x[:, i]
		plt.scatter(x_col, y, label='True prices')
		for model in models:
			y_hat = model.predict_(x_norm)
			print(f'feature {feature}, lambda_={model.lambda_}, y_hat[0] = {y_hat[0]}')
			np.savetxt(f'thetas_lambda{model.lambda_:.1f}.txt', model.thetas)
			np.savetxt(f'prediction_lambda{model.lambda_:.1f}.txt', y_hat)
			plt.scatter(x_col, y_hat, label=f'Predicted price with lambda_={model.lambda_:.1f}')
		plt.legend(loc='best')
		plt.show()
