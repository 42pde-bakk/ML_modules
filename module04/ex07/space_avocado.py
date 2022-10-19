import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ridge import MyRidge
from data_splitter import data_splitter

MODELS_PICKLE_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'
FEATURES = ['weight', 'prod_distance', 'time_delivery']


def prepare_data():
	data = pd.read_csv(CSV_FILE_PATH)
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[FEATURES].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	return data_splitter(x, y, 0.9)


def plot_mse_scores(losses: list, title: str):
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


def plot_true_price(model: MyRidge, x: np.ndarray, y: np.ndarray):
	print(f'{y=}')
	for i, feature in enumerate(FEATURES):
		plt.title(f'{feature} vs the true and predicted prices')
		x_col = x[:, i]
		plt.plot(x_col, y, label='True prices')
		for lambda_ in np.arange(0.0, 1.2, step=0.2):
			model.set_params(lambda_=lambda_)
			y_hat = model.predict_(x)
			plt.plot(x_col, y_hat, label=f'Predicted price with lambda_={lambda_}')
		plt.legend(loc='best')
		plt.show()


def test_model(model: MyRidge, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, fit_again=False):
	x_train = MyRidge.add_polynomial_features(x_train, model.polynomial)
	x_test = MyRidge.add_polynomial_features(x_test, model.polynomial)
	mean, std = x_train.mean(), x_train.std()
	x_train = MyRidge.zscore_precomputed(x_train, mean, std)
	x_test = MyRidge.zscore_precomputed(x_test, mean, std)

	if fit_again:
		model.fit_(x_train, y_train)

	y_hat = model.predict_(x_test)
	test_loss = model.loss_(y_test, y_hat)
	return test_loss


def space_avocado(models: list[MyRidge]):
	x_train, x_test, y_train, y_test = prepare_data()
	default_x_columns = x_train.shape[1]
	for idx, model in enumerate(models):
		polynomial_degree = (model.thetas.shape[0] - 1) // default_x_columns
		setattr(model, 'polynomial', polynomial_degree)
		print(f'Testing model #{idx} again.', end='')
		loss = test_model(model, x_train, x_test, y_train, y_test)
		print(f' Has loss of {loss:.1f}')
		assert not np.isnan(loss)
		setattr(model, 'loss', loss)

	best_model = min(models, key=lambda x: x.loss)
	print(f'We found that the best model was with polynomial {best_model.polynomial} and lambda_ {best_model.lambda_}')
	print(best_model, vars(best_model))
	new_loss = test_model(best_model, x_train, x_test, y_train, y_test, fit_again=True)
	setattr(best_model, 'loss', new_loss)
	print(f'New loss is {new_loss}')

	# plot_evaluation_curve(models)
	plot_true_price(best_model, x_test, y_test)

	# plot_mse_scores(losses, 'Training losses')


if __name__ == '__main__':
	if not os.path.exists(MODELS_PICKLE_FILE):
		print(f'Please run benchmark_train.py before running space_avocado.py', file=sys.stderr)
		sys.exit(1)
	with open(MODELS_PICKLE_FILE, 'rb') as handle:
		all_models = pickle.load(handle)
	space_avocado(all_models)
