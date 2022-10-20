import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from data_splitter import data_splitter
from plotting_like_the_lannisters import *
from data_utils import normalize_data, add_polynomials_and_normalize
from ridge import MyRidge

MODELS_PICKLE_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'
FEATURES = ['weight', 'prod_distance', 'time_delivery']


def prepare_data():
	data = pd.read_csv(CSV_FILE_PATH)
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[FEATURES].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	return data_splitter(x, y, 0.9)


def test_model(model: MyRidge, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, fit_again=False):
	x_train_norm, x_test_norm = add_polynomials_and_normalize(model, x_train, x_test)
	if fit_again:
		model.fit_(x_train_norm, y_train)

	y_hat = model.predict_(x_test_norm)
	test_loss = model.loss_(y_test, y_hat)
	return test_loss


def space_avocado(models: list[MyRidge]):
	x_train, x_test, y_train, y_test = prepare_data()
	print(f'complete dataset: mean={x_train.mean(axis=0)}, std={x_train.std(axis=0)}')
	default_x_columns = x_train.shape[1]
	for idx, model in enumerate(models):
		polynomial_degree = (model.thetas.shape[0] - 1) // default_x_columns
		model.set_params(polynomial=polynomial_degree)
		print(f'Testing model #{idx} again.')
		loss = test_model(model, x_train, x_test, y_train, y_test)
		print(f' Has loss of {loss:.1f}')
		print(f'thetas = {model.thetas}')
		assert not np.isnan(loss)
		model.set_params(loss=loss)

	plot_evaluation_curve(models)

	# Plot the evaluation curve which help you to select the best model (evaluation metrics vs models + λ factor).
	best_model = min(models, key=lambda x: x.loss)
	print(f'We found that the best model was with polynomial {best_model.polynomial} and lambda_ {best_model.lambda_}')
	new_loss = test_model(best_model, x_train, x_test, y_train, y_test, fit_again=False)
	best_model.set_params(loss=new_loss)

	# Plot for all lambda_ values of our best model
	# Plot the true price and the predicted price obtained via your best model with the
	# different λ values (meaning the dataset + the 5 predicted curves).
	best_models = [m for m in models if m.polynomial == best_model.polynomial]
	axes = setup_triple_plot('Best model', x_test, y_test)
	for idx, model in enumerate(best_models):
		plot(axes, model, x_test)
	plt.show()

	# Let's plot all the different polynomials against each other
	axes = setup_triple_plot('Showcase', x_test, y_test)
	submodels = [m for m in models if np.isclose(m.lambda_, 0.0)]
	for idx, model in enumerate(submodels):
		plot(axes, model, x_test)
	plt.show()


if __name__ == '__main__':
	if not os.path.exists(MODELS_PICKLE_FILE):
		print(f'Please run benchmark_train.py before running space_avocado.py', file=sys.stderr)
		sys.exit(1)
	with open(MODELS_PICKLE_FILE, 'rb') as handle:
		all_models = pickle.load(handle)
	space_avocado(all_models)
