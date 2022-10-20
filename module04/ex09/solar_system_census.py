import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from data_splitter import data_splitter
from data_utils import normalize_data, add_polynomials_and_normalize
from my_logistic_regression import MyLogisticRegression as MyLogR

MODELS_PICKLE_FILE = 'models.pickle'
CENSUS_CSV_PATH = '../resources/solar_system_census.csv'
PLANETS_CSV_PATH = '../resources/solar_system_census_planets.csv'
POLYNOMIAL_DEGREE = 3
FEATURES = ['height', 'weight', 'bone_density']


def prepare_data():
	data_x = pd.read_csv(CENSUS_CSV_PATH)
	data_y = pd.read_csv(PLANETS_CSV_PATH)
	x = data_x[FEATURES].to_numpy().reshape(-1, 3)
	y = data_y['Origin'].to_numpy().reshape(-1, 1)
	return data_splitter(x, y, 0.9)


def do_predictions_with_models(models: list[MyLogR], x_test: np.ndarray, y_test: np.ndarray):
	preds = [m.predict_(x_test) for m in models]
	final_preds = []
	correct_predictions = 0

	for i in range(x_test.shape[0]):
		sub_preds = [p[i] for p in preds]
		highest = sub_preds.index(max(sub_preds))
		final_preds.append(highest)
		if highest == int(y_test[i]):
			correct_predictions += 1
	print(f'Correctly predicted values {correct_predictions} / {y_test.shape[0]} = {correct_predictions / y_test.shape[0] * 100}%')
	return final_preds


def test_model(model: MyLogR, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, fit_again=False):
	x_train_norm, x_test_norm = add_polynomials_and_normalize(model, x_train, x_test)
	if fit_again:
		model.fit_(x_train_norm, y_train)

	y_hat = model.predict_(x_test_norm)
	test_loss = model.loss_(y_test, y_hat)
	return test_loss


def solar_system_census(models: list[MyLogR]):
	x_train, x_test, y_train, y_test = prepare_data()
	print(f'complete dataset: mean={x_train.mean(axis=0)}, std={x_train.std(axis=0)}')
	default_x_columns = x_train.shape[1]
	for idx, model in enumerate(models):
		polynomial_degree = (model.theta.shape[0] - 1) // default_x_columns
		model.set_params(polynomial=polynomial_degree)
		print(f'Testing model #{idx} again.')
		loss = test_model(model, x_train, x_test, y_train, y_test)
		print(f' Has loss of {loss:.1f}')
		print(f'thetas = {model.theta}')
		assert not np.isnan(loss)
		model.set_params(loss=loss)

	# plot_evaluation_curve(models)

	# Plot the evaluation curve which help you to select the best model (evaluation metrics vs models + λ factor).
	best_model = min(models, key=lambda x: x.loss)
	print(f'We found that the best model was with polynomial {best_model.polynomial} and lambda_ {best_model.lambda_}')
	new_loss = test_model(best_model, x_train, x_test, y_train, y_test, fit_again=False)
	best_model.set_params(loss=new_loss)

	# best_models = [m for m in models if m.polynomial == best_model.polynomial]
	# axes = setup_triple_plot('Best model', x_test, y_test)
	# for idx, model in enumerate(best_models):
	# 	plot(axes, model, x_test)
	# plt.show()
	#
	# # Let's plot all the different polynomials against each other
	# axes = setup_triple_plot('Showcase', x_test, y_test)
	# submodels = [m for m in models if np.isclose(m.lambda_, 0.0)]
	# for idx, model in enumerate(submodels):
	# 	plot(axes, model, x_test)
	# plt.show()


if __name__ == '__main__':
	if not os.path.exists(MODELS_PICKLE_FILE):
		print(f'Please run benchmark_train.py before running solar_system_census.py', file=sys.stderr)
		sys.exit(1)
	with open(MODELS_PICKLE_FILE, 'rb') as handle:
		all_models = pickle.load(handle)
	solar_system_census(all_models)
