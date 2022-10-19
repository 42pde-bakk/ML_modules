import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from data_splitter import data_splitter
from plotting_like_the_lannisters import plot_true_price, plot_evaluation_curve
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
	x_train = MyRidge.add_polynomial_features(x_train, model.polynomial)
	x_test = MyRidge.add_polynomial_features(x_test, model.polynomial)
	mean, std = x_train.mean(axis=0), x_train.std(axis=0)
	x_train = MyRidge.zscore_precomputed(x_train, mean, std)
	x_test = MyRidge.zscore_precomputed(x_test, mean, std)

	# if fit_again:
	# 	print('fitting again!')
	# 	model.fit_(x_train, y_train)

	y_hat = model.predict_(x_test)
	np.savetxt(f'predictions_{model.polynomial}_{model.lambda_:.1f}.np', y_hat)
	print(f' First pred = {y_hat[100]}. real value = {y_test[100]}')
	test_loss = model.loss_(y_test, y_hat)
	return test_loss


def normalize_data(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	mean, std = x_train.mean(), x_train.std()
	normalized_x_train = MyRidge.zscore_precomputed(x_train, mean, std)
	normalized_x_test = MyRidge.zscore_precomputed(x_test, mean, std)
	return normalized_x_train, normalized_x_test


def space_avocado(models: list[MyRidge]):
	x_train, x_test, y_train, y_test = prepare_data()
	x_train_norm, x_test_norm = normalize_data(x_train, x_test)
	print(f'complete dataset: mean={x_train.mean(axis=0)}, std={x_train.std(axis=0)}')
	default_x_columns = x_train.shape[1]
	for idx, model in enumerate(models):
		polynomial_degree = (model.thetas.shape[0] - 1) // default_x_columns
		model.set_params(polynomial=polynomial_degree)
		print(f'Testing model #{idx} again.')
		loss = test_model(model, x_train_norm, x_test_norm, y_train, y_test)
		print(f' Has loss of {loss:.1f}')
		print(f'thetas = {model.thetas}')
		assert not np.isnan(loss)
		model.set_params(loss=loss)

	best_model = min(models, key=lambda x: x.loss)
	print(f'We found that the best model was with polynomial {best_model.polynomial} and lambda_ {best_model.lambda_}')
	new_loss = test_model(best_model, x_train, x_test, y_train, y_test, fit_again=True)
	best_model.set_params(loss=new_loss)
	print(f' New loss is {new_loss}')

	# plot_evaluation_curve(models)
	best_models = [m for m in models if m.polynomial == best_model.polynomial]
	plot_true_price(best_models, x_test, x_test_norm, y_test)


if __name__ == '__main__':
	if not os.path.exists(MODELS_PICKLE_FILE):
		print(f'Please run benchmark_train.py before running space_avocado.py', file=sys.stderr)
		sys.exit(1)
	with open(MODELS_PICKLE_FILE, 'rb') as handle:
		all_models = pickle.load(handle)
	space_avocado(all_models)
