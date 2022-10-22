import os
import pickle
import sys

import numpy as np
import pandas as pd

from data_splitter import data_splitter
from data_utils import normalize_data, add_polynomials_and_normalize
from my_logistic_regression import MyLogisticRegression as MyLogR
from other_metrics import f1_score_, accuracy_score_
from plotting_like_the_lannisters import plot_f1_scores, plot_true_price

MODELS_PICKLE_FILE = 'models.pickle'
CENSUS_CSV_PATH = '../resources/solar_system_census.csv'
PLANETS_CSV_PATH = '../resources/solar_system_census_planets.csv'
POLYNOMIAL_DEGREE = 3
FEATURES = ['height', 'weight', 'bone_density']


def prepare_data():
	data_x = pd.read_csv(CENSUS_CSV_PATH)
	data_y = pd.read_csv(PLANETS_CSV_PATH)
	x = data_x[FEATURES].to_numpy().reshape(-1, 3)
	x = MyLogR.zscore(MyLogR.add_polynomial_features(x, POLYNOMIAL_DEGREE))
	y = data_y['Origin'].to_numpy().reshape(-1, 1)
	return data_splitter(x, y, 0.9)


def combine_models(models: list[MyLogR], x_test: np.ndarray, y_test: np.ndarray) -> float:
	predict_together = np.hstack([m.predict_(x_test) for m in models])
	y_hat = predict_together.argmax(axis=1).reshape(-1, 1)
	f1 = f1_score_(y_test, y_hat)
	accuracy = accuracy_score_(y_test, y_hat)
	print(f'Correctly predicted {accuracy * 100:.1f}%, f1_score = {f1:.1f}')
	return f1


def test_model(model: MyLogR, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, fit_again=False):
	x_train_norm, x_test_norm = add_polynomials_and_normalize(model, x_train, x_test)
	if fit_again:
		model.fit_(x_train_norm, y_train)

	y_hat = model.predict_(x_test_norm)
	test_loss = model.loss_(y_test, y_hat)
	return test_loss


def solar_system_census(all_models: list[list[MyLogR]]):
	x_train, x_test, y_train, y_test = prepare_data()
	lambda_range = np.arange(0.0, 1.2, step=0.2)
	f1_scores = []

	for lambda_, models in zip(lambda_range, all_models):
		f1_score = combine_models(models, x_test, y_test)
		f1_scores.append(f1_score)

	plot_f1_scores(f1_scores, 'F1 scores on the test set', lambda_range)


if __name__ == '__main__':
	if not os.path.exists(MODELS_PICKLE_FILE):
		print(f'Please run benchmark_train.py before running solar_system_census.py', file=sys.stderr)
		sys.exit(1)
	with open(MODELS_PICKLE_FILE, 'rb') as handle:
		pickled_models = pickle.load(handle)
	solar_system_census(pickled_models)
