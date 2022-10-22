import os
import pickle
import sys

import numpy as np
import pandas as pd

from benchmark_train import train_models_for_all_zipcodes
from data_utils import add_polynomials_and_normalize
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
	y = data_y['Origin'].to_numpy().reshape(-1, 1)

	_, x_test_orig = np.split(x, [int(0.9 * x.shape[0])])
	x = MyLogR.add_polynomial_features(x, POLYNOMIAL_DEGREE)
	x = MyLogR.zscore(x)

	indices = [int(0.25 * x.shape[0]), int(0.35 * x.shape[0])]
	x_split = np.split(x, indices)
	y_split = np.split(y, indices)

	x, x_test = np.vstack((x_split[0], x_split[2])), x_split[1]
	y, y_test = np.vstack((y_split[0], y_split[2])), y_split[1]
	return (x, x_test, y, y_test), x_test_orig


def generate_new_thetas(pol: int, ncols: int) -> np.ndarray:
	return np.ones(shape=(ncols * pol + 1, 1))


def combine_models(models: list[MyLogR], x_test: np.ndarray) -> np.ndarray:
	predict_together = np.hstack([m.predict_(x_test) for m in models])
	return predict_together.argmax(axis=1).reshape(-1, 1)


def test_model(model: MyLogR, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, fit_again=False):
	x_train_norm, x_test_norm = add_polynomials_and_normalize(model, x_train, x_test)
	if fit_again:
		model.fit_(x_train_norm, y_train)

	y_hat = model.predict_(x_test_norm)
	test_loss = model.loss_(y_test, y_hat)
	return test_loss


def solar_system_census(all_models: list[list[MyLogR]]) -> None:
	(x_train, x_test, y_train, y_test), x_test_orig = prepare_data()
	lambda_range = np.arange(0.0, 1.2, step=0.2)
	f1_scores = []

	for lambda_, models in zip(lambda_range, all_models):
		y_hat = combine_models(models, x_test)
		f1 = f1_score_(y_test, y_hat)
		accuracy = accuracy_score_(y_test, y_hat)
		print(f'Correctly predicted {accuracy * 100:.1f}%, f1_score = {f1:.1f}')
		f1_scores.append(f1)

	plot_f1_scores(f1_scores, 'F1 scores on the test set', lambda_range)

	best_idx = f1_scores.index(max(f1_scores))
	models = all_models[best_idx]
	# re-train.... lord knows why
	models = train_models_for_all_zipcodes(x_train, y_train, lambda_=models[0].lambda_)
	y_hat = combine_models(models, x_test)
	plot_true_price(x_test_orig, y_hat, y_test)


if __name__ == '__main__':
	if not os.path.exists(MODELS_PICKLE_FILE):
		print(f'Please run benchmark_train.py before running solar_system_census.py', file=sys.stderr)
		sys.exit(1)
	with open(MODELS_PICKLE_FILE, 'rb') as handle:
		pickled_models = pickle.load(handle)
	solar_system_census(pickled_models)
