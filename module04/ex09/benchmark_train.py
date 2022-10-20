import copy
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from data_splitter import data_splitter
from cross_validation import build_cross_validation_sets
from data_utils import add_polynomials_and_normalize
from my_logistic_regression import MyLogisticRegression as MyLogR
from other_metrics import f1_score_
from plotting_like_the_lannisters import plot_f1_scores

MODELS_PICKLE_FILE = 'models.pickle'
CENSUS_CSV_PATH = '../resources/solar_system_census.csv'
PLANETS_CSV_PATH = '../resources/solar_system_census_planets.csv'
POLYNOMIAL_DEGREE = 3


def find_unique_y_values(y: np.ndarray) -> int:
	unique = np.unique(y).flatten()
	return len(set(unique))


def prepare_data() -> Tuple[list[dict], int]:
	features = ['height', 'weight', 'bone_density']
	data_x = pd.read_csv(CENSUS_CSV_PATH)
	data_y = pd.read_csv(PLANETS_CSV_PATH)
	x = data_x[features].to_numpy().reshape(-1, 3)
	y = data_y['Origin'].to_numpy().reshape(-1, 1)
	x, _, y, _ = data_splitter(x, y, 0.9)
	# Here we discard x_test and y_test because we will use them in solar_system_census.py to test on
	return build_cross_validation_sets(x, y, 0.75), find_unique_y_values(y)


def generate_new_thetas(pol: int, ncols: int) -> np.ndarray:
	return np.ones(shape=(ncols * pol + 1, 1))
	# return np.random.rand(ncols * pol + 1, 1)


def do_predictions_with_models(models: Tuple[MyLogR, MyLogR, MyLogR, MyLogR], x_test: np.ndarray, y_test: np.ndarray):
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


def benchmark_train(cross_validation_sets: list[dict], amount_unique_y_values: int):
	complete_x = np.vstack((cross_validation_sets[0]['x']['training'], cross_validation_sets[0]['x']['testing']))
	complete_x_norm = MyLogR.zscore(MyLogR.add_polynomial_features(complete_x, POLYNOMIAL_DEGREE))
	complete_y = np.vstack((cross_validation_sets[0]['y']['training'], cross_validation_sets[0]['y']['testing']))
	lambda_range = np.arange(0.0, 1.2, step=0.2)
	models, f1_scores = [], []

	for lambda_ in lambda_range:
		current_models = []
		for zipcode in range(amount_unique_y_values):
			print(f'Let\'s train a new model, polynomial={zipcode}, λ {lambda_=:.1f}')
			model = MyLogR(theta=generate_new_thetas(pol=POLYNOMIAL_DEGREE, ncols=3), alpha=0.005, max_iter=1_000, lambda_=lambda_)
			model.set_params(polynomial=POLYNOMIAL_DEGREE, zipcode=zipcode)
			cross_validation_f1scores = []
			for idx, sets in enumerate(cross_validation_sets):
				model.set_params(theta=generate_new_thetas(pol=POLYNOMIAL_DEGREE, ncols=3))  # Reset thetas
				x_train, y_train = sets['x']['training'].copy(), sets['y']['training'].copy()
				x_test, y_test = sets['x']['testing'].copy(), sets['y']['testing'].copy()
				x_train, x_test = add_polynomials_and_normalize(model, x_train, x_test)
				y_train = (y_train == zipcode).astype(int)
				y_test = (y_test == zipcode).astype(int)
				print(f'Training Model {zipcode} (x.shape={x_train.shape}) with λ {lambda_=:.1f}', end='')
				model.fit_(x_train, y_train)

				y_hat = model.predict_(x_test)
				rounded = np.round(y_hat)
				f1 = f1_score_(y_test, rounded)
				print(f' has an f1_score of {f1} on cross-validation set {idx}.')
				cross_validation_f1scores.append(f1)
			average_f1score = sum(cross_validation_f1scores) / len(cross_validation_f1scores)
			print(f'Model {zipcode} with λ {lambda_=:.1f} had an average f1_score of {average_f1score}')

			# And now we train our model again on the entire dataset

			model.set_params(theta=generate_new_thetas(pol=POLYNOMIAL_DEGREE, ncols=3))
			complete_y_for_zipcode = (complete_y == zipcode).astype(int)
			model.fit_(complete_x_norm, complete_y_for_zipcode)

			y_hat = model.predict_(complete_x_norm)
			rounded = np.round(y_hat)
			model.f1 = f1_score_(complete_y_for_zipcode, rounded)
			current_models.append(copy.deepcopy(model))
			models.append(copy.deepcopy(model))
		predict_together = np.array([np.array(m.predict_(complete_x_norm)).T for m in current_models]).reshape(-1, 4)
		argmax = predict_together.argmax(axis=1).reshape(-1, 1)
		f1 = f1_score_(complete_y, argmax)
		f1_scores.append(f1)
		print(f'added {f1_scores}')

	print(f'lets dump {len(models)} models')
	with open(MODELS_PICKLE_FILE, 'wb') as handle:
		pickle.dump(models, handle)
	# plot_f1_scores()
	plot_f1_scores(f1_scores, 'F1 scores', lambda_range)
	# plot_evaluation_curve(models)


if __name__ == '__main__':
	benchmark_train(*prepare_data())
