import copy
import pickle
import sys

import numpy as np
import pandas as pd

from cross_validation import build_cross_validation_sets
from my_logistic_regression import MyLogisticRegression as MyLogR
from other_metrics import f1_score_, accuracy_score_
from plotting_like_the_lannisters import plot_f1_scores

MODELS_PICKLE_FILE = 'models.pickle'
CENSUS_CSV_PATH = '../resources/solar_system_census.csv'
PLANETS_CSV_PATH = '../resources/solar_system_census_planets.csv'
POLYNOMIAL_DEGREE = 3
AMOUNT_UNIQUE_Y_VALUES = 0


def find_unique_y_values(y: np.ndarray) -> int:
	unique = np.unique(y).flatten()
	return len(set(unique))


def show_distribution(y: np.ndarray):
	y_arrs = np.split(y, 4)
	unique, counts = np.unique(y, return_counts=True)
	print('total:', dict(zip(unique, counts)))
	for idx, arr in enumerate(y_arrs):
		unique, counts = np.unique(arr, return_counts=True)
		print(f'\t{idx}:', dict(zip(unique, counts)))


def prepare_data():
	features = ['height', 'weight', 'bone_density']
	data_x = pd.read_csv(CENSUS_CSV_PATH)
	data_y = pd.read_csv(PLANETS_CSV_PATH)
	x = data_x[features].to_numpy().reshape(-1, 3)
	y = data_y['Origin'].to_numpy().reshape(-1, 1)
	x = MyLogR.add_polynomial_features(x, POLYNOMIAL_DEGREE)
	x = MyLogR.zscore(x)

	indices = [int(0.25 * x.shape[0]), int(0.35 * x.shape[0])]
	x_split = np.split(x, indices)
	y_split = np.split(y, indices)

	x, x_test = np.vstack((x_split[0], x_split[2])), x_split[1]
	y, y_test = np.vstack((y_split[0], y_split[2])), y_split[1]
	global AMOUNT_UNIQUE_Y_VALUES
	AMOUNT_UNIQUE_Y_VALUES = find_unique_y_values(y)
	# Here we discard x_test and y_test because we will use them in solar_system_census.py to test on
	return build_cross_validation_sets(x, y, 0.75), x, x_test, y, y_test


def generate_new_thetas(pol: int, ncols: int) -> np.ndarray:
	return np.ones(shape=(ncols * pol + 1, 1))
	# return np.random.rand(ncols * pol + 1, 1)


def combine_models(models: list[MyLogR], x_test: np.ndarray, y_test: np.ndarray) -> float:
	predict_together = np.hstack([m.predict_(x_test) for m in models])
	y_hat = predict_together.argmax(axis=1).reshape(-1, 1)
	f1 = f1_score_(y_test, y_hat)
	accuracy = accuracy_score_(y_test, y_hat)
	print(f'Correctly predicted {accuracy * 100:.1f}%, f1_score = {f1:.2f}')
	return f1


def train_models_for_all_zipcodes(x_train: np.ndarray, y_train: np.ndarray, lambda_: float) -> list:
	current_models = []
	for zipcode in range(AMOUNT_UNIQUE_Y_VALUES):
		thetas = generate_new_thetas(POLYNOMIAL_DEGREE, 3)
		model = MyLogR(theta=thetas, alpha=0.05, max_iter=15_000, lambda_=lambda_)
		model.set_params(polynomial=POLYNOMIAL_DEGREE, zipcode=zipcode)

		y_train_zipcode = np.where(y_train == zipcode, 1, 0)
		_ = model.fit_(x_train, y_train_zipcode)
		current_models.append(model)
	return current_models


def benchmark_train(cross_validation_sets: list[dict], x, x_test_full, y, y_test_full):
	lambda_range = np.arange(0.0, 1.2, step=0.2)
	models, lambda_f1s = [], []

	for lambda_ in lambda_range:
		print(f'Lambda = {lambda_:.1f}')
		cross_validation_f1scores = []
		models_backup = []
		for idx, sets in enumerate(cross_validation_sets):
			print(f'Cross validation attempt {idx}')
			x_train, y_train = sets['x']['training'].copy(), sets['y']['training'].copy()
			x_test, y_test = sets['x']['testing'].copy(), sets['y']['testing'].copy()

			# for zipcode in zipcodes: do stuff
			current_models = train_models_for_all_zipcodes(x_train, y_train, lambda_)

			print(f'All together now!')
			models_backup.append(copy.deepcopy(current_models))
			f1 = combine_models(current_models, x_test, y_test)
			cross_validation_f1scores.append(f1)
		# get the average score of the cross-validation models
		average_f1_score = sum(cross_validation_f1scores) / len(cross_validation_f1scores)
		print(f'Average score of cross validation with lambda_={lambda_:.1f} is {average_f1_score:.2f}')
		lambda_f1s.append(average_f1_score)

		# and redo the fitting of the model,
		# but this time with the entire dataset as training data
		current_models = train_models_for_all_zipcodes(x, y, lambda_)
		# combine_models(current_models, x_test_full, y_test_full)
		# sys.exit(1)
		models.append(copy.deepcopy(current_models))

	print(f'lets dump {len(models)} models')
	with open(MODELS_PICKLE_FILE, 'wb') as handle:
		pickle.dump(models, handle)
	# plot_f1_scores()
	plot_f1_scores(lambda_f1s, 'F1 scores', lambda_range)
	# plot_evaluation_curve(models)


if __name__ == '__main__':
	benchmark_train(*prepare_data())
