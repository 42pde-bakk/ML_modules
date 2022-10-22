import copy
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

from cross_validation import build_cross_validation_sets
from data_splitter import data_splitter
from ridge import MyRidge
from plotting_like_the_lannisters import plot_evaluation_curve, plot_true_price
from data_utils import normalize_data, add_polynomials_and_normalize

MODELS_PICKLE_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'


def prepare_data() -> list[dict]:
	features = ['weight', 'prod_distance', 'time_delivery']
	data = pd.read_csv(CSV_FILE_PATH)
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[features].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	x, _, y, _ = data_splitter(x, y, 0.9)
	# Here we discard x_test and y_test because we will use them in space_avocado to test on
	return build_cross_validation_sets(x, y, 0.75)


def generate_random_thetas(n: int) -> np.ndarray:
	return np.ones(shape=(3 * n + 1, 1))
	# return np.random.rand(3 * n + 1, 1)


def benchmark_train(cross_validation_sets: list[dict]):
	models = []
	complete_x = np.vstack((cross_validation_sets[0]['x']['training'], cross_validation_sets[0]['x']['testing']))
	complete_y = np.vstack((cross_validation_sets[0]['y']['training'], cross_validation_sets[0]['y']['testing']))
	for i in range(1, 5):
		lambda_range = np.arange(0.0, 1.2, step=0.2)
		for lambda_ in lambda_range:
			print(f'Let\'s train a new model, polynomial={i}, λ {lambda_=:.1f}')
			model = MyRidge(thetas=generate_random_thetas(i), alpha=0.005, max_iter=100_000, lambda_=lambda_)
			model.set_params(polynomial=i)
			cross_validation_losses = []
			for idx, sets in enumerate(cross_validation_sets):
				model.set_params(thetas=generate_random_thetas(i))  # Reset thetas
				x_train, y_train = sets['x']['training'], sets['y']['training']
				x_test, y_test = sets['x']['testing'], sets['y']['testing']
				x_train, x_test = add_polynomials_and_normalize(model, x_train, x_test)

				print(f'Training Model {i} (x.shape={x_train.shape}) with λ {lambda_=:.1f}', end='')
				model.fit_(x_train, y_train)

				y_hat = model.predict_(x_test)
				loss = model.loss_(y_test, y_hat)
				print(f' has a loss of {loss} on cross-validation set {idx}.')
				cross_validation_losses.append(loss)
			average_loss = sum(cross_validation_losses) / len(cross_validation_losses)
			print(f'Model {i} with λ {lambda_=:.1f} had an average loss of {average_loss}')

			# And now we train our model again on the entire dataset

			new_x = MyRidge.add_polynomial_features(complete_x, i)
			new_x = MyRidge.zscore(new_x)
			model.set_params(thetas=generate_random_thetas(i))
			model.fit_(new_x, complete_y)
			y_hat = model.predict_(new_x)
			model.loss = model.loss_(complete_y, y_hat)
			print(f'Final model has a loss of {model.loss:.1f}.')
			models.append(copy.deepcopy(model))

	with open(MODELS_PICKLE_FILE, 'wb') as handle:
		pickle.dump(models, handle)
	plot_evaluation_curve(models)


if __name__ == '__main__':
	benchmark_train(prepare_data())
