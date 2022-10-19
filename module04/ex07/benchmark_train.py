import copy
import pickle
import sys

import numpy as np
import pandas as pd
from ridge import MyRidge
from data_splitter import data_splitter
from cross_validation import build_cross_validation_sets


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


def benchmark_train(cross_validation_sets: list[dict]):
	models = []
	for i in range(1, 5):
		default_thetas = np.ones(shape=(3 * i + 1, 1))
		lambda_range = np.arange(0.0, 1.2, step=0.2)
		for lambda_ in lambda_range:
			print(f'Let\'s train a new model, polynomial={i}, λ {lambda_=}')
			model = MyRidge(default_thetas, alpha=0.001, max_iter=100_000, lambda_=lambda_)
			cross_validation_losses = []
			for idx, sets in enumerate(cross_validation_sets):
				model.set_params(thetas=default_thetas)  # Reset thetas
				x_train, y_train = sets['x']['training'], sets['y']['training']
				x_train = MyRidge.add_polynomial_features(x_train, i)
				mean, std = x_train.mean(), x_train.std()
				x_train = MyRidge.zscore_precomputed(x_train, mean, std)

				print(f'Training Model {i} (x.shape={x_train.shape}) with λ {lambda_=:.1f}', end='')
				model.fit_(x_train, y_train)

				x_test, y_test = sets['x']['testing'], sets['y']['testing']
				x_test = MyRidge.zscore_precomputed(MyRidge.add_polynomial_features(x_test, i), mean, std)

				y_hat = model.predict_(x_test)
				loss = model.loss_(y_test, y_hat)
				print(f' has a loss of {loss} on cross-validation set {idx}.')
				cross_validation_losses.append(loss)
			average_loss = sum(cross_validation_losses) / len(cross_validation_losses)
			print(f'Model {i} with λ {lambda_=} had an average loss of {average_loss}')

			# And now we train our model again on the entire dataset
			x = np.vstack((cross_validation_sets[0]['x']['training'], cross_validation_sets[0]['x']['testing']))
			y = np.vstack((cross_validation_sets[0]['y']['training'], cross_validation_sets[0]['y']['testing']))
			x = MyRidge.zscore(MyRidge.add_polynomial_features(x, i))
			model.set_params(thetas=default_thetas)
			model.fit_(x, y)
			loss = model.loss_(y, model.predict_(x))
			print(f'Final model has a loss of {loss:.1f}.')
			models.append(model)

	print(f'lets dump {len(models)} models')
	with open(MODELS_PICKLE_FILE, 'wb') as handle:
		pickle.dump(models, handle)


if __name__ == '__main__':
	benchmark_train(prepare_data())
