import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ridge import MyRidge
from cross_validation import build_cross_validation_sets


MODELS_PICKLE_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'


def prepare_data() -> list[dict]:
	features = ['weight', 'prod_distance', 'time_delivery']
	data = pd.read_csv(CSV_FILE_PATH)
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[features].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	return build_cross_validation_sets(x, y, 0.75)


def benchmark_train(cross_validation_sets: list[dict]):
	models, losses = [], []
	for i in range(1, 5):
		print(f'i = {i}\n')
		thetas = np.ones(shape=(3 * i + 1, 1))
		print(f'created {thetas.shape=}')
		lambda_range = np.arange(0.0, 1.2, step=0.2)
		for lambda_ in lambda_range:
			print(f'Let\'s train a new model, polynomial={i}, λ {lambda_=}')
			model = MyRidge(thetas, alpha=0.001, max_iter=10_000, lambda_=lambda_)
			cross_validation_losses = []
			for idx, sets in enumerate(cross_validation_sets):
				model.set_params(thetas=thetas)  # Reset thetas
				x_train, y_train = sets['x']['training'], sets['y']['training']
				x_train = MyRidge.add_polynomial_features(x_train, i)
				x_train = MyRidge.zscore(x_train)

				print(f'Training Model {i} with λ {lambda_=}', end='')
				model.fit_(x_train, y_train)

				x_test, y_test = sets['x']['testing'], sets['y']['testing']
				x_test = MyRidge.zscore(MyRidge.add_polynomial_features(x_test, i))

				y_hat = model.predict_(x_test)
				loss = model.loss_(y_test, y_hat)
				print(f' has a loss of {loss} on cross-validation set {idx}.')
				cross_validation_losses.append(loss)
			average_loss = sum(cross_validation_losses) / len(cross_validation_losses)
			print(f'Model {i} with λ {lambda_=} had an average loss of {average_loss}')
			losses.append(average_loss)

	with open(MODELS_PICKLE_FILE, 'wb') as handle:
		pickle.dump(models, handle)


if __name__ == '__main__':
	benchmark_train(prepare_data())
