import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ridge import MyRidge
from data_splitter import data_splitter
from typing import Tuple


THETAS_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	features = ['weight', 'prod_distance', 'time_delivery']
	data = pd.read_csv(CSV_FILE_PATH)
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[features].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	return data_splitter(x, y, proportion=0.7, crossval_proprtion=0.15)


def plot_mse_scores(losses: list, title: str):
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.title(title)
	print(len(losses))
	print(losses)
	plt.plot(range(1, len(losses) + 1), losses)
	plt.xlabel('Model nb')
	plt.ylabel('Loss score')
	plt.show()


def benchmark_train(x: np.ndarray, y: np.ndarray):
	models, losses = [], []
	for i in range(1, 5):
		print(f'i = {i}\n')
		new_x = MyRidge.add_polynomial_features(x, i)
		new_x = MyRidge.zscore(new_x)
		thetas = np.ones(shape=(new_x.shape[1] + 1, 1))
		print(f'created new_x with shape={new_x.shape}, {thetas.shape=}')
		lambda_range = np.arange(0.0, 1.2, step=0.2)
		for lambda_ in lambda_range:
			print(f'Let\'s train a new model, polynomial={i}, λ {lambda_=}')
			model = MyRidge(thetas, alpha=0.001, max_iter=1_000_000, lambda_=lambda_)
			model.fit_(new_x, y)
			loss = model.loss_(y, model.predict_(new_x))
			models.append(model)
			losses.append(loss)
			print(f'Model {i} training loss: {loss}')
		# break

	thetas = [lr.thetas for lr in models]
	with open(THETAS_FILE, 'wb') as handle:
		pickle.dump(thetas, handle)
	# plot_mse_scores(losses, 'Training losses')


if __name__ == '__main__':
	x_train, x_crossval, x_test, y_train, y_crossval, y_test = prepare_data()
	print(x_train.shape, x_crossval.shape, x_test.shape)
	print(y_train.shape, y_crossval.shape, y_test.shape)
	benchmark_train(x_train, y_train)
