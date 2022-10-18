import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ridge import MyRidge
from data_splitter import data_splitter
from typing import Tuple


THETAS_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	features = ['weight', 'prod_distance', 'time_delivery']
	data = pd.read_csv(CSV_FILE_PATH)
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[features].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	return data_splitter(x, y, 0.8)


def plot_mse_scores(losses: list, title: str):
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.title(title)
	plt.plot(range(1, len(losses) + 1), losses)
	plt.xlabel('Amount of polynomials')
	plt.ylabel('MSE score')
	plt.show()


def benchmark_train(x: np.ndarray, y: np.ndarray):
	models = []
	for polynomial in range(1, 4):
		new_x = MyRidge.add_polynomial_features(x, polynomial)
		models.append(MyRidge(np.ones(new_x.shape[0], 1)))
	losses_train, losses_test = [], []
	for i in range(1, 5):
		lambda_ = 0.1 + i * 0.2
		new_x = MyRidge.add_polynomial_features(x, i)
		models.append(MyRidge(np.ones(new_x.shape[0], 1)), lambda_=lambda_)
		# polynomial_degree = idx + 1
		# x_ = add_polynomial_features(x, polynomial_degree)
		x_ = MyLR.zscore(x_)
		model.fit_(x_, y)
		losses_train.append(model.mse_(y, model.predict_(x_)))
		print(f'Polynomial {polynomial_degree} training loss: {losses_train[-1]}')

	thetas = [lr.thetas for lr in models]
	with open(THETAS_FILE, 'wb') as handle:
		pickle.dump(thetas, handle)
	plot_mse_scores(losses_train, 'Training losses')


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = prepare_data()
	benchmark_train(x_train, y_train)
