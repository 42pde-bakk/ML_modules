import pickle
from matplotlib import pyplot as plt
import numpy as np
from polynomial_model import add_polynomial_features
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
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
	models = [MyLR(np.ones(shape=(i, 1)), alpha=0.005, max_iter=100_000) for i in range(4, 8)]
	losses_train, losses_test = [], []
	for idx, model in enumerate(models):
		polynomial_degree = idx + 1
		x_ = add_polynomial_features(x, polynomial_degree)
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
