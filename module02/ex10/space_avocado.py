import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features
from data_splitter import data_splitter


def plot_mse_scores(losses: list):
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.plot(range(1, 5), losses)
	plt.xlabel('Amount of polynomials')
	plt.ylabel('MSE score')
	plt.show()


def main():
	features = ['weight', 'prod_distance', 'time_delivery']
	data = pd.read_csv('space_avocado.csv')
	data.drop('Unnamed: 0', axis=1, inplace=True)
	x = data[features].to_numpy().reshape(-1, 3)
	y = data['target'].to_numpy().reshape(-1, 1)
	x_train, x_test, y_train, y_test = data_splitter(x, y, 0.8)

	models = [MyLR(np.ones(shape=(3 * i + 1, 1)), alpha=0.000001, max_iter=10_000) for i in range(1, 5)]
	losses = []
	for idx, model in enumerate(models):
		polynomial_degree = idx + 1
		x_ = add_polynomial_features(x_train, polynomial_degree)
		print(f'{x_.shape=}')
		print(f'{model.thetas.shape = }')
		model.fit_(x_, y_train)
		losses.append(model.mse_(y_train, model.predict_(x_)))
		print(f'Model {polynomial_degree} has a loss of {losses[-1]}')
	plot_mse_scores(losses)


if __name__ == '__main__':
	main()
