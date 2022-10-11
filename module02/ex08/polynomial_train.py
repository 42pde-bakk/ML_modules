import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


def plot_mse_scores(losses: list):
	""" Plots a bar plot showing the MSE score of the models
		in function of the polynomial degree of the hypothesis"""
	plt.bar(range(1, 7), losses)
	plt.xlabel('Amount of polynomials')
	plt.ylabel('MSE score')
	plt.show()


def main():
	data = pd.read_csv('are_blue_pills_magic.csv')
	x = data['Micrograms'].to_numpy().reshape(-1, 1)
	y = data['Score'].to_numpy().reshape(-1, 1)

	losses = []
	models = [
		MyLR(np.ones(shape=(2, 1))),
		MyLR(np.ones(shape=(3, 1))),
		MyLR(np.ones(shape=(4, 1))),
		MyLR(np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)),
		MyLR(np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)),
		MyLR(np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]]).reshape(-1, 1))
	]

	for i in range(6):
		models[i].alpha /= (i + 1) * 100000
		x_ = add_polynomial_features(x, i + 1)
		print(f'model[{i}].fit_() gives {models[i].fit_(x_, y)}')
		y_hat = models[i].predict_(x_)
		loss = models[i].mse_(y, y_hat)
		print(f'model[{i}]\'s MSE score = {loss}')
		losses.append(loss)

	plot_mse_scores(losses)


if __name__ == '__main__':
	main()
