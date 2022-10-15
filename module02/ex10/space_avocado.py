import numpy as np
import os
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features
from benchmark_train import benchmark_train, plot_mse_scores, prepare_data
import pickle
from matplotlib import pyplot as plt


THETAS_FILE = 'models.pickle'
CSV_FILE_PATH = '../resources/space_avocado.csv'


def benchmark_test(x: np.ndarray, y: np.ndarray):
	features = ['weight', 'prod_distance', 'time_delivery']
	with open(THETAS_FILE, 'rb') as handle:
		all_thetas = pickle.load(handle)
	models = [MyLR(thetas, alpha=0.00001, max_iter=1_000_000) for thetas in all_thetas]
	losses = []
	preds = []
	for idx, model in enumerate(models):
		polynomial_degree = idx + 1
		x_test_ = add_polynomial_features(x, polynomial_degree)
		x_test_ = MyLR.zscore(x_test_)
		y_hat = model.predict_(x_test_)
		preds.append(y_hat)
		losses.append(model.mse_(y, y_hat))
		print(f'Polynomial {polynomial_degree}, testing loss: {losses[-1]}')

	plot_mse_scores(losses, 'Loss (in MSE) on the testing dataset')

	best_model_idx = losses.index(min(losses))
	y_hat = preds[best_model_idx]

	for idx, feature in enumerate(features):
		plt.title(feature)

		x_axis = x[:, idx].reshape(-1, 1)
		plt.scatter(x_axis, y, label='Actual target value')
		plt.scatter(x_axis, y_hat, label='My predictions')

		plt.legend(loc='best')
		plt.show()


def main() -> None:
	x_train, x_test, y_train, y_test = prepare_data()
	if not os.path.exists(THETAS_FILE):
		print(f'apparently thetas file does not exist')
		benchmark_train(x_train, y_train)
	benchmark_test(x_test, y_test)


if __name__ == '__main__':
	main()
