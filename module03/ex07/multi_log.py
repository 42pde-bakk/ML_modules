import numpy as np
import pandas as pd
from data_splitter import data_splitter
from my_logistic_regression import MyLogisticRegression as MyLR
from matplotlib import pyplot as plt


INFORMATION_CSV_FILEPATH = '../resources/solar_system_census.csv'
PLANETS_CSV_FILEPATH = '../resources/solar_system_census_planets.csv'


def prepare_data():
	data_x = pd.read_csv(INFORMATION_CSV_FILEPATH)
	data_y = pd.read_csv(PLANETS_CSV_FILEPATH)
	data_x.drop('Unnamed: 0', axis=1, inplace=True), data_y.drop('Unnamed: 0', axis=1, inplace=True)
	entire_x = data_x.to_numpy().reshape(-1, len(data_x.columns))
	entire_y = data_y.to_numpy().reshape(-1, len(data_y.columns))
	return data_splitter(entire_x, entire_y, 0.8), len(data_x.columns)


def create_mono_log_model(zipcode: int, x_train, y_train, x_test, y_test, column_len) -> MyLR:
	y2_train = (y_train == zipcode).astype(int)
	y2_test = (y_test == zipcode).astype(int)

	initial_thetas = np.ones(shape=(column_len + 1, 1))
	model = MyLR(initial_thetas, alpha=0.0001, max_iter=300_000)
	model.fit_(x_train, y2_train)
	return model


def main():
	features = ['weight', 'height', 'bone_density']
	(x_train, x_test, y_train, y_test), column_len = prepare_data()

	models, preds = [], []
	for zipcode in range(4):
		model = create_mono_log_model(zipcode, x_train, y_train, x_test, y_test, column_len)
		models.append(model)
		preds.append(model.predict_(x_test))

	correct_predictions = 0
	final_preds = []
	for i in range(y_test.shape[0]):
		subpreds = [p[i] for p in preds]
		highest = subpreds.index(max(subpreds))
		final_preds.append(highest)
		if highest == int(y_test[i]):
			correct_predictions += 1

	print(f'Correctly predicted values {correct_predictions} / {y_test.shape[0]} = {correct_predictions / y_test.shape[0] * 100}%')

	for idx, feature in enumerate(features):
		plt.title(feature)
		column = x_test[:, idx]
		plt.scatter(column, y_test, label='Real values', s=200)
		plt.scatter(column, final_preds, label='My prediction', s=50)

		plt.legend(loc='best')
		plt.show()


if __name__ == '__main__':
	main()
