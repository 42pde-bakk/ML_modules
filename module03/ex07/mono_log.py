import numpy as np
import pandas as pd
import argparse
import sys
from data_splitter import data_splitter
from my_logistic_regression import MyLogisticRegression as MyLR
from matplotlib import pyplot as plt


INFORMATION_CSV_FILEPATH = '../resources/solar_system_census.csv'
PLANETS_CSV_FILEPATH = '../resources/solar_system_census_planets.csv'


def get_zipcode() -> int:
	parser = argparse.ArgumentParser(description='Give me a zipcode')
	parser.add_argument('-zipcode', type=int)

	args = parser.parse_args()
	if args.zipcode is None or not 0 <= args.zipcode <= 3:
		parser.print_usage()
		exit(1)
	return args.zipcode


def prepare_data():
	try:
		data_x = pd.read_csv(INFORMATION_CSV_FILEPATH)
		data_y = pd.read_csv(PLANETS_CSV_FILEPATH)
	except (FileNotFoundError, pd.errors.EmptyDataError) as e:
		print(f'Error: {e}')
		sys.exit(1)
	data_x.drop('Unnamed: 0', axis=1, inplace=True), data_y.drop('Unnamed: 0', axis=1, inplace=True)
	entire_x = data_x.to_numpy().reshape(-1, len(data_x.columns))
	entire_y = data_y.to_numpy().reshape(-1, len(data_y.columns))
	return data_splitter(entire_x, entire_y, 0.8), len(data_x.columns)


def main():
	zipcode_amount = 4
	zipcode_names = ['Venus (0)', 'Earth (1)', 'Mars (2)', 'The Belt (3)']
	features = ['weight', 'height', 'bone_density']
	favourite_zipcode: int = get_zipcode()
	ytick_labels = ['Not ' + zipcode_names[favourite_zipcode], zipcode_names[favourite_zipcode]]
	(x_train, x_test, y_train, y_test), column_len = prepare_data()

	y2_train = (y_train == favourite_zipcode).astype(int)
	y2_test = (y_test == favourite_zipcode).astype(int)

	initial_thetas = np.ones(shape=(column_len + 1, 1))
	model = MyLR(initial_thetas, alpha=0.0001, max_iter=500_000)
	model.fit_(x_train, y2_train)
	predictions = model.predict_(x_test)
	correct = np.sum(np.round(predictions) == y2_test)
	print(f'Correctly predicted values {correct} / {y2_test.shape[0]} = {correct / y2_test.shape[0] * 100}%')

	for idx, feature in enumerate(features):
		plt.title(f'Zipcode predicted based on {feature}')
		plt.xlabel(feature.capitalize())
		plt.ylabel('Zipcode')
		plt.yticks(range(0, 2), ytick_labels)
		column = x_test[:, idx]
		plt.scatter(column, y2_test, label='Real values', s=200)
		plt.scatter(column, predictions, label='My prediction', s=50)
		plt.scatter(column, np.round(predictions), label='My rounded prediction')

		plt.legend(loc='best')
		plt.show()


if __name__ == '__main__':
	main()
