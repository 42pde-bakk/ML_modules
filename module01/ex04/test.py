import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error


def main() -> None:
	try:
		data = pd.read_csv('../resources/are_blue_pills_magic.csv')
	except (FileNotFoundError, pd.errors.EmptyDataError) as e:
		print('Error. Please supply a valid path to the csv file')
		exit(1)

	Xpill = np.array(data['Micrograms']).reshape(-1, 1)
	Yscore = np.array(data['Score']).reshape(-1, 1)
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))
	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)

	print(f'Model 1:')
	my_mse = MyLR.mse_(Yscore, Y_model1)
	sklearn_mse = mean_squared_error(Yscore, Y_model1)
	print(f'{my_mse = }')
	print(f'{sklearn_mse = }\n')
	assert np.isclose(my_mse, sklearn_mse)
	# 57.603042857142825

	print(f'Model 2:')
	my_mse = MyLR.mse_(Yscore, Y_model2)
	sklearn_mse = mean_squared_error(Yscore, Y_model2)
	print(f'{my_mse = }')
	print(f'{sklearn_mse = }\n')
	assert np.isclose(my_mse, sklearn_mse)
	# 232.16344285714285


if __name__ == '__main__':
	main()
