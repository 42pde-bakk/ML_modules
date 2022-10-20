import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLogR


def new_tests() -> None:
	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	# Example 1:
	model1 = MyLogR(theta, lambda_=5.0)
	assert model1.penalty == 'l2'
	assert model1.lambda_ == 5.0

	# Example 2:
	model2 = MyLogR(theta, penalty=None)
	assert model2.penalty is None
	assert model2.lambda_ == 0.0

	# Example 3:
	model3 = MyLogR(theta, penalty=None, lambda_=2.0)
	assert model3.penalty is None
	assert model3.lambda_ == 0.0


def module_03_ex_06_tests() -> None:
	X = np.array([
		[1., 1., 2., 3.],
		[5., 8., 13., 21.],
		[3., 5., 9., 14.]
	])
	Y = np.array([
		[1],
		[0],
		[1]
	], dtype=float)
	thetas = np.array([
		[2],
		[0.5],
		[7.1],
		[-4.3],
		[2.09]
	])
	mylr = MyLogR(thetas, penalty=None)

	# Example 0:
	predictions = mylr.predict_(X)
	answer = np.array([
		[0.99930437],
		[1.],
		[1.]
	])
	print(f'Example 0: {predictions = }\n')
	assert np.allclose(predictions, answer)

	# Example 1:
	loss = mylr.loss_(Y, y_hat=predictions)
	print(f'Example 1: {loss = }\n')
	assert np.isclose(loss, 11.513157421577004)

	# Example 2:
	mylr.fit_(X, Y)
	answer = np.array([
		[2.11826435],
		[0.10154334],
		[6.43942899],
		[-5.10817488],
		[0.6212541]
	])
	print(f'Example 2: {mylr.theta = }\n')
	assert np.allclose(mylr.theta, answer)

	# Example 3:
	predictions = mylr.predict_(X)
	answer = np.array([
		[0.57606717],
		[0.68599807],
		[0.06562156]
	])
	print(f'Example 3: {predictions = }\n')
	assert np.allclose(predictions, answer)

	# Example 4:
	loss = mylr.loss_(Y, predictions)
	print(f'Example 4: {loss = }\n')
	assert np.isclose(loss, 1.4779126923052268)

	# Extra tests
	y = np.array([[0], [0]])
	y_hat = np.array([[0], [0]])
	result = mylr.loss_(y, y_hat)
	print(f'Correction test 1: {result}')
	assert np.isclose(result, 1e-15)

	y = np.array([[0], [1]])
	y_hat = np.array([[0], [1]])
	result = mylr.loss_(y, y_hat)
	print(f'Correction test 2: {result}')
	assert np.isclose(result, 1e-15)

	y = np.array([[0], [0], [0]])
	y_hat = np.array([[1], [0], [0]])
	result = mylr.loss_(y, y_hat)
	print(f'Correction test 3: {result}')
	assert np.isclose(result, 11.51292546)

	y = np.array([[0], [0], [0]])
	y_hat = np.array([[1], [0], [1]])
	result = mylr.loss_(y, y_hat)
	print(f'Correction test 4: {result}')
	assert np.isclose(result, 23.02585093)

	y = np.array([[0], [1], [0]])
	y_hat = np.array([[1], [0], [1]])
	result = mylr.loss_(y, y_hat)
	print(f'Correction test 5: {result}\n\n')
	assert np.isclose(result, 34.53877639)


if __name__ == '__main__':
	module_03_ex_06_tests()
	new_tests()
