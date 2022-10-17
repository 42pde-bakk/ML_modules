import numpy as np
from log_pred import logistic_predict_
from vec_log_loss import vec_log_loss_


def main():
	# Example 1:
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	y_hat1 = logistic_predict_(x1, theta1)
	result = vec_log_loss_(y1, y_hat1)
	print(f'Example 1: {result}\n')
	assert np.allclose(result, 0.01814992791780973)

	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	y_hat2 = logistic_predict_(x2, theta2)
	# result = log_loss_(y2, y_hat2)
	result = vec_log_loss_(y2, y_hat2)
	print(f'Example 2: {result}\n')
	assert np.allclose(result, 2.4825011602474483)

	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	y_hat3 = logistic_predict_(x3, theta3)
	result = vec_log_loss_(y3, y_hat3)
	print(f'Example 3: {result}\n')
	assert np.allclose(result, 2.9938533108607053)


def correction_tests():
	y = np.array([[0.0], [0.0]])
	y_hat = np.array([[0.0], [0.0]])
	result = vec_log_loss_(y, y_hat)
	print(f'Correction test 1: {result}')
	assert np.isclose(result, 1e-15)

	y = np.array([[0], [1]])
	y_hat = np.array([[0], [1]])
	result = vec_log_loss_(y, y_hat)
	print(f'Correction test 2: {result}')
	assert np.isclose(result, 1e-15)

	y = np.array([[0], [0], [0]])
	y_hat = np.array([[1], [0], [0]])
	result = vec_log_loss_(y, y_hat)
	print(f'Correction test 3: {result}')
	assert np.isclose(result, 11.51292546)

	y = np.array([[0], [0], [0]])
	y_hat = np.array([[1], [0], [1]])
	result = vec_log_loss_(y, y_hat)
	print(f'Correction test 4: {result}')
	assert np.isclose(result, 23.02585093)

	y = np.array([[0], [1], [0]])
	y_hat = np.array([[1], [0], [1]])
	result = vec_log_loss_(y, y_hat)
	print(f'Correction test 5: {result}')
	assert np.isclose(result, 34.53877639)


if __name__ == '__main__':
	main()
	correction_tests()
