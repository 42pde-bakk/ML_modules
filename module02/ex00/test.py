import numpy as np
from prediction import simple_predict


def main() -> None:
	X = np.arange(1, 13).reshape((4, -1))
	print(f'{X=}')
	# Example 1:
	print('Example 1:')
	theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
	y_hat = simple_predict(X, theta1)
	answer = np.array([[5.], [5.], [5.], [5.]])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, answer)
	# Do you understand why y_hat contains only 5’s here?

	# Example 2:
	print('\n\nExample 2:')
	theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
	y_hat = simple_predict(X, theta2)
	answer = np.array([[1.], [4.], [7.], [10.]])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, answer)
	# Do you understand why y_hat == x[:,0] here?

	# Example 3:
	print('\n\nExample 3:')
	theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
	y_hat = simple_predict(X, theta3)
	answer = np.array([[9.64], [24.28], [38.92], [53.56]])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, answer)

	# Example 4:
	theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
	y_hat = simple_predict(X, theta4)
	answer = np.array([[12.5], [32.], [51.5], [71.]])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, answer)


if __name__ == '__main__':
	main()
