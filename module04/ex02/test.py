import numpy as np
from linear_loss_reg import reg_loss_


def main():
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

	# Example 1:
	result = reg_loss_(y, y_hat, theta, .5)
	print(f'Example 1: {result}\n')
	assert np.isclose(result, 0.8503571428571429)

	# Example 2:
	result = reg_loss_(y, y_hat, theta, .05)
	print(f'Example 2: {result}\n')
	assert np.isclose(result, 0.5511071428571429)

	# Example 3:
	result = reg_loss_(y, y_hat, theta, .9)
	print(f'Example 3: {result}\n')
	assert np.isclose(result, 1.116357142857143)


if __name__ == '__main__':
	main()
