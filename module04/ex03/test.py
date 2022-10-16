import numpy as np
from logistic_loss_reg import reg_log_loss_


def main():
	y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
	y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

	# Example 1:
	result1 = reg_log_loss_(y, y_hat, theta, .5)
	print(f'Example 1: {result1}\n')
	assert np.isclose(result1, 0.43377043716475955)

	# Example :
	result2 = reg_log_loss_(y, y_hat, theta, .05)
	print(f'Example 2: {result2}\n')
	assert np.isclose(result2, 0.13452043716475953)

	# Example :
	result3 = reg_log_loss_(y, y_hat, theta, .9)
	print(f'Example 3: {result3}\n')
	assert np.isclose(result3, 0.6997704371647596)



if __name__ == '__main__':
	main()
