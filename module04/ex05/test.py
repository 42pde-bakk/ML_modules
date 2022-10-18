import numpy as np
from reg_logistic_grad import reg_logistic_grad, vec_reg_logistic_grad


def main():
	x = np.array([
		[0, 2, 3, 4],
		[2, 4, 5, 5],
		[1, 3, 2, 7]
	])
	y = np.array([
		[0],
		[1],
		[1]
	])
	theta = np.array([
		[-2.4],
		[-1.5],
		[0.3],
		[-1.4],
		[0.7]
	])

	# Example 1.1:
	result = reg_logistic_grad(y, x, theta, 1)
	# Output:
	answer = np.array([
		[-0.55711039],
		[-1.40334809],
		[-1.91756886],
		[-2.56737958],
		[-3.03924017]
	])
	print(f'Example 1.1: \n{result}\n')
	assert np.allclose(result, answer)

	# Example 1.2:
	result = vec_reg_logistic_grad(y, x, theta, 1)
	# Output:
	answer = np.array([
		[-0.55711039],
		[-1.40334809],
		[-1.91756886],
		[-2.56737958],
		[-3.03924017]
	])
	print(f'Example 1.2: \n{result}\n')
	assert np.allclose(result, answer)

	# Example 2.1:
	result = reg_logistic_grad(y, x, theta, 0.5)
	# Output:
	answer = np.array([
		[-0.55711039],
		[-1.15334809],
		[-1.96756886],
		[-2.33404624],
		[-3.15590684]
	])
	print(f'Example 2.1: \n{result}\n')
	assert np.allclose(result, answer)

	# Example 2.2:
	result = vec_reg_logistic_grad(y, x, theta, 0.5)
	# Output:
	answer = np.array([
		[-0.55711039],
		[-1.15334809],
		[-1.96756886],
		[-2.33404624],
		[-3.15590684]
	])
	print(f'Example 2.2: \n{result}\n')
	assert np.allclose(result, answer)

	# Example 3.1:
	result = reg_logistic_grad(y, x, theta, 0.0)
	# Output:
	answer = np.array([
		[-0.55711039],
		[-0.90334809],
		[-2.01756886],
		[-2.10071291],
		[-3.27257351]
	])
	print(f'Example 3.1: \n{result}\n')
	assert np.allclose(result, answer)

	# Example 3.2:
	result = vec_reg_logistic_grad(y, x, theta, 0.0)
	# Output:
	answer = np.array([
		[-0.55711039],
		[-0.90334809],
		[-2.01756886],
		[-2.10071291],
		[-3.27257351]
	])
	print(f'Example 3.2: \n{result}\n')
	assert np.allclose(result, answer)


if __name__ == '__main__':
	main()
