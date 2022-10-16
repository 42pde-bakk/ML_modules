import numpy as np
from reg_linear_grad import reg_linear_grad, vec_reg_linear_grad


def main():
	x = np.array([
		[-6, -7, -9],
		[13, -2, 14],
		[-7, 14, -1],
		[-8, -4, 6],
		[-5, -9, 6],
		[1, -5, 11],
		[9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])

	# Example 1.1:
	result_11 = reg_linear_grad(y, x, theta, 1)
	print(f'Example 1.1: {result_11}\n')
	answer_11 = np.array([
		[-60.99],
		[-195.64714286],
		[863.46571429],
		[-644.52142857]
	])
	assert np.allclose(result_11, answer_11)

	# Example 1.2:
	result_12 = vec_reg_linear_grad(y, x, theta, 1)
	print(f'Example 1.2: {result_12}\n')
	answer_12 = np.array([
		[-60.99],
		[-195.64714286],
		[863.46571429],
		[-644.52142857]
	])
	assert np.allclose(result_12, answer_12)

	# Example 2.1:
	result_21 = reg_linear_grad(y, x, theta, 0.5)
	print(f'Example 2.1: {result_21}\n')
	answer_21 = np.array([
		[-60.99],
		[-195.86142857],
		[862.71571429],
		[-644.09285714]
	])
	assert np.allclose(result_21, answer_21)

	# Example 2.2:
	result_22 = vec_reg_linear_grad(y, x, theta, 0.5)
	print(f'Example 2.2: {result_22}\n')
	answer_22 = np.array([
		[-60.99],
		[-195.86142857],
		[862.71571429],
		[-644.09285714]
	])
	assert np.allclose(result_22, answer_22)

	# Example 3.1:
	result_31 = reg_linear_grad(y, x, theta, 0.0)
	print(f'Example 3.1: {result_31}\n')
	answer_31 = np.array([
		[-60.99],
		[-196.07571429],
		[861.96571429],
		[-643.66428571]
	])
	assert np.allclose(result_31, answer_31)

	# Example 3.2:
	result_32 = vec_reg_linear_grad(y, x, theta, 0.0)
	print(f'Example 3.2: {result_32}\n')
	answer_32 = np.array([
		[-60.99],
		[-196.07571429],
		[861.96571429],
		[-643.66428571]
	])
	assert np.allclose(result_32, answer_32)


if __name__ == '__main__':
	main()
