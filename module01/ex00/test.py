import numpy as np
from gradient import simple_gradient


def main() -> None:
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	res = simple_gradient(x, y, theta1)
	print(f'first simple_gradient:\n{res}')
	a = np.array([
		[-19.0342574],
		[-586.66875564]
	])
	assert np.allclose(res, a)

	# Example 1:
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	res = simple_gradient(x, y, theta2)
	print(f'second simple gradient:\n{res}\n')
	assert np.allclose(res, np.array([
		[-57.86823748],
		[-2230.12297889]
	]))


def correction_tests() -> None:
	x = np.array(range(1, 11)).reshape(-1, 1)
	y = 1.25 * x

	print(f'Correction tests:\n')
	theta_1 = np.array([[1.0], [1.0]])
	result_1 = simple_gradient(x, y, theta_1)
	print(f'Example 1: {result_1}\n')
	answer_1 = np.array([[-0.375], [-4.125]])
	assert np.allclose(result_1, answer_1)

	theta_2 = np.array([[1.], [-0.4]])
	result_2 = simple_gradient(x, y, theta_2)
	print(f'Example 2: {result_2}\n')
	answer_2 = np.array([[-8.075], [-58.025]])
	assert np.allclose(result_2, answer_2)

	theta_3 = np.array([[0.], [1.25]])
	result_3 = simple_gradient(x, y, theta_3)
	print(f'Example 3: {result_3}\n')
	answer_3 = np.array([[-0.0], [0.0]])
	assert np.allclose(result_3, answer_3)


if __name__ == '__main__':
	main()
	correction_tests()
