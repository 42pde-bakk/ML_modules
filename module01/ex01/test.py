import numpy as np
from vec_gradient import simple_gradient


def main() -> None:
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	res = simple_gradient(x, y, theta1)
	print(f'first simple gradient is {res}')
	a = np.array([
		[-19.0342574],
		[-586.66875564]
	])
	assert np.allclose(res, a)

	# Example 1:
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	res = simple_gradient(x, y, theta2)
	print(f'first simple gradient is {res}\n')
	assert np.allclose(res, np.array([
		[-57.86823748],
		[-2230.12297889]
	]))


def correction_tests() -> None:
	theta_1 = np.array([[1.0], [1.0]])
	theta_2 = np.array([[4.0], [-1.0]])
	answers_1 = [np.array([[-11.625], [-795.375]]), np.array([[-124.125], [-82957.875]]), np.array([[-1.24912500e+03], [-8.32958288e+06]])]
	answers_2 = [np.array([[-13.625], [-896.375]]), np.array([[-126.125], [-83958.875]]), np.array([[-1.25112500e+03], [-8.33958388e+06]])]

	print(f'Correction tests:\n')
	for n, answer in zip([100, 1000, 10000], answers_1):
		x = np.array(range(1, n + 1)).reshape(-1, 1)
		y = 1.25 * x
		result = simple_gradient(x, y, theta_1)
		print(f'Result = {result}\n')
		assert np.allclose(result, answer)

	print('Correction tests part 2:\n')
	for n, answer in zip([100, 1000, 10000], answers_2):
		x = np.array(range(1, n + 1)).reshape(-1, 1)
		y = -0.75 * x + 5
		result = simple_gradient(x, y, theta_2)
		print(f'Result = {result}\n')
		assert np.allclose(result, answer)


if __name__ == '__main__':
	main()
	correction_tests()
