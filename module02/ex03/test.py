import numpy as np
from gradient import gradient


def main() -> None:
	x = np.array([
		[-6, -7, -9],
		[13, -2, 14],
		[-7, 14, -1],
		[-8, -4, 6],
		[-5, -9, 6],
		[1, -5, 11],
		[9, -11, 8]
	])
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

	print('Example 1:')
	# I added an extra column to theta1 and theta2
	# https://github.com/42-AI/bootcamp_machine-learning/issues/225
	theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))
	result = gradient(x, y, theta1)
	print(result)
	answer = np.array([[-33.71428571], [-37.35714286], [183.14285714], [-393.]])
	assert np.allclose(result, answer)

	print('Example 2:')
	theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
	result = gradient(x, y, theta2)
	print(result)
	answer = np.array([[-0.71428571], [0.85714286], [23.28571429], [-26.42857143]])
	assert np.allclose(result, answer)


if __name__ == '__main__':
	main()
