import numpy as np
from minmax import minmax


def main():
	# Example 1:
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	print(f'{X=}')
	answer = np.array([0.58333333, 1., 0.33333333, 0.77777778, 0.91666667,
	       0.66666667, 0.]).reshape(-1, 1)
	result = minmax(X)
	print(f'\n{result}\n')
	assert np.allclose(result, answer)

	# Example 2:
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	answer = np.array([0.63636364, 1., 0.18181818, 0.72727273, 0.93939394,
	       0.6969697, 0.]).reshape(-1, 1)
	result = minmax(Y)
	print(result)
	assert np.allclose(result, answer)


if __name__ == '__main__':
	main()
