import numpy as np
from loss import loss_


def main() -> None:
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

	# Example 1:
	print(f'Example 1:')
	result = loss_(X, Y)
	print(f'{result = }')
	assert np.isclose(result, 2.142857142857143)

	# Example 2:
	print(f'Example 2:')
	result = loss_(X, X)
	print(f'{result = }')
	assert np.isclose(result, 0.0)


if __name__ == '__main__':
	main()
