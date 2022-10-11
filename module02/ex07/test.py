import numpy as np
from polynomial_model import add_polynomial_features


def main() -> None:
	x = np.arange(1, 6).reshape(-1, 1)

	# Example 0:
	print('Example 0')
	result = add_polynomial_features(x, 3)
	print(f'{result = }')
	# Output:
	answer = np.array([
		[1, 1, 1],
		[2, 4, 8],
		[3, 9, 27],
		[4, 16, 64],
		[5, 25, 125]
	])
	assert np.equal(result, answer).all()

	# Example 1:
	print('\n\nExample 1')
	result = add_polynomial_features(x, 6)
	print(f'{result = }')
	# Output:
	answer = np.array([
		[1, 1, 1, 1, 1, 1],
		[2, 4, 8, 16, 32, 64],
		[3, 9, 27, 81, 243, 729],
		[4, 16, 64, 256, 1024, 4096],
		[5, 25, 125, 625, 3125, 15625]
	])
	assert np.equal(result, answer).all()


if __name__ == '__main__':
	main()
