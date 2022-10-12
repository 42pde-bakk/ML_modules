import numpy as np
from data_splitter import data_splitter


def main():
	x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

	# Example 1:
	result = data_splitter(x1, y, 0.8)
	print(f'Example 1:\n{result}\n')

	# Example 2:
	result = data_splitter(x1, y, 0.5)
	print(f'Example 2:\n{result}\n')

	x2 = np.array([
		[1, 42],
		[300, 10],
		[59, 1],
		[300, 59],
		[10, 42]
	])
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

	# Example 3:
	result = data_splitter(x2, y, 0.8)
	print(f'Example 3:\n{result}\n')

	# Example 4:
	result = data_splitter(x2, y, 0.5)
	print(f'Example 4:\n{result}\n')


if __name__ == '__main__':
	main()
