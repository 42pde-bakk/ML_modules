import numpy as np
from l2_reg import l2, iterative_l2


def main():
	x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

	# Example 1:
	my_result = iterative_l2(x)
	print(f'Example 1: {my_result}\n')
	assert np.isclose(my_result, 911.0)

	# Example 2:
	my_result = l2(x)
	print(f'Example 2: {my_result}\n')
	assert np.isclose(my_result, 911.0)

	y = np.array([3, 0.5, -6]).reshape((-1, 1))

	# Example 3:
	my_result = iterative_l2(y)
	print(f'Example 3: {my_result}\n')
	assert np.isclose(my_result, 36.25)

	# Example 4:
	my_result = l2(y)
	print(f'Example 4: {my_result}\n')
	assert np.isclose(my_result, 36.25)


if __name__ == '__main__':
	main()
