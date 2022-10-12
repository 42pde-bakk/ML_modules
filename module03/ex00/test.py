import numpy as np
from sigmoid import sigmoid_, real_sigmoid


def main():
	# Example 1:
	print('Example 1:')
	x = np.array([[-4]])
	result = sigmoid_(x)
	answer = np.array([[0.01798620996209156]])
	print(result)
	assert np.allclose(result, answer)
	assert np.allclose(real_sigmoid(x, 1, 1), answer)

	# Example 2:
	print('\nExample 2:')
	x = np.array([[2]])
	result = sigmoid_(x)
	answer = np.array([[0.8807970779778823]])
	print(result)
	assert np.allclose(result, answer)

	# Example 3:
	print('\nExample 3:')
	x = np.array([
		[-4],
		[2],
		[0]
	])
	result = sigmoid_(x)
	answer = np.array([
		[0.01798620996209156],
		[0.8807970779778823],
		[0.5]
	])
	print(result)
	assert np.allclose(result, answer)
	assert np.allclose(real_sigmoid(x, 1, 1), answer)


if __name__ == '__main__':
	main()
