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


def correction_tests():
	x = np.array([0])
	result_1 = sigmoid_(x)
	answer_1 = np.array([0.5])
	print(f'Correction test 1: {result_1}')
	assert np.allclose(result_1, answer_1)

	x = np.array([1])
	result_2 = sigmoid_(x)
	answer_2 = np.array([0.73105857863])
	print(f'Correction test 2: {result_2}')
	assert np.allclose(result_2, answer_2)

	x = np.array([-1])
	result_3 = sigmoid_(x)
	answer_3 = np.array([0.26894142137])
	print(f'Correction test 3: {result_3}')
	assert np.allclose(result_3, answer_3)

	x = np.array([50])
	result_4 = sigmoid_(x)
	answer_4 = np.array([1])
	print(f'Correction test 4: {result_4}')
	assert np.allclose(result_4, answer_4)

	x = np.array([-50])
	result_5 = sigmoid_(x)
	answer_5 = np.array([1.928749847963918e-22])
	print(f'Correction test 1: {result_5}')
	assert np.allclose(result_5, answer_5)

	x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
	result_6 = sigmoid_(x)
	answer_6 = np.array([0.07585818, 0.18242552, 0.37754067, 0.62245933, 0.81757448, 0.92414182])
	print(f'Correction test 6: {result_6}')
	assert np.allclose(result_6, answer_6)


if __name__ == '__main__':
	main()
	correction_tests()
