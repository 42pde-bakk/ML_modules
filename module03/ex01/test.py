import numpy as np
from log_pred import logistic_predict_


def main():
	# Example 1
	x = np.array([4]).reshape((-1, 1))
	theta = np.array([
		[2],
		[0.5]
	])
	result = logistic_predict_(x, theta)
	answer = np.array([
		[0.98201379]
	])
	print(f'Example 1:\n{result}')
	assert np.allclose(result, answer)

	# Example 2
	x2 = np.array([
		[4],
		[7.16],
		[3.2],
		[9.37],
		[0.56]
	])
	theta2 = np.array([
		[2],
		[0.5]
	])
	result = logistic_predict_(x2, theta2)
	answer = np.array([
		[0.98201379],
		[0.99624161],
		[0.97340301],
		[0.99875204],
		[0.90720705]
	])
	print(f'\nExample 2:\n{result}')
	assert np.allclose(result, answer)

	# Example 3
	x3 = np.array([
		[0, 2, 3, 4],
		[2, 4, 5, 5],
		[1, 3, 2, 7]
	])
	theta3 = np.array([
		[-2.4],
		[-1.5],
		[0.3],
		[-1.4],
		[0.7]
	])
	result = logistic_predict_(x3, theta3)
	# Output:
	answer = np.array([
		[0.03916572],
		[0.00045262],
		[0.2890505]
	])
	print(f'\nExample 3:\n{result}')
	assert np.allclose(result, answer)


if __name__ == '__main__':
	main()
