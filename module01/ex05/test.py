from z_score import zscore
import numpy as np


def main():
	# Example 1:
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	x_result = zscore(X)
	print(f'Example 1:\n{x_result}')
	x_answer = np.array([
		-0.08620324, 1.2068453, -0.86203236, 0.51721942,
		0.94823559, 0.17240647, -1.89647119]
	)
	assert np.allclose(x_result, x_answer)
	print(f'{x_result.shape=}, {x_answer.shape=}')

	# Example 2:
	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	y_result = zscore(Y)
	y_answer = np.array([
		0.11267619, 1.16432067, -1.20187941, 0.37558731,
		0.98904659, 0.28795027, -1.72770165]
	)
	print(f'Example 2:\n{y_result}')
	assert np.allclose(y_result, y_answer)


if __name__ == '__main__':
	main()
