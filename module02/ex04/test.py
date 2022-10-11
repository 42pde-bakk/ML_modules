import numpy as np
from fit import fit_, predict_


def main() -> None:
	x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
	theta = np.array([[42.], [1.], [1.], [1.]])

	# Example 0:
	theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
	answer = np.array([
		[41.99888822],
		[0.97792316],
		[0.77923161],
		[-1.20768386]
	])
	print(f'{theta2 = }')
	assert np.allclose(theta2, answer)

	# Example 1:
	preds = predict_(x, theta2)
	print(f'predictions = {preds}')
	# Output:
	answer = np.array([
		[19.59925884],
		[-2.80037055],
		[-25.19999994],
		[-47.59962933]
	])
	assert np.allclose(preds, answer)


if __name__ == '__main__':
	main()
