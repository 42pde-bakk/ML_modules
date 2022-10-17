import numpy as np
from fit import fit_, predict


def main() -> None:
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	theta = np.array([1, 1]).reshape((-1, 1))

	theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1_500_000)
	print(f'{theta1 = }')
	assert np.allclose(theta1, np.array([
		[1.40709365],
		[1.1150909]
	]))

	preds = predict(x, theta1)
	print(f'{preds = }')
	assert np.allclose(preds, np.array([
		[15.3408728],
		[25.38243697],
		[36.59126492],
		[55.95130097],
		[65.53471499]
	]))


def correction_tests() -> None:
	x = np.array(range(1, 101)).reshape(-1, 1)
	y = 0.75 * x + 5
	theta = np.array([[1.], [1.]])

	print(type(x), type(y), type(theta), type(1e-5), type(20000))
	result = fit_(x, y, theta, alpha=1e-5, max_iter=20000)
	answer = np.array([[1.18949688], [0.80687732]])
	print(f'Correction result: {result}')
	assert np.allclose(result, answer)


if __name__ == '__main__':
	main()
	# correction_tests()
