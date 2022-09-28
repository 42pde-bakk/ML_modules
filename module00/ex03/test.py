from tools import add_intercept
import numpy as np


def main() -> None:
	x = np.arange(1, 6)
	print(add_intercept(x))
	assert (add_intercept(x) == np.array([
		[1., 1.],
		[1., 2.],
		[1., 3.],
		[1., 4.],
		[1., 5.]
	])).all()
	y = np.arange(1, 10).reshape((3, 3))
	print(add_intercept(y))
	assert (add_intercept(y) == np.array([
		[1., 1., 2., 3.],
		[1., 4., 5., 6.],
		[1., 7., 8., 9.]
	])).all()


if __name__ == '__main__':
	main()
