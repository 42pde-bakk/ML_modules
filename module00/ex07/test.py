import numpy as np
from vec_loss import loss_


def main() -> None:
	x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	# Example 1:
	assert loss_(x, y) == 2.142857142857143
	# Example 2:
	assert loss_(x, x) == 0.0


if __name__ == '__main__':
	main()
