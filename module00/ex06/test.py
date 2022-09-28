import numpy as np
from loss import loss_, loss_elem_
from prediction import predict_


def main() -> None:
	x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	theta1 = np.array([[2.], [4.]])
	y_hat1 = predict_(x1, theta1)
	y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

	assert (loss_elem_(y1, y_hat1) == np.array([
		[0.], [1], [4], [9], [16]
	])).all()

	assert loss_(y1, y_hat1) == 3.0

	x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
	theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
	y_hat2 = predict_(x2, theta2)
	y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
	assert loss_(y2, y_hat2) == 2.142857142857143
	assert loss_(y2, y2) == 0.0


if __name__ == '__main__':
	main()
