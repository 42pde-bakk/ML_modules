import numpy as np
from prediction import predict_


def main() -> None:
	x = np.arange(1, 6)
	# Example 1:
	theta1 = np.array([[5], [0]])
	res = predict_(x, theta1)
	print(f'res =\n{res}')
	assert (res == np.array([
		[5.], [5.], [5.], [5.], [5.]
	])).all()
	# Do you remember why y_hat contains only 5’s here?
	# Example 2:
	theta2 = np.array([[0], [1]])
	res = predict_(x, theta2)
	print(f'res =\n{res}')
	assert (res == np.array([
		[1.], [2.], [3.], [4.], [5.]
	])).all(), f'my result is {res} and wanted = {np.array([[1.], [2.], [3.], [4.], [5.]])}'
	# Do you remember why y_hat == x here?
	# Example 3:
	theta3 = np.array([[5], [3]])
	res = predict_(x, theta3)
	print(f'res =\n{res}')
	assert (res == np.array([
		[8.], [11.], [14.], [17.], [20.]
	])).all()
	# Example 4:
	theta4 = np.array([[-3], [1]])
	res = predict_(x, theta4)
	print(f'res =\n{res}')
	assert (res == np.array([
		[-2.], [-1.], [0.], [1.], [2.]
	])).all()


if __name__ == '__main__':
	main()
