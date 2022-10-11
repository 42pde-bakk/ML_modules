import numpy as np
from mylinearregression import MyLinearRegression as MyLR


def main() -> None:
	X = np.array([
		[1., 1., 2., 3.],
		[5., 8., 13., 21.],
		[34., 55., 89., 144.]
	])
	Y = np.array([
		[23.],
		[48.],
		[218.]
	])
	thetas = np.array([
		[1.],
		[1.],
		[1.],
		[1.],
		[1]
	])
	mylr = MyLR(thetas)

	# Example 0:
	print('Example 0:')
	y_hat = mylr.predict_(X)
	expected_yhat = np.array([
		[8.],
		[48.],
		[323.]
	])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, expected_yhat)

	# Example 1:
	print('\n\nExample 1:')
	my_loss_elem = mylr.loss_elem_(Y, y_hat)
	expected_loss_elem = np.array([
		[225.],
		[0.],
		[11025.]
	])
	print(f'{my_loss_elem = }')
	assert np.allclose(my_loss_elem, expected_loss_elem)

	# Example 2:
	print('\n\nExample 2:')
	my_loss = mylr.loss_(Y, y_hat)
	expected_loss = 1875.0
	print(f'{my_loss = }')
	assert np.allclose(my_loss, expected_loss)

	# Example 3:
	print('\n\nExample 3:')
	mylr.alpha = 1.6e-4
	print(mylr.alpha)
	mylr.max_iter = 200000
	new_thetas = mylr.fit_(X, Y)
	# expected_new_thetas = np.array([
	# 	[18.1883792],
	# 	[2.76697788],
	# 	[-0.374782024],
	# 	[1.39219585],
	# 	[0.174138279]
	# ])
	expected_new_thetas = np.array([
		[1.81883792e+01],   # [18.188]
		[2.76697788e+00],   # [2.767]
		[-3.74782024e-01],  # [-0.374]
		[1.39219585e+00],   # [1.392]
		[1.74138279e-02]    # [0.017]
	])

	print(f'{new_thetas = }')
	assert np.allclose(new_thetas, expected_new_thetas)

	# Example 4:
	print('\n\nExample 4:')
	y_hat = mylr.predict_(X)
	expected_yhat = np.array([
		[23.41720822],
		[47.48924883],
		[218.06563769]
	])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, expected_yhat)

	# Example 5:
	print('\n\nExample 5:')
	my_loss_elem = mylr.loss_elem_(Y, y_hat)
	expected_loss_elem = np.array([
		[0.1740627],
		[0.26086676],
		[0.00430831]
	])
	print(f'{my_loss_elem = }')
	assert np.allclose(my_loss_elem, expected_loss_elem)

	# Example 6:
	print('\n\nExample 6:')
	my_loss = mylr.loss_(Y, y_hat)
	expected_loss = 0.0732062937695697
	print(f'{my_loss = }')
	assert np.allclose(my_loss, expected_loss)


if __name__ == '__main__':
	main()
