import numpy as np
from ridge import MyRidge


def test_params():
	print(f'Testing MyRidge.set_params and MyRidge.get_params:\n\n')

	thetas = np.array([
		[-2.4],
		[-1.5],
	])
	thetas2 = np.array([
		[6.9],
		[69.0],
		[0.69],
		[13.37]
	])
	model = MyRidge(thetas)
	print(model.get_params())
	assert model.get_params().keys() == {'thetas', 'alpha', 'lambda_', 'max_iter'}

	model.set_params(thetas=thetas2, lambda_=1.0, max_iter=500_000_000)
	print(model.get_params())
	assert np.allclose(model.thetas, thetas2)
	assert model.lambda_ == 1.0
	assert model.max_iter == 500_000_000


def test_l2():
	print(f'Testing MyRidge.l2:\n\n')

	thetas = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	model = MyRidge(thetas)
	my_result = model.l2()
	print(f'L2:: Example 1: {my_result}\n')
	assert np.isclose(my_result, 911.0)

	y = np.array([3, 0.5, -6]).reshape((-1, 1))
	model.set_params(thetas=y)
	my_result = model.l2()
	print(f'L2:: Example 2: {my_result}\n')
	assert np.isclose(my_result, 36.25)


def test_loss():
	print(f'Testing MyRidge.loss_ and MyRidge.loss_elem_:\n\n')

	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	model = MyRidge(theta, lambda_=0.5)

	# Example 1:
	result = model.loss_(y, y_hat)
	print(f'Loss:: Example 1: {result}\n')
	assert np.isclose(result, 0.8503571428571429)

	# Example 2:
	model.set_params(lambda_=0.05)
	result = model.loss_(y, y_hat)
	print(f'Loss:: Example 2: {result}\n')
	assert np.isclose(result, 0.5511071428571429)

	# Example 3:
	model.set_params(lambda_=0.9)
	result = model.loss_(y, y_hat)
	print(f'Loss:: Example 3: {result}\n')
	assert np.isclose(result, 1.116357142857143)

	# Test Loss elem
	result = model.loss_elem_(y, y_hat)
	answer = (y_hat - y) * (y_hat - y) + model.lambda_ * model.l2()
	print(f'Loss_elem:: Example 1: {result}')
	assert np.allclose(result, answer)


def test_predict():
	print(f'Testing MyRidge.predict_:\n\n')
	X = np.array([
		[1., 1., 2., 3.],
		[5., 8., 13., 21.],
		[34., 55., 89., 144.]
	])
	thetas = np.array([
		[1.],
		[1.],
		[1.],
		[1.],
		[1]
	])
	mylr = MyRidge(thetas)

	print('Example 1:')
	y_hat = mylr.predict_(X)
	expected_yhat = np.array([
		[8.],
		[48.],
		[323.]
	])
	print(f'{y_hat = }')
	assert np.allclose(y_hat, expected_yhat)


def test_gradient():
	print(f'Testing MyRidge.gradient_:\n\n')
	x = np.array([
		[-6, -7, -9],
		[13, -2, 14],
		[-7, 14, -1],
		[-8, -4, 6],
		[-5, -9, 6],
		[1, -5, 11],
		[9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])
	model = MyRidge(theta, lambda_=1)

	# Example 1:
	result_1 = model.gradient_(y, x)
	print(f'Example 1: {result_1}\n')
	answer_1 = np.array([
		[-60.99],
		[-195.64714286],
		[863.46571429],
		[-644.52142857]
	])
	print(f'{result_1.shape}, {answer_1.shape}')
	assert np.allclose(result_1, answer_1)

	# Example 2:
	model.set_params(lambda_=0.5)
	result_2 = model.gradient_(y, x)
	print(f'Example 2: {result_2}\n')
	answer_2 = np.array([
		[-60.99],
		[-195.86142857],
		[862.71571429],
		[-644.09285714]
	])
	assert np.allclose(result_2, answer_2)

	# Example 3:
	model.set_params(lambda_=0.0)
	result_3 = model.gradient_(y, x)
	print(f'Example 3: {result_3}\n')
	answer_3 = np.array([
		[-60.99],
		[-196.07571429],
		[861.96571429],
		[-643.66428571]
	])
	assert np.allclose(result_3, answer_3)


def test_fit():
	print(f'Testing MyRidge.fit_:\n\n')
	X = np.array([
		[1., 1., 2., 3.],
		[5., 8., 13., 21.],
		[34., 55., 89., 144.]])
	print(f'x.shape = {X.shape}')
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
		[1.]
	])

	model = MyRidge(thetas, alpha=1.6e-4, max_iter=200_000, lambda_=0.6)
	print(model.fit_)
	model.fit_(X, Y)
	print(f'Newly fitted thetas: {model.thetas}')


if __name__ == '__main__':
	# test_params()
	# test_l2()
	# test_loss()
	# test_predict()
	# test_gradient()
	test_fit()
