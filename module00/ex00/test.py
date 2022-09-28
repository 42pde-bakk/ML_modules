from matrix import Matrix, Vector


def main() -> None:
	# Simple constructor and Matrix.T() tests
	assert Matrix((3, 3)).shape == (3, 3), 'Error in Matrix((3, 3))'
	m1 = Matrix([
		[0.0, 1.0],
		[2.0, 3.0],
		[4.0, 5.0]
	])
	assert m1.shape == (3, 2), 'Shape wrong'
	assert m1.data == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]

	m1_t = m1.T()
	assert m1_t.shape == (2, 3)
	assert m1_t.data == [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]

	# Maybe also do some tests on __repr__ and __str__ ?

	m1 = Matrix([
		[0., 2., 4.],
		[1., 3., 5.]
	])
	assert m1.shape == (2, 3)

	m1_t = m1.T()
	assert m1_t.shape == (3, 2)
	assert m1_t.data == [[0., 1.], [2., 3.], [4., 5.]]

	# Let's test matrix multiplication

	m1 = Matrix([
		[0.0, 1.0, 2.0, 3.0],
		[0.0, 2.0, 4.0, 6.0]
	])
	m2 = Matrix([
		[0.0, 1.0],
		[2.0, 3.0],
		[4.0, 5.0],
		[6.0, 7.0]
	])

	matmult = m1 * m2
	assert matmult.shape == (2, 2)
	assert matmult.data == [[28., 34.], [56., 68]]

	m1_plus_m1 = m1 + m1
	assert m1_plus_m1.shape == m1.shape
	assert m1_plus_m1.data == [[0., 2., 4., 6.], [0., 4., 8., 12.]]

	v1 = Vector([
		[1, 2, 3]
	])  # create a row vector
	assert v1.shape == (1, 3)
	v2 = Vector([
		[1],
		[2],
		[3]
	])  # create a column vector
	assert v2.shape == (3, 1)

	try:
		v3 = Vector([[1, 2], [3, 4]])  # return an error
		assert False, 'v3 = Vector([[1, 2], [3, 4]]) should have raised an error!'
	except TypeError:
		pass

	m1 = Matrix([
		[0.0, 1.0, 2.0],
		[0.0, 2.0, 4.0]
	])
	v1 = Vector([
		[1],
		[2],
		[3]
	])
	v1_m1 = m1 * v1
	assert isinstance(v1_m1, Vector)
	assert v1_m1.shape == (m1.shape[0], v1.shape[1])

	v1 = Vector([
		[1],
		[2],
		[3]
	])
	v2 = Vector([
		[2],
		[4],
		[8]
	])
	v12 = v1 + v2
	assert v12.shape == (3, 1)
	assert v12.data == [[3], [6], [11]]

	dot = v1.dot(v2)
	assert isinstance(dot, (int, float))
	assert dot == 1 * 2 + 2 * 4 + 3 * 8  # 34

	print(repr(v12))
	assert type(eval(repr(v12))) == Vector
	assert type(eval(repr(m1))) == Matrix
	print(str(v12))


if __name__ == '__main__':
	main()
