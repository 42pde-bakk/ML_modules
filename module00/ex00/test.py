from matrix import Matrix


def main() -> None:
	matrix = Matrix((3, 3))
	# print(f'{matrix.shape = }')
	# print(f'{matrix.data = }')
	assert matrix.shape == (3, 3)

	m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
	print(f'{m1.shape = }')
	print(f'{m1.data = }')
	assert m1.shape == (3, 2)

	m1_t = m1.T()
	assert m1_t.shape == (2, 3)

	m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
	assert m1.shape == (2, 3)

	m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
				 [0.0, 2.0, 4.0, 6.0]])
	m2 = Matrix([[0.0, 1.0],
				 [2.0, 3.0],
				 [4.0, 5.0],
				 [6.0, 7.0]])



if __name__ == '__main__':
	main()
