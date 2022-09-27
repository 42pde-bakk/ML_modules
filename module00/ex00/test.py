from matrix import Matrix


def main() -> None:
    # Simple constructor and Matrix.T() tests
    assert Matrix((3, 3)).shape == (3, 3), 'Error in Matrix((3, 3))'
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    assert m1.shape == (3, 2), 'Shape wrong'
    assert m1.data == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]

    m1_t = m1.T()
    assert m1_t.shape == (2, 3)
    assert m1_t.data == [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]

    m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    assert m1.shape == (2, 3)

    m1_t = m1.T()
    assert m1_t.shape == (3, 2)
    assert m1_t.data == [[0., 1.], [2., 3.], [4., 5.]]

    # Let's test matrix multiplication

    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])

    matmult = m1 * m2
    print(matmult)
    assert matmult.shape == (2, 2)
    assert matmult.data == [[28., 34.], [56., 68]]


if __name__ == '__main__':
    main()
