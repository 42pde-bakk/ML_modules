from __future__ import annotations
from copy import deepcopy


def get_matrix_or_vector(data: list[list]) -> Matrix | Vector:
	if len(data) == 1 or len(data[0]) == 1:
		return Vector(data)
	return Matrix(data)


class Matrix:
	def __init__(self, arg: tuple | list):
		if isinstance(arg, tuple):
			if len(arg) != 2:
				raise RuntimeError('Bad tuple given to Matrix constructor')
			self.data = [[0.0 for _ in range(arg[0])] for _ in range(arg[1])]
			self.shape = deepcopy(arg)
		elif isinstance(arg, list):
			self.data = deepcopy(arg)
			self.shape = len(arg), len(arg[0])
			pass
		else:
			raise NotImplementedError

	def _is_vector(self):
		if any(x == 0 for x in self.shape):
			return False
		return self.shape[0] == 1 or self.shape[1] == 1

	def __add__(self, other: Matrix):
		if not isinstance(other, Matrix) or self.shape != other.shape:
			raise TypeError("add: only matrices of same dimensions")
		data = [[col + col2 for col, col2 in zip(row, row2)] for row, row2 in zip(self.data, other.data)]
		return get_matrix_or_vector(data)

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		return self.__add__(other * -1)

	def __rsub__(self, other):
		return self.__rsub__(other)

	def __truediv__(self, other: int | float) -> Matrix | Vector:
		if not isinstance(other, (int, float)):
			raise TypeError("div: only scalars")
		data = [[self.data[col][idx] / other for col in range(self.shape[0])] for idx in range(self.shape[1])]
		return get_matrix_or_vector(data)

	def __rtruediv__(self, other):
		return NotImplemented

	def __mul__(self, other: Matrix | int | float) -> Matrix | Vector:
		if isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError('The number of columns in matrix a has to be equal to the number of rows in matrix b')
			bt = other.T()
			data = [[sum(a * b for a, b in zip(row, row2)) for row2 in bt.data] for row in self.data]
			return get_matrix_or_vector(data)
		elif isinstance(other, (int, float)):
			data = [[self.data[col][idx] * other for col in range(self.shape[0])] for idx in range(self.shape[1])]
			return get_matrix_or_vector(data)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def __str__(self):
		return str(self.data)

	def __repr__(self):
		return self.__str__()

	def T(self) -> Matrix | Vector:
		data = [[self.data[col][idx] for col in range(self.shape[0])] for idx in range(self.shape[1])]
		return get_matrix_or_vector(data)


class Vector(Matrix):
	def __init__(self, arg: list | Matrix | Vector):
		if isinstance(arg, Matrix):
			super().__init__(arg.data)
		else:
			super().__init__(arg)
		if self.shape[0] != 1 and self.shape[1] != 1:
			raise TypeError('ERROR: You must pass a row or column vector as arg')

	def dot(self, v: Vector) -> int | float | None:
		if self.shape not in (v.shape, v.shape[::-1]):
			print('Please make sure the shapes of both vectors match!')
			return None
		if self.shape == v.shape[::-1]:
			v = v.T()
		tot = 0
		for row, row2 in zip(self.data, v.data):
			for col, col2 in zip(row, row2):
				tot += col * col2
		return sum(sum(a * b for a, b in zip(row, row2)) for row, row2 in zip(self.data, v.data))
