from __future__ import annotations
from copy import deepcopy


class Matrix:
	def __init__(self, arg):
		self.data = []  # List of lists
		self.shape = tuple
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

	def __add__(self, other: Matrix):
		if not isinstance(other, Matrix) or self.shape != other.shape:
			raise TypeError("add: only matrices of same dimensions")
		data = [[self.data[col][idx] + other.data[col][idx] for col in range(self.shape[0])] for idx in range(self.shape[1])]
		return Matrix(data)

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		return self.__add__(other * -1)

	def __rsub__(self, other):
		return self.__rsub__(other)

	def __truediv__(self, other: int | float) -> Matrix:
		if not isinstance(other, (int, float)):
			raise TypeError("div: only scalars")
		data = [[self.data[col][idx] / other for col in range(self.shape[0])] for idx in range(self.shape[1])]
		return Matrix(data)

	def __rtruediv__(self, other):
		return NotImplemented

	def __mul__(self, other: Matrix | int | float) -> Matrix:
		if isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError('The number of columns in matrix a has to be equal to the number of rows in matrix b')
			bt = other.T()
			data = [[sum(a * b for a, b in zip(row, row2))] for row2 in bt.data for row in self.data]
			return Matrix(data)
		elif isinstance(other, (int, float)):
			data = [[self.data[col][idx] * other for col in range(self.shape[0])] for idx in range(self.shape[1])]
			return Matrix(data)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def __str__(self):
		return str(self.data)

	def __repr__(self):
		return self.__str__()

	def T(self) -> Matrix:
		data = [[self.data[col][idx] for col in range(self.shape[0])] for idx in range(self.shape[1])]
		return Matrix(data)
