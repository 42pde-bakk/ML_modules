# from math import sqrt
import numpy as np


class TinyStatistician:
	@staticmethod
	def mean(x: np.ndarray) -> float | None:
		if not x or not isinstance(x, np.ndarray):
			return None
		return sum(item for item in x) / len(x)
		# return sum(x) / len(x)  # I would do this, but they want me to use a for-loop

	@staticmethod
	def median(x: np.ndarray) -> float | None:
		length = len(x)
		if x is None or length == 0:
			return None
		sorted_lst = sorted(x)
		if length % 2 == 1:
			middle_idx = (length - 1) // 2
			return float(sorted_lst[middle_idx])
		else:
			idx_a, idx_b = length // 2 - 1, length // 2
			print(f'{length=}, {idx_a=}, {idx_b=}')
			return float(sorted_lst[idx_a] + sorted_lst[idx_b]) / 2

	@staticmethod
	def quartiles(x: list) -> list[float] | None:
		if not x:
			return None
		sorted_x = sorted(x)
		length = len(x)
		middle_index = length // 2
		if length % 2 == 1:
			lst_a, lst_b = sorted_x[:middle_index + 1], sorted_x[middle_index:]
		else:
			lst_a, lst_b = sorted_x[:middle_index], sorted_x[middle_index:]
		return [TinyStatistician.median(lst_a), TinyStatistician.median(lst_b)]

	@staticmethod
	def var(x: list) -> float | None:
		if not x:
			return None
		mean = TinyStatistician.mean(x)
		return sum([(elem - mean) ** 2 for elem in x]) / len(x)

	@staticmethod
	def std(x: list) -> float | None:
		if not x:
			return None
		return sqrt(TinyStatistician.var(x))


if __name__ == '__main__':
	tstat = TinyStatistician
	a = [1, 42, 300, 10, 59]
	assert tstat.mean(a) == 82.4, f'I came up with {tstat.mean(a)}'
	assert tstat.median(a) == 42.0, f'I came up with {tstat.median(a)}'
	assert tstat.quartiles(a) == [10.0, 59.0], f'I came up with {tstat.quartiles(a)}'
	assert tstat.var(a) == 12279.439999999999, f'I came up with {tstat.var(a)}'
	assert tstat.std(a) == 110.81263465868862, f'I came up with {tstat.std(a)}'
