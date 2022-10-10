from math import sqrt
import numpy as np


def accepts(*types):
	def check_accepts(f):
		assert len(types) == f.__code__.co_argcount

		def new_f(*args, **kwargs):
			if not args or any(not isinstance(arg, t) for arg, t in zip(args, types)):
				return None
			if len(args[0]) == 0:
				return None
			return f(*args, **kwargs)
		# new_f.__name__ = f.__name__
		return new_f
	return check_accepts


class TinyStatistician:
	@staticmethod
	@accepts(np.ndarray)
	def mean(x: np.ndarray) -> float | None:
		# if not x or not isinstance(x, np.ndarray):
		# 	return None
		return np.sum(x) / x.shape[0]

	@staticmethod
	@accepts(np.ndarray)
	def median(x: np.ndarray) -> float | None:
		arr = np.sort(x)
		if arr.shape[0] % 2 == 1:
			mid_idx = (arr.shape[0] - 1) // 2
			return arr[mid_idx]
		else:
			idx_a, idx_b = arr.shape[0] // 2 - 1, arr.shape[0] // 2
			return (arr[idx_a] + arr[idx_b]) / 2

	@staticmethod
	@accepts(np.ndarray)
	def quartile(x: np.ndarray) -> list[float] | None:
		arr = np.sort(x)
		middle_index = arr.shape[0] // 2
		if arr.shape[0] % 2 == 1:
			lst_a, lst_b = arr[:middle_index + 1], arr[middle_index:]
		else:
			lst_a, lst_b = arr[:middle_index], arr[middle_index:]
		return [TinyStatistician.median(lst_a), TinyStatistician.median(lst_b)]

	@staticmethod
	@accepts(np.ndarray, (int, float))
	def percentile(x: np.ndarray, p: float | int) -> float | None:
		arr = np.sort(x)
		idx = (arr.shape[0] - 1) * (p / 100)
		if idx.is_integer():
			return arr[idx]
		floor, ceiling = int(idx), int(idx + 1.0)
		return arr[floor] + (idx - floor) * (arr[ceiling] - arr[floor])

	@staticmethod
	@accepts(np.ndarray)
	def var(x: np.ndarray) -> float | None:
		mean = TinyStatistician.mean(x)
		return sum([(elem - mean) ** 2 for elem in x]) / (x.shape[0] - 1)

	@staticmethod
	@accepts(np.ndarray)
	def std(x: np.ndarray) -> float | None:
		return sqrt(TinyStatistician.var(x))
