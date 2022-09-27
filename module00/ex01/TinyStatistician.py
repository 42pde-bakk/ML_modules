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
	def quartiles(x: np.ndarray) -> list[float] | None:
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
		idx = (arr.shape[0] - 1) * p / 100
		if idx.is_integer():
			return arr[idx]
		f, c = int(idx), int(idx + 1.0)
		f_diff, c_diff = idx - f, c - idx
		return c_diff * arr[f] + f_diff * arr[c]

	@staticmethod
	@accepts(np.ndarray)
	def var(x: np.ndarray) -> float | None:
		mean = TinyStatistician.mean(x)
		return sum([(elem - mean) ** 2 for elem in x]) / x.shape[0]

	@staticmethod
	@accepts(np.ndarray)
	def std(x: np.ndarray) -> float | None:
		return sqrt(TinyStatistician.var(x))


if __name__ == '__main__':
	tstat = TinyStatistician
	a = np.array([1, 42, 300, 10, 59])
	# assert tstat.mean(a) == 82.4, f'I came up with {tstat.mean(a)}'
	# assert tstat.median(a) == 42.0, f'I came up with {tstat.median(a)}'
	# assert tstat.quartiles(a) == [10.0, 59.0], f'I came up with {tstat.quartiles(a)}'
	# assert tstat.var(a) == 12279.439999999999, f'I came up with {tstat.var(a)}'
	# assert tstat.std(a) == 110.81263465868862, f'I came up with {tstat.std(a)}'

	assert tstat.percentile(a, 10) == 4.6, f'I came up with {tstat.percentile(a, 10)}'
	assert tstat.percentile(a, 15) == 6.4, f'I came up with {tstat.percentile(a, 10)}'
	assert tstat.percentile(a, 20) == 8.2, f'I came up with {tstat.percentile(a, 10)}'
