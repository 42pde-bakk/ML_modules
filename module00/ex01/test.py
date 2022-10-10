from TinyStatistician import TinyStatistician
import numpy as np


def main() -> None:
	tstat = TinyStatistician()
	a = np.array([1, 42, 300, 10, 59])
	mean = tstat.mean(a)
	print(f'{mean = }')
	assert mean == 82.4
	median = tstat.median(a)
	print(f'{median = }')
	assert median == 42.0
	quartiles = tstat.quartile(a)
	print(f'{quartiles = }')
	assert quartiles == [10.0, 59.0]
	variance = tstat.var(a)
	print(f'{variance = }')
	assert variance == 15349.3
	std_deviation = tstat.std(a)
	print(f'{std_deviation = }')
	assert std_deviation == 123.89229193133849

	percentiles: dict = {
		10: tstat.percentile(a, 10),
		15: tstat.percentile(a, 15),
		20: tstat.percentile(a, 20)
	}
	print(f'{percentiles = }')
	assert abs(percentiles[10] - 4.6) < 1e-5, f'I came up with {percentiles[10]}'
	assert abs(percentiles[15] - 6.4) < 1e-5, f'I came up with {percentiles[15]}'
	assert abs(percentiles[20] - 8.2) < 1e-5, f'I came up with {percentiles[20]}'


if __name__ == '__main__':
	main()
