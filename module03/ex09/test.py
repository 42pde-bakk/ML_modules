import pandas as pd
import numpy as np
from confusion_matrix import confusion_matrix_
from sklearn.metrics import confusion_matrix


def main():
	y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
	y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])

	# Example 1:
	my_matrix = confusion_matrix_(y, y_hat)
	print(f'{my_matrix = }')
	sklearn_matrix = confusion_matrix(y, y_hat)
	# answer = np.array([
	# 	[0, 0, 0],
	# 	[0, 2, 1],
	# 	[1,	0, 2]
	# ])
	assert np.array_equal(my_matrix, sklearn_matrix)

	# Example 2:
	my_matrix = confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])
	print(f'{my_matrix = }')
	sklearn_matrix = confusion_matrix(y, y_hat, labels=['dog', 'norminet'])
	# answer = np.array([
	# 	[2, 1],
	# 	[0, 2]
	# ])
	assert np.array_equal(my_matrix, sklearn_matrix)

	print(f'\nOptional part:\n')
	# Optional part:
	# Example 3:
	print(confusion_matrix_(y, y_hat, df_option=True))

	# Example 4
	print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))


if __name__ == '__main__':
	main()
