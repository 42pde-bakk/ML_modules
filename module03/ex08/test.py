import numpy as np
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def example_1():
	print(f'Example 1:')
	y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
	y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

	# Accuracy
	accuracy = accuracy_score_(y, y_hat)
	print(f'{accuracy = }')
	assert np.isclose(accuracy, accuracy_score(y, y_hat))  # 0.5

	# Precision
	precision = precision_score_(y, y_hat)
	print(f'{precision = }')
	assert np.isclose(precision, precision_score(y, y_hat))  # 0.4

	# Recall
	recall = recall_score_(y, y_hat)
	print(f'{recall = }')
	assert np.isclose(recall, recall_score(y, y_hat))  # 0.6666666666666666

	# F1-score
	f1 = f1_score_(y, y_hat)
	print(f'{f1 = }\n')
	assert np.isclose(f1, f1_score(y, y_hat))  # 0.5


def example_2():
	print(f'Example 2:')
	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

	# Accuracy
	accuracy = accuracy_score_(y, y_hat)
	print(f'{accuracy = }')
	assert np.isclose(accuracy, accuracy_score(y, y_hat))  # 0.625

	# Precision
	precision = precision_score_(y, y_hat, pos_label='dog')
	print(f'{precision = }')
	assert np.isclose(precision, precision_score(y, y_hat, pos_label='dog'))  # 0.6

	# Recall
	recall = recall_score_(y, y_hat, pos_label='dog')
	print(f'{recall = }')
	assert np.isclose(recall, recall_score(y, y_hat, pos_label='dog'))  # 0.75

	# F1-score
	f1 = f1_score_(y, y_hat, pos_label='dog')
	print(f'{f1 = }\n')
	assert np.isclose(f1, f1_score(y, y_hat, pos_label='dog'))  # 0.6666666666666665


def example_3():
	print(f'Example 3:')

	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

	# Precision
	precision = precision_score_(y, y_hat, pos_label='norminet')
	print(f'{precision = }')
	assert np.isclose(precision, precision_score(y, y_hat, pos_label='norminet'))  # 0.6666666666666666

	# Recall
	recall = recall_score_(y, y_hat, pos_label='norminet')
	print(f'{recall = }')
	assert np.isclose(recall, recall_score(y, y_hat, pos_label='norminet'))  # 0.5

	# F1-score
	f1 = f1_score_(y, y_hat, pos_label='norminet')
	print(f'{f1 = }')
	assert np.isclose(f1, f1_score(y, y_hat, pos_label='norminet'))  # 0.5714285714285715


if __name__ == '__main__':
	example_1()
	example_2()
	example_3()
