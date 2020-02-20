import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors

class Test(unittest.TestCase):
	def setUp(self):
		import Exercise1_01
		self.exercises = Exercise1_01

		self.A = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		self.matmult = self.A * self.A
		self.det = np.linalg.det(self.A)
		self.transpose = np.matrix.transpose(self.A)

	def test_A(self):
		np_testing.assert_array_equal(self.exercises.A, self.A)

	def test_matmult(self):
		np_testing.assert_array_equal(self.exercises.matmult, self.matmult)

	def test_det(self):
		self.assertEqual(self.exercises.det, self.det)

	def test_transpose(self):
		np_testing.assert_array_equal(self.exercises.transpose, self.transpose)

if __name__ == '__main__':
	unittest.main()
