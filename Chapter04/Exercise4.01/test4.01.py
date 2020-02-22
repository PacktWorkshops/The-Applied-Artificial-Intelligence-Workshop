import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing


class Test(unittest.TestCase):
	def setUp(self):
		import Exercise4_01
		self.exercises = Exercise4_01

		import numpy as np
		def entropy(probabilities):
			minus_probabilities = [-x for x in probabilities]
			log_probabilities = [x for x in map(np.log2, probabilities)]
			return np.dot(minus_probabilities, log_probabilities)

		self.H_employed = entropy([4 / 7, 3 / 7])
		self.H_income = entropy([1 / 7, 2 / 7, 1 / 7, 2 / 7, 1 / 7])
		self.H_loanType = entropy([3 / 7, 2 / 7, 2 / 7])
		self.H_LoanAmount = entropy([1 / 7, 1 / 7, 3 / 7, 1 / 7, 1 / 7])

	def test_H_employed(self):
		self.assertEqual(self.exercises.H_employed, self.H_employed)

	def test_H_income(self):
		self.assertEqual(self.exercises.H_income, self.H_income)

	def test_H_loanType(self):
		self.assertEqual(self.exercises.H_loanType, self.H_loanType)

	def test_H_LoanAmount(self):
		self.assertEqual(self.exercises.H_LoanAmount, self.H_LoanAmount)


if __name__ == '__main__':
	unittest.main()
