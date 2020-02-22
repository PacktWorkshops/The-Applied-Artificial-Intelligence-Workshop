import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing


class Test(unittest.TestCase):
	def setUp(self):
		import Exercise4_02
		self.exercises = Exercise4_02

		import numpy as np

		self.real_labels = np.array([True, True, False, True, True])
		self.model_1_preds = np.array([True, False, False, False, False])
		self.model_2_preds = np.array([True, True, True, True, True])
		self.model_1_tp_cond = (self.real_labels == True) & (self.model_1_preds == True)
		self.model_1_tp = self.model_1_tp_cond.sum()
		self.model_1_fp = ((self.real_labels == False) & (self.model_1_preds == True)).sum()
		self.model_1_fn = ((self.real_labels == True) & (self.model_1_preds == False)).sum()
		self.model_1_precision = self.model_1_tp / (self.model_1_tp + self.model_1_fp)
		self.model_1_recall = self.model_1_tp / (self.model_1_tp + self.model_1_fn)
		self.model_1_f1 = 2 * self.model_1_precision * self.model_1_recall / (self.model_1_precision + self.model_1_recall)
		self.model_2_tp = ((self.real_labels == True) & (self.model_2_preds == True)).sum()
		self.model_2_fp = ((self.real_labels == False) & (self.model_2_preds == True)).sum()
		self.model_2_fn = ((self.real_labels == True) & (self.model_2_preds == False)).sum()
		self.model_2_precision = self.model_2_tp / (self.model_2_tp + self.model_2_fp)
		self.model_2_recall = self.model_2_tp / (self.model_2_tp + self.model_2_fn)
		self.model_2_f1 = 2 * self.model_2_precision * self.model_2_recall / (self.model_2_precision + self.model_2_recall)

	def test_real_labels(self):
		np_testing.assert_array_equal(self.exercises.real_labels, self.real_labels)

	def test_model_1_preds(self):
		np_testing.assert_array_equal(self.exercises.model_1_preds, self.model_1_preds)

	def test_model_2_preds(self):
		np_testing.assert_array_equal(self.exercises.model_2_preds, self.model_2_preds)

	def test_model_1_tp_cond(self):
		np_testing.assert_array_equal(self.exercises.model_1_tp_cond, self.model_1_tp_cond)

	def test_model_1_tp(self):
		self.assertEqual(self.exercises.model_1_tp, self.model_1_tp)

	def test_model_1_fp(self):
		self.assertEqual(self.exercises.model_1_fp, self.model_1_fp)

	def test_model_1_fn(self):
		self.assertEqual(self.exercises.model_1_fn, self.model_1_fn)

	def test_model_1_precision(self):
		self.assertEqual(self.exercises.model_1_precision, self.model_1_precision)

	def test_model_1_recall(self):
		self.assertEqual(self.exercises.model_1_recall, self.model_1_recall)

	def test_model_2_tp(self):
		self.assertEqual(self.exercises.model_2_tp, self.model_2_tp)

	def test_model_2_fp(self):
		self.assertEqual(self.exercises.model_2_fp, self.model_2_fp)

	def test_model_2_fn(self):
		self.assertEqual(self.exercises.model_2_fn, self.model_2_fn)

	def test_model_2_precision(self):
		self.assertEqual(self.exercises.model_2_precision, self.model_2_precision)

	def test_model_2_recall(self):
		self.assertEqual(self.exercises.model_2_recall, self.model_2_recall)


if __name__ == '__main__':
	unittest.main()
