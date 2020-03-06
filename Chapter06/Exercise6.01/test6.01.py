import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import tensorflow as tf


class Test(unittest.TestCase):
	def setUp(self):
		import Exercise6_01
		self.exercises = Exercise6_01

		self.W = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[1, 6])
		self.X = tf.constant([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[6, 1])
		self.b = tf.constant(-88.0)
		self.mult = tf.matmul(self.W, self.X)
		self.Z = self.mult + self.b
		self.a = tf.math.sigmoid(self.Z)

	def test_W(self):
		np_testing.assert_array_equal(self.exercises.W, self.W)

	def test_X(self):
		np_testing.assert_array_equal(self.exercises.X, self.X)

	def test_b(self):
		np_testing.assert_array_equal(self.exercises.b, self.b)

	def test_mult(self):
		np_testing.assert_array_equal(self.exercises.mult, self.mult)

	def test_Z(self):
		np_testing.assert_array_equal(self.exercises.Z, self.Z)

	def test_a(self):
		np_testing.assert_array_equal(self.exercises.a, self.a)


if __name__ == '__main__':
	unittest.main()
