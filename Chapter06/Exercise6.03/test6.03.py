import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import tensorflow as tf
from numpy import loadtxt


class Test(unittest.TestCase):
	def setUp(self):
		import Exercise6_03
		self.exercises = Exercise6_03

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/german_scaled.csv'
		self.data = loadtxt(self.file_url, delimiter=',')
		self.features = self.data[:, 1:]
		self.label = self.data[:, 0]

		from sklearn.model_selection import train_test_split

		self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(self.features, self.label, test_size=0.2,
																				  random_state=7)
		import numpy as np
		import tensorflow as tf
		from tensorflow.keras import layers

		np.random.seed(1)
		tf.random.set_seed(1)

		self.model = tf.keras.Sequential()
		layer1 = layers.Dense(16, activation='relu', input_shape=[19])
		final_layer = layers.Dense(1, activation='sigmoid')
		self.model.add(layer1)
		self.model.add(final_layer)
		optimizer = tf.keras.optimizers.Adam(0.001)
		self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		#self.model.fit(self.features_train, self.label_train, epochs=10)

	def test_file_url(self):
		self.assertEqual(self.exercises.file_url, self.file_url)

	def test_data(self):
		np_testing.assert_array_equal(self.exercises.data, self.data)

	def test_features_train(self):
		np_testing.assert_array_equal(self.exercises.features_train, self.features_train)

	def test_features_test(self):
		np_testing.assert_array_equal(self.exercises.features_test, self.features_test)

	def test_label_train(self):
		np_testing.assert_array_equal(self.exercises.label_train, self.label_train)

	def test_label_test(self):
		np_testing.assert_array_equal(self.exercises.label_test, self.label_test)

	def test_label(self):
		np_testing.assert_array_equal(self.exercises.label, self.label)

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())


if __name__ == '__main__':
	unittest.main()
