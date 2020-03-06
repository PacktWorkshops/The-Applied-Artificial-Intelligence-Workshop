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
		import Activity6_01
		self.exercises = Activity6_01

		import tensorflow.keras.datasets.mnist as mnist

		(self.features_train, self.label_train), (self.features_test, self.label_test) = mnist.load_data()

		self.features_train = self.features_train / 255.0
		self.features_test = self.features_test / 255.0

		import numpy as np
		import tensorflow as tf
		from tensorflow.keras import layers

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		input_layer = layers.Flatten(input_shape=(28, 28))
		layer1 = layers.Dense(128, activation='relu')
		final_layer = layers.Dense(10, activation='softmax')

		self.model.add(input_layer)
		self.model.add(layer1)
		self.model.add(layers.Dropout(0.25))
		self.model.add(final_layer)

		optimizer = optimizer = tf.keras.optimizers.Adam(0.001)
		self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
		#self.model.fit(self.features_train, self.label_train, epochs=10, validation_split = 0.2, callbacks=[callback], verbose=2)

	def test_features_train(self):
		np_testing.assert_array_equal(self.exercises.features_train, self.features_train)

	def test_features_test(self):
		np_testing.assert_array_equal(self.exercises.features_test, self.features_test)

	def test_label_train(self):
		np_testing.assert_array_equal(self.exercises.label_train, self.label_train)

	def test_label_test(self):
		np_testing.assert_array_equal(self.exercises.label_test, self.label_test)

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())


if __name__ == '__main__':
	unittest.main()
