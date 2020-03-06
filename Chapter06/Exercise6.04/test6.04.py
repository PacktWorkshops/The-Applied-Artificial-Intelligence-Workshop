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
		import Exercise6_04
		self.exercises = Exercise6_04

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/boston_house_price.csv'
		self.df = pd.read_csv(self.file_url)
		self.label = self.df.pop('MEDV')

		from sklearn.preprocessing import scale

		self.scaled_features = scale(self.df)

		from sklearn.model_selection import train_test_split

		self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(self.scaled_features, self.label, test_size=0.1, random_state=8)

		import numpy as np
		import tensorflow as tf
		from tensorflow.keras import layers

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()
		regularizer = tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)
		layer1 = layers.Dense(10, activation='relu', input_shape=[12], kernel_regularizer=regularizer)
		final_layer = layers.Dense(1)
		self.model.add(layer1)
		self.model.add(layers.Dropout(0.25))
		self.model.add(final_layer)

		optimizer = tf.keras.optimizers.SGD(0.001)
		self.model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
		#self.model.fit(self.features_train, self.label_train, epochs=50, validation_split = 0.2, callbacks=[callback], verbose=2)

	def test_file_url(self):
		self.assertEqual(self.exercises.file_url, self.file_url)

	def test_df(self):
		pd_testing.assert_frame_equal(self.exercises.df, self.df)

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
