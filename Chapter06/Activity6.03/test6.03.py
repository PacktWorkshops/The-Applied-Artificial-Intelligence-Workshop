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
		import Activity6_03
		self.exercises = Activity6_03

		import numpy as np

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/yahoo_spx.csv'
		self.df = pd.read_csv(self.file_url)
		self.stock_data = self.df.iloc[:, 1:2].values

		from sklearn.preprocessing import MinMaxScaler
		sc = MinMaxScaler()
		self.stock_data_scaled = sc.fit_transform(self.stock_data)
		self.X_data = []
		self.y_data = []
		self.window = 30
		for i in range(self.window, len(self.df)):
			self.X_data.append(self.stock_data_scaled[i - self.window:i, 0])
			self.y_data.append(self.stock_data_scaled[i, 0])
		self.X_data = np.array(self.X_data)
		self.y_data = np.array(self.y_data)
		self.X_data = np.reshape(self.X_data, (self.X_data.shape[0], self.X_data.shape[1], 1))
		self.features_train = self.X_data[:1000]
		self.label_train = self.y_data[:1000]
		self.features_test = self.X_data[:1000]
		self.label_test = self.y_data[:1000]

		import numpy as np
		import tensorflow as tf
		from tensorflow.keras import layers

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		lstm_layer1 = layers.LSTM(units=50,return_sequences=True,input_shape=(self.X_data.shape[1], 1))
		lstm_layer2 = layers.LSTM(units=50, return_sequences=True)
		lstm_layer3 = layers.LSTM(units=50, return_sequences=True)
		lstm_layer4 = layers.LSTM(units=50)
		fc_layer = layers.Dense(1)

		self.model.add(lstm_layer1)
		self.model.add(layers.Dropout(0.2))
		self.model.add(lstm_layer2)
		self.model.add(layers.Dropout(0.2))
		self.model.add(lstm_layer3)
		self.model.add(layers.Dropout(0.2))
		self.model.add(lstm_layer4)
		self.model.add(layers.Dropout(0.2))
		self.model.add(fc_layer)

		optimizer = optimizer = tf.keras.optimizers.Adam(0.001)
		self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
		#self.model.fit(features_train, label_train, epochs=10, validation_split = 0.2, verbose=2)
		#self.model.evaluate(features_test, label_test)

	def test_file_url(self):
		self.assertEqual(self.exercises.file_url, self.file_url)

	def test_df(self):
		pd_testing.assert_frame_equal(self.exercises.df, self.df)

	def test_stock_data(self):
		np_testing.assert_array_equal(self.exercises.stock_data, self.stock_data)

	def test_window(self):
		self.assertEqual(self.exercises.window, self.window)

	def test_stock_data_scaled(self):
		np_testing.assert_array_equal(self.exercises.stock_data_scaled, self.stock_data_scaled)

	def test_X_data(self):
		np_testing.assert_array_equal(self.exercises.X_data, self.X_data)

	def test_y_data(self):
		np_testing.assert_array_equal(self.exercises.y_data, self.y_data)

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
