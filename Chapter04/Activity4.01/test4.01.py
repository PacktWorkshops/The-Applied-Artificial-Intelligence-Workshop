import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing


class Test(unittest.TestCase):
	def setUp(self):
		import Activity4_01
		self.exercises = Activity4_01

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/car.csv'
		self.df = pd.read_csv(self.file_url)

		from sklearn import preprocessing
		def encode(data_frame, column):
			label_encoder = preprocessing.LabelEncoder()
			label_encoder.fit(data_frame[column].unique())
			return label_encoder.transform(data_frame[column])

		for column in self.df.columns:
			self.df[column] = encode(self.df, column)

		self.label = self.df.pop('class')
		from sklearn import model_selection
		self.features_train, self.features_test, self.label_train, self.label_test = model_selection.train_test_split(self.df, self.label, test_size=0.1, random_state=88)
		from sklearn.tree import DecisionTreeClassifier
		decision_tree = DecisionTreeClassifier()
		decision_tree.fit(self.features_train, self.label_train)
		decision_tree.score(self.features_test, self.label_test)

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


if __name__ == '__main__':
	unittest.main()
