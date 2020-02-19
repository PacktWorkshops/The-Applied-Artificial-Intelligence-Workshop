import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm

class Test(unittest.TestCase):
	def setUp(self):
		import Activity3_02
		self.exercises = Activity3_02

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/german_prepared.csv'
		self.df = pd.read_csv(self.file_url)

		self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
		self.scaled_credit = self.scaler.fit_transform(self.df)
		self.label = self.scaled_credit[:, 0]
		self.features = self.scaled_credit[:, 1:]
		self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(self.features, self.label, test_size=0.2, random_state=7)

		def fit_svm(features_train, label_train, features_test, label_test, kernel="linear", C=1, degree=3, gamma='scale'):
			classifier = svm.SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
			classifier.fit(features_train, label_train)
			return classifier.score(features_train, label_train), classifier.score(features_test, label_test)

		self.acc_train_1, self.acc_test_1 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test)
		self.acc_train_2, self.acc_test_2 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="poly",  C=1, degree=4, gamma=0.05)
		self.acc_train_3, self.acc_test_3 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="poly",  C=2, degree=4, gamma=0.05)
		self.acc_train_4, self.acc_test_4 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="poly",  C=1, degree=4, gamma=0.25)
		self.acc_train_5, self.acc_test_5 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="poly",  C=1, degree=4, gamma=0.5)
		self.acc_train_6, self.acc_test_6 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="poly",  C=1, degree=4, gamma=0.16)
		self.acc_train_7, self.acc_test_7 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="sigmoid")
		self.acc_train_8, self.acc_test_8 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="rbf", gamma=0.15)
		self.acc_train_9, self.acc_test_9 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="rbf", gamma=0.25)
		self.acc_train_10, self.acc_test_10 = fit_svm(self.features_train, self.label_train, self.features_test, self.label_test, kernel="rbf", gamma=0.35)

	def test_features_train(self):
		np_testing.assert_array_equal(self.exercises.features_train, self.features_train)

	def test_features_test(self):
		np_testing.assert_array_equal(self.exercises.features_test, self.features_test)

	def test_label_train(self):
		np_testing.assert_array_equal(self.exercises.label_train, self.label_train)

	def test_label_test(self):
		np_testing.assert_array_equal(self.exercises.label_test, self.label_test)

	def test_acc_1(self):
		self.assertEqual(self.exercises.acc_train_1, self.acc_train_1)
		self.assertEqual(self.exercises.acc_test_1, self.acc_test_1)

	def test_acc_2(self):
		self.assertEqual(self.exercises.acc_train_2, self.acc_train_2)
		self.assertEqual(self.exercises.acc_test_2, self.acc_test_2)

	def test_acc_3(self):
		self.assertEqual(self.exercises.acc_train_3, self.acc_train_3)
		self.assertEqual(self.exercises.acc_test_3, self.acc_test_3)

	def test_acc_4(self):
		self.assertEqual(self.exercises.acc_train_4, self.acc_train_4)
		self.assertEqual(self.exercises.acc_test_4, self.acc_test_4)

	def test_acc_5(self):
		self.assertEqual(self.exercises.acc_train_5, self.acc_train_5)
		self.assertEqual(self.exercises.acc_test_5, self.acc_test_5)

	def test_acc_6(self):
		self.assertEqual(self.exercises.acc_train_6, self.acc_train_6)
		self.assertEqual(self.exercises.acc_test_6, self.acc_test_6)

	def test_acc_7(self):
		self.assertEqual(self.exercises.acc_train_7, self.acc_train_7)
		self.assertEqual(self.exercises.acc_test_7, self.acc_test_7)

	def test_acc_8(self):
		self.assertEqual(self.exercises.acc_train_8, self.acc_train_8)
		self.assertEqual(self.exercises.acc_test_8, self.acc_test_8)

	def test_acc_9(self):
		self.assertEqual(self.exercises.acc_train_9, self.acc_train_9)
		self.assertEqual(self.exercises.acc_test_9, self.acc_test_9)

	def test_acc_10(self):
		self.assertEqual(self.exercises.acc_train_10, self.acc_train_10)
		self.assertEqual(self.exercises.acc_test_10, self.acc_test_10)


if __name__ == '__main__':
	unittest.main()
