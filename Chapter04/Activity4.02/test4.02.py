import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing


class Test(unittest.TestCase):
	def setUp(self):
		import Activity4_02
		self.exercises = Activity4_02

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

		from sklearn.ensemble import RandomForestClassifier
		random_forest_classifier = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=168)
		random_forest_classifier.fit(self.features_train, self.label_train)
		self.rf_preds_test = random_forest_classifier.predict(self.features_test)
		self.rf_varimp = random_forest_classifier.feature_importances_
		from sklearn.ensemble import ExtraTreesClassifier
		extra_trees_classifier = ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=168)
		extra_trees_classifier.fit(self.features_train, self.label_train)
		self.et_preds_test = extra_trees_classifier.predict(self.features_test)
		self.et_varimp = extra_trees_classifier.feature_importances_

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

	def test_rf_preds_test(self):
		np_testing.assert_array_equal(self.exercises.rf_preds_test, self.rf_preds_test)

	def test_rf_varimp(self):
		np_testing.assert_array_equal(self.exercises.rf_varimp, self.rf_varimp)

	def test_et_preds_test(self):
		np_testing.assert_array_equal(self.exercises.et_preds_test, self.et_preds_test)

	def test_et_varimp(self):
		np_testing.assert_array_equal(self.exercises.et_varimp, self.et_varimp)

if __name__ == '__main__':
	unittest.main()
