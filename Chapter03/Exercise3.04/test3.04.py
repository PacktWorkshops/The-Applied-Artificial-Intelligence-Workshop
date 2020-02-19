import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics.pairwise import euclidean_distances


class Test(unittest.TestCase):
    def setUp(self):
        import Exercise3_04
        self.exercises = Exercise3_04

        self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/german_prepared.csv'
        self.df = pd.read_csv(self.file_url)

        self.scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        self.scaled_credit = self.scaler.fit_transform(self.df)
        self.label = self.scaled_credit[:, 0]
        self.features = self.scaled_credit[:, 1:]
        self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(self.features,
                                                                                                      self.label,
                                                                                                      test_size=0.2,
                                                                                                      random_state=7)
        self.classifier = neighbors.KNeighborsClassifier()
        self.classifier.fit(self.features_train, self.label_train)
        self.acc_train = self.classifier.score(self.features_train, self.label_train)
        self.acc_test = self.classifier.score(self.features_test, self.label_test)

    def test_df(self):
        pd_testing.assert_frame_equal(self.exercises.df, self.df)

    def test_file_url(self):
        self.assertEqual(self.exercises.file_url, self.file_url)

    def test_features_train(self):
        np_testing.assert_array_equal(self.exercises.features_train, self.features_train)

    def test_features_test(self):
        np_testing.assert_array_equal(self.exercises.features_test, self.features_test)

    def test_label_train(self):
        np_testing.assert_array_equal(self.exercises.label_train, self.label_train)

    def test_label_test(self):
        np_testing.assert_array_equal(self.exercises.label_test, self.label_test)

    def test_scaled_credit(self):
        np_testing.assert_array_equal(self.exercises.scaled_credit, self.scaled_credit)

    def test_label(self):
        np_testing.assert_array_equal(self.exercises.label, self.label)

    def test_features(self):
        np_testing.assert_array_equal(self.exercises.features, self.features)

    def test_acc_train(self):
        self.assertEqual(self.exercises.acc_train, self.acc_train)

    def test_acc_test(self):
        self.assertEqual(self.exercises.acc_test, self.acc_test)

if __name__ == '__main__':
    unittest.main()
