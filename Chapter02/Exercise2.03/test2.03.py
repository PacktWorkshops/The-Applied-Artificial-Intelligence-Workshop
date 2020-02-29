import quandl
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
data_frame = quandl.get("YALE/SPCOMP", start_date="1950-01-01", end_date="2019-12-31")
data_frame.head()

import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import quandl



class Test(unittest.TestCase):
    def setUp(self):
        import Exercise2_03
        self.exercises = Exercise2_03

        self.data_frame = quandl.get("YALE/SPCOMP", start_date="1950-01-01", end_date="2019-12-31")

        self.data_frame = self.data_frame[['Long Interest Rate', 'Real Price', 'Real Dividend']]
        self.data_frame.fillna(method='ffill', inplace=True)
        self.data_frame['Real Price Label'] = self.data_frame['Real Price'].shift(-3)
        self.features = np.array(self.data_frame.drop('Real Price Label', 1))
        self.scaled_features = preprocessing.scale(self.features)
        self.scaled_features_latest_3 = self.scaled_features[-3:]
        self.scaled_features = self.scaled_features[:-3]
        self.data_frame.dropna(inplace=True)
        self.label = np.array(self.data_frame['Real Price Label'])
        (self.features_train, self.features_test, self.label_train, self.label_test) = model_selection.train_test_split(self.scaled_features, self.label,test_size=0.1,random_state=8)

    def test_features(self):
        np_testing.assert_array_equal(self.exercises.features, self.features)

    def test_label(self):
        np_testing.assert_array_equal(self.exercises.label, self.label)

    def test_label(self):
        np_testing.assert_array_equal(self.exercises.scaled_features, self.scaled_features)

    def test_split(self):
        np_testing.assert_array_equal(self.exercises.features_train, self.features_train)
        np_testing.assert_array_equal(self.exercises.features_test, self.features_test)
        np_testing.assert_array_equal(self.exercises.label_train, self.label_train)
        np_testing.assert_array_equal(self.exercises.label_test, self.label_test)


if __name__ == '__main__':
    unittest.main()