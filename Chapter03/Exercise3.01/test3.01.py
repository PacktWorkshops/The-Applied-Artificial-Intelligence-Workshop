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
        import Exercise3_02
        self.exercises = Exercise3_02

        self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/german_credit.csv'
        self.df = pd.read_csv(self.file_url)

    def test_df(self):
        pd_testing.assert_frame_equal(self.exercises.df, self.df)

    def test_file_url(self):
        self.assertEqual(self.exercises.file_url, self.file_url)

if __name__ == '__main__':
    unittest.main()
