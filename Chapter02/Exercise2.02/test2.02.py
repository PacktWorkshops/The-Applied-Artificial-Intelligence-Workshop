import unittest
import quandl
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

class Test(unittest.TestCase):
    def setUp(self):
        import Exercise2_02
        self.exercises = Exercise2_02

        self.data_frame = quandl.get("YALE/SPCOMP")
        self.earnings = self.data_frame.at['1871-01-31','Earnings']

    def test_data_frame(self):
        earnings = self.exercises.data_frame.at['1871-01-31','Earnings']
        self.assertEqual(earnings, self.earnings)

if __name__ == '__main__':
    unittest.main()
