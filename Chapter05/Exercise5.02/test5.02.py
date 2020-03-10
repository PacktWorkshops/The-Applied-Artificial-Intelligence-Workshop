import unittest
import import_ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans
import numpy.testing as np_testing

class Test(unittest.TestCase):
    def setUp(self):
        import Exercise5_02
        self.exercises = Exercise5_02

        self.data_points = np.array([
            [1, 1],
            [1, 1.5],
            [2, 2],
            [8, 1],
            [8, 0],
            [8.5, 1],
            [6, 1],
            [1, 10],
            [1.5, 10],
            [1.5, 9.5],
            [10, 10],
            [1.5, 8.5]])

        self.k_means_model = KMeans(n_clusters=4, random_state=8)
        self.k_means_model.fit(self.data_points)
        self.centers = self.k_means_model.cluster_centers_
        self.labels = self.k_means_model.labels_

    def test_dataset(self):
        np_testing.assert_array_equal(self.exercises.data_points, self.data_points)

    def test_prediction(self):
        np_testing.assert_array_equal(self.exercises.centers, self.centers)
        np_testing.assert_array_equal(self.exercises.labels, self.labels)

if __name__ == '__main__':
	unittest.main()