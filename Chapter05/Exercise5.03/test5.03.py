import unittest
import import_ipynb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import numpy.testing as np_testing
from scipy.spatial import distance

class Test(unittest.TestCase):
    def setUp(self):
        import Exercise5_03
        self.exercises = Exercise5_03

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

        self.P1 = [1, 1]
        self.r = 2

        self.points1 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P1) <= self.r])
        self.P2 = [np.mean(self.points1.transpose()[0]),np.mean(self.points1.transpose()[1])]
        self.points2 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P2) <= self.r])
        self.P3 = [8, 1]
        self.points3 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P3) <= self.r])
        self.P4 = [np.mean(self.points3.transpose()[0]),np.mean(self.points3.transpose()[1])]
        self.P5 = [8, 0]
        self.points4 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P5) <= self.r])
        self.P6 = [np.mean(self.points4.transpose()[0]),np.mean(self.points4.transpose()[1])]
        self.P7 = [8.5, 1]
        self.points5 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P7) <= self.r])
        self.P8 = [6, 1]
        self.points6 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P8) <= self.r])
        self.P9 = [np.mean(self.points6.transpose()[0]), np.mean(self.points6.transpose()[1])]
        self.points7 = np.array([p0 for p0 in self.data_points if distance.euclidean(p0, self.P9) <= self.r])

    def test_dataset(self):
        np_testing.assert_array_equal(self.exercises.data_points, self.data_points)
        self.assertEqual(self.exercises.r, self.r)

    def test_prediction(self):
        np_testing.assert_array_equal(self.exercises.P1, self.P1)
        np_testing.assert_array_equal(self.exercises.P2, self.P2)
        np_testing.assert_array_equal(self.exercises.P3, self.P3)
        np_testing.assert_array_equal(self.exercises.P4, self.P4)
        np_testing.assert_array_equal(self.exercises.P5, self.P5)
        np_testing.assert_array_equal(self.exercises.P6, self.P6)
        np_testing.assert_array_equal(self.exercises.P7, self.P7)
        np_testing.assert_array_equal(self.exercises.P8, self.P8)
        np_testing.assert_array_equal(self.exercises.P9, self.P9)
        np_testing.assert_array_equal(self.exercises.points1, self.points1)
        np_testing.assert_array_equal(self.exercises.points2, self.points2)
        np_testing.assert_array_equal(self.exercises.points3, self.points3)
        np_testing.assert_array_equal(self.exercises.points4, self.points4)
        np_testing.assert_array_equal(self.exercises.points5, self.points5)
        np_testing.assert_array_equal(self.exercises.points6, self.points6)
        np_testing.assert_array_equal(self.exercises.points7, self.points7)

if __name__ == '__main__':
	unittest.main()