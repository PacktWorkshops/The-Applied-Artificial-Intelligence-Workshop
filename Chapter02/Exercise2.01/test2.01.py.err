import unittest
import import_ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


class Test(unittest.TestCase):
    def setUp(self):
        import Exercise2_01
        self.exercises = Exercise2_01

        self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/population.csv'
        self.df = pd.read_csv(self.file_url)

        self.x = np.array(range(1, 19))
        self.y = np.array(self.df['population'])
        self.a, self.b = np.polyfit(self.x, self.y, 1)

        self.population_2025 = 25*self.a+self.b
        self.population_2030 = 30*self.a+self.b

    def test_polyfit(self):
        self.assertEqual(self.exercises.a, self.a)
        self.assertEqual(self.exercises.b, self.b)

    def test_prediction(self):
        self.assertEqual(self.exercises.population_2025, self.population_2025)
        self.assertEqual(self.exercises.population_2030, self.population_2030)

if __name__ == '__main__':
	unittest.main()
