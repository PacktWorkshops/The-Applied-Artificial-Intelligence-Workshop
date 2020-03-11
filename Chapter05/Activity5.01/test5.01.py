import unittest
import import_ipynb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import numpy.testing as np_testing

class Test(unittest.TestCase):
    def setUp(self):
        import Activity5_01
        self.exercises = Activity5_01

        self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/Sales_Transactions_Dataset_Weekly.csv'
        self.df = pd.read_csv(self.file_url)

        self.df2 = self.df.drop(self.df.iloc[:, 0:55], inplace=False, axis=1)
        self.k_means_model = KMeans(n_clusters=8, random_state=8)
        self.k_means_model.fit(self.df2)
        self.labels = self.k_means_model.labels_

        self.df.drop(self.df.iloc[:, 53:], inplace=True, axis=1)
        self.df.drop('Product_Code', inplace=True, axis=1)
        self.df['label'] = self.labels
        self.df_agg = self.df.groupby('label').sum()
        self.df_final = self.df[['label', 'W0']].groupby('label').count()
        self.df_final = self.df_final.rename(columns={'W0': 'count_product'})
        self.df_final['total_sales'] = self.df_agg.sum(axis=1)
        self.df_final['yearly_average_sales'] = self.df_final['total_sales'] / self.df_final['count_product']
        self.df_final.sort_values(by='yearly_average_sales', ascending=False, inplace=True)

    def test_dataset(self):
        np_testing.assert_array_equal(self.exercises.df, self.df)
        np_testing.assert_array_equal(self.exercises.df2, self.df2)

    def test_prediction(self):
        np_testing.assert_array_equal(self.exercises.labels, self.labels)

    def test_aggregation(self):
        np_testing.assert_array_equal(self.exercises.labels, self.labels)
        np_testing.assert_array_equal(self.exercises.df_final, self.df_final)

if __name__ == '__main__':
	unittest.main()