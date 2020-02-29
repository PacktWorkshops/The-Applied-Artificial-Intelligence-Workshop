import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures



class Test(unittest.TestCase):
    def setUp(self):
        import Activity2_01
        self.exercises = Activity2_01

        self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Applied-Artificial-Intelligence-Workshop/master/Datasets/boston_house_price.csv'
        self.df = pd.read_csv(self.file_url)
        self.features = np.array(self.df.drop('MEDV', 1))
        self.label = np.array(self.df['MEDV'])
        self.scaled_features = preprocessing.scale(self.features)

        self.poly_1_scaled_features = PolynomialFeatures(degree=1).fit_transform(self.scaled_features)
        self.poly_2_scaled_features = PolynomialFeatures(degree=2).fit_transform(self.scaled_features)
        self.poly_3_scaled_features = PolynomialFeatures(degree=3).fit_transform(self.scaled_features)

        (self.poly_1_features_train, self.poly_1_features_test, self.poly_label_train,self.poly_label_test) = model_selection.train_test_split(self.poly_1_scaled_features, self.label, test_size=0.1,random_state=8)
        (self.poly_2_features_train, self.poly_2_features_test, self.poly_label_train,self.poly_label_test2) = model_selection.train_test_split(self.poly_2_scaled_features, self.label, test_size=0.1,random_state=8)
        (self.poly_3_features_train, self.poly_3_features_test, self.poly_label_train,self.poly_label_test) = model_selection.train_test_split(self.poly_3_scaled_features, self.label, test_size=0.1,random_state=8)

        self.model_1 = linear_model.LinearRegression()
        self.model_1.fit(self.poly_1_features_train, self.poly_label_train)
        self.model_1_score_train = self.model_1.score(self.poly_1_features_train, self.poly_label_train)
        self.model_1_score_test = self.model_1.score(self.poly_1_features_test, self.poly_label_test)

        self.model_2 = linear_model.LinearRegression()
        self.model_2.fit(self.poly_2_features_train, self.poly_label_train)
        self. model_2_score_train = self.model_2.score(self.poly_2_features_train, self.poly_label_train)
        self.model_2_score_test = self.model_2.score(self.poly_2_features_test, self.poly_label_test)

        self.model_3 = linear_model.LinearRegression()
        self.model_3.fit(self.poly_3_features_train, self.poly_label_train)
        self.model_3_score_train = self.model_3.score(self.poly_3_features_train, self.poly_label_train)
        self.model_3_score_test = self.model_3.score(self.poly_3_features_test, self.poly_label_test)

        self.model_1_prediction = self.model_1.predict(self.poly_1_features_test)
        self.model_2_prediction = self.model_2.predict(self.poly_2_features_test)
        self.model_3_prediction = self.model_3.predict(self.poly_3_features_test)

        self.df_prediction = pd.DataFrame(self.poly_label_test)
        self.df_prediction.rename(columns={0: 'label'}, inplace=True)
        self.df_prediction['model_1_prediction'] = pd.DataFrame(self.model_1_prediction)
        self.df_prediction['model_2_prediction'] = pd.DataFrame(self.model_2_prediction)
        self.df_prediction['model_3_prediction'] = pd.DataFrame(self.model_3_prediction)

    def test_data_prep(self):
        np_testing.assert_array_equal(self.exercises.features, self.features)
        np_testing.assert_array_equal(self.exercises.label, self.label)
        np_testing.assert_array_equal(self.exercises.scaled_features, self.scaled_features)

    def test_poly_scale(self):
        np_testing.assert_array_equal(self.exercises.poly_1_scaled_features, self.poly_1_scaled_features)
        np_testing.assert_array_equal(self.exercises.poly_2_scaled_features, self.poly_2_scaled_features)
        np_testing.assert_array_equal(self.exercises.poly_3_scaled_features, self.poly_3_scaled_features)

    def test_data_split_1(self):
        np_testing.assert_array_equal(self.exercises.poly_1_features_train, self.poly_1_features_train)
        np_testing.assert_array_equal(self.exercises.poly_1_features_test, self.poly_1_features_test)
        np_testing.assert_array_equal(self.exercises.poly_label_train, self.poly_label_train)
        np_testing.assert_array_equal(self.exercises.poly_label_test, self.poly_label_test)

    def test_data_split_2(self):
        np_testing.assert_array_equal(self.exercises.poly_2_features_train, self.poly_2_features_train)
        np_testing.assert_array_equal(self.exercises.poly_2_features_test, self.poly_2_features_test)
        np_testing.assert_array_equal(self.exercises.poly_label_train, self.poly_label_train)
        np_testing.assert_array_equal(self.exercises.poly_label_test, self.poly_label_test)

    def test_data_split_3(self):
        np_testing.assert_array_equal(self.exercises.poly_3_features_train, self.poly_3_features_train)
        np_testing.assert_array_equal(self.exercises.poly_3_features_test, self.poly_3_features_test)
        np_testing.assert_array_equal(self.exercises.poly_label_train, self.poly_label_train)
        np_testing.assert_array_equal(self.exercises.poly_label_test, self.poly_label_test)

    def test_model_1(self):
        np_testing.assert_array_equal(self.exercises.model_1_score_train, self.model_1_score_train)
        np_testing.assert_array_equal(self.exercises.model_1_score_test, self.model_1_score_test)

    def test_model_2(self):
        np_testing.assert_array_equal(self.exercises.model_2_score_train, self.model_2_score_train)
        np_testing.assert_array_equal(self.exercises.model_2_score_test, self.model_2_score_test)

    def test_model_3(self):
        np_testing.assert_array_equal(self.exercises.model_3_score_train, self.model_3_score_train)
        np_testing.assert_array_equal(self.exercises.model_3_score_test, self.model_3_score_test)

    def test_prediction(self):
        np_testing.assert_array_equal(self.exercises.model_1_prediction, self.model_1_prediction)
        np_testing.assert_array_equal(self.exercises.model_2_prediction, self.model_2_prediction)
        np_testing.assert_array_equal(self.exercises.model_3_prediction, self.model_3_prediction)

if __name__ == '__main__':
    unittest.main()