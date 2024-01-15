import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.finance_ml.outliers.outliers import DataOutliers


class TestDataOutliers(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        np.random.seed(0)
        data = {
            'TSLA_VOLUME': np.random.randn(100),
            'AAPL_VOLUME': np.random.randn(100),
            'outliers': np.random.choice([True, False], 100)
        }
        self.df = pd.DataFrame(data)
        self.data_outliers = DataOutliers(self.df)

    def test_calculate_mean_value(self):
        mean_value = self.data_outliers.calculate_mean_value('TSLA_VOLUME')
        self.assertIsInstance(mean_value, float)
        self.assertAlmostEqual(mean_value, self.df['TSLA_VOLUME'].mean())

    def test_calculate_standard_deviation(self):
        std_deviation = self.data_outliers.calculate_standard_deviation('TSLA_VOLUME')
        self.assertIsInstance(std_deviation, float)
        self.assertAlmostEqual(std_deviation, self.df['TSLA_VOLUME'].std())

    def test_calculate_zscore(self):
        z_scores = self.data_outliers.calculate_zscore('TSLA_VOLUME')
        self.assertIsInstance(z_scores, pd.Series)
        self.assertEqual(len(z_scores), len(self.df))

    def test_hist_plot_std_dev(self):
        mean_value = self.data_outliers.calculate_mean_value('TSLA_VOLUME')
        std_deviation = self.data_outliers.calculate_standard_deviation('TSLA_VOLUME')
        with self.assertRaises(Exception):  
            self.data_outliers.hist_plot_std_dev(mean_value, std_deviation, 'INVALID_COLUMN_NAME')

    def test_hist_plot(self):
        outlier_indices = self.df[self.df['outliers']].index
        with self.assertRaises(Exception):  
            self.data_outliers.hist_plot('INVALID_COLUMN_NAME', outlier_indices)

    def test_scatter_plot(self):
        with self.assertRaises(Exception):  
            self.data_outliers.scatter_plot('INVALID_COLUMN_NAME')

    def test_remove_outliers(self):
        cleaned_data = self.data_outliers.remove_outliers(['TSLA_VOLUME'])
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertTrue('TSLA_VOLUME' in cleaned_data.columns)

    def test_detect_outliers(self):
        outliers = self.data_outliers.detect_outliers('TSLA_VOLUME')
        self.assertIsInstance(outliers, pd.Series)
        self.assertEqual(len(outliers), len(self.df))

    def test_create_outliers_column(self):
        outliers_column = self.data_outliers.create_outliers_column('TSLA_VOLUME')
        self.assertIsInstance(outliers_column, pd.DataFrame)
        self.assertTrue('outliers' in outliers_column.columns)

    def test_visualize_outliers(self):
        with self.assertRaises(Exception):  
            self.data_outliers.visualize_outliers('INVALID_COLUMN_NAME')

    def test_calculate_outliers_using_std(self):
        outliers = self.data_outliers.calculate_outliers_using_std('TSLA_VOLUME')
        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertTrue('TSLA_VOLUME' in outliers.columns)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
