"""
Created on Friday, December 01 14:00:00 2023
@Group:
    Vinaybhai Bakulbhai Chavada (cy62koco)
    Pansuriya, Yagnesh Arvindbhai (we85jabo)
    Disha Vinesh Jethva (it19efiw)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


class DataOutliers:
    """
    ----------------------------------------------------------------------------------
    Class to detect outliers in a dataset.
    ----------------------------------------------------------------------------------
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Outliers class with data.
        :param data: Pandas DataFrame containing the data.
        """
        self.data = data
        self.upperBound = None
        self.lowerBound = None

    """
    ----------------------------------------------------------------------------------
    Basic statistics
    ----------------------------------------------------------------------------------
    """

    def calculate_mean_value(self, column_name: str) -> float:
        """
        Calculate the mean value of a specified column.
        :param column_name: Name of the column.
        :return: Mean value of the specified column.
        """
        mean = self.data[column_name].mean()
        return mean

    def calculate_standard_deviation(self, column_name: str) -> float:
        """
        Calculate the standard deviation of a specified column.
        :param column_name: Name of the column.
        :return: Standard deviation of the specified column.
        """
        std_dev = self.data[column_name].std()
        return std_dev

    """
    ----------------------------------------------------------------------------------
    Outliers detection using Standard Deviation
    ----------------------------------------------------------------------------------
    """

    def calculate_outliers_using_std(self, column_name: str, threshold: int = 3) -> pd.DataFrame:
        """
        Detect outliers in a column using Standard Deviation.
        :param column_name: Name of the column.
        :param threshold: Number of standard deviations to consider as an outlier.
        :return: DataFrame containing outliers.
        """
        mean_value = self.calculate_mean_value(column_name)
        standard_deviation = self.calculate_standard_deviation(column_name)
        outliers = self.data[self.data[column_name].abs() -
                             mean_value >
                             threshold * standard_deviation]
        self.hist_plot_std_dev(mean_value, standard_deviation, column_name)
        return outliers

    def hist_plot_std_dev(self, mean_value: float, std_deviation: float, column_name: str) -> None:
        """
        Visualize outliers using a histogram based on Standard Deviation.
        :param mean_value: Mean value of the column.
        :param std_deviation: Standard deviation of the column.
        :param column_name: Name of the column.
        """

        # Define outlier threshold (e.g., 2 standard deviations away from the mean)
        threshold = 2 * std_deviation

        # Identify outliers
        outliers = self.data[
            (self.data[column_name] < mean_value - threshold) | (self.data[column_name] > mean_value + threshold)]

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the data points
        plt.scatter(self.data.index, self.data[column_name], color='blue', label='Data Points')

        # Highlight outliers
        plt.scatter(outliers.index, outliers[column_name], color='red', label='Outliers')

        # Add mean and threshold lines for visualization
        plt.axhline(mean_value, color='green', linestyle='dashed', linewidth=1, label=f'Mean = {mean_value:.2f}')
        plt.axhline(mean_value - threshold, color='orange', linestyle='dashed', linewidth=1,
                    label=f'-2 Std Dev = {mean_value - threshold:.2f}')
        plt.axhline(mean_value + threshold, color='orange', linestyle='dashed', linewidth=1,
                    label=f'+2 Std Dev = {mean_value + threshold:.2f}')

        # Add labels and title
        plt.title(f'Visualizing Outliers in {column_name} Using Standard Deviation')
        plt.xlabel('Index')
        plt.ylabel('Value')

        plt.legend()
        plt.show()

    """
    ----------------------------------------------------------------------------------
    Outliers detection using z-score
    ----------------------------------------------------------------------------------
    """

    def calculate_zscore(self, column_name: str) -> pd.Series:
        """
        Calculate Z-scores for a specific column in the dataset.
        :param column_name: Name of the column for which Z-scores need to be calculated.
        :return: Pandas Series containing Z-scores.
        """
        mean = self.calculate_mean_value(column_name)
        std_dev = self.calculate_standard_deviation(column_name)
        z_scores = (self.data[column_name] - mean) / std_dev
        return z_scores

    def hist_plot(self, column_name: str, outlier_indices: list) -> None:
        """
        Detect and remove outliers based on Z-score for a specific column in a DataFrame.

        :param data: Pandas DataFrame containing the data.
        :param column_name: Name of the column for which outliers need to be detected and removed.
        :param threshold: Z-score threshold for outlier detection.
        :return: DataFrame with outliers removed.
        """

        # Get indices of outliers
        # Visualize the data
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data[column_name], kde=True, color='blue')
        plt.title(f'Distribution of {column_name} with Z-score Outliers Highlighted')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.axvline(self.data[column_name].iloc[outlier_indices].min(), color='red', linestyle='--', label='Outliers')
        plt.axvline(self.data[column_name].iloc[outlier_indices].max(), color='red', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.show()

    def scatter_plot(self, column_name: str) -> None:
        """
        Create a scatter plot for a specified column.
        :param column_name: Name of the column.
        """
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='TSLA_VOLUME', y=self.data.index)
        plt.title(f'Scatter plot for Column ${column_name}')
        plt.show()

    # Function to remove outliers based on Z-score
    def remove_outliers(self, columns: list, threshold: int = 3) -> pd.DataFrame:
        """
        Remove outliers from specified columns using Z-score.
        :param columns: List of columns to remove outliers from.
        :param threshold: Z-score threshold for outlier removal.
        :return: DataFrame with outliers removed.
        """
        for col in columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            # self.data = self.data[z_scores < threshold]
            outliers = z_scores >= threshold
            self.data.loc[outliers, col] = self.data[col].mean()
        return self.data

    """
    ----------------------------------------------------------------------------------
    Outliers detection using IQR (InterQuartileRange)
    ----------------------------------------------------------------------------------
    """

    def detect_outliers(self, column_name: str) -> pd.Series:
        """
        Detect outliers in a column using InterQuartile Range (IQR).
        :param column_name: Name of the column.
        :return: Series of True/False indicating outliers.
        """
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        self.lowerBound = lower_bound
        upper_bound = Q3 + 1.5 * IQR
        self.upperBound = upper_bound
        outliers = (self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound)
        return outliers

    def create_outliers_column(self, column_name: str) -> pd.DataFrame:
        """
        Create a new column indicating outliers based on IQR.
        :param column_name: Name of the column.
        :return: DataFrame with an added 'outliers' column.
        """
        outliers = self.detect_outliers(column_name)
        self.data.loc[:, 'outliers'] = outliers.values
        return self.data

    def visualize_outliers(self, column_name: str) -> None:
        """
        Visualize outliers in a column using scatter plots.
        :param column_name: Name of the column.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data.index, self.data[column_name], color='blue', label='Data Points')

        outliers_data = self.data[self.data['outliers']]
        plt.scatter(outliers_data.index, outliers_data[column_name], color='red', label='Outliers')

        plt.title(f'Visualizing Outliers in {column_name}')
        plt.xlabel('Index')
        plt.ylabel('Value')

        plt.legend()
        plt.show()

    """
    ----------------------------------------------------------------------------------
    ------------------------------- Getters and Setters ------------------------------
    """

    def get_data(self) -> pd.DataFrame:
        """
        Get the current DataFrame.
        :return: Current DataFrame.
        """
        return self.data

    def set_data(self, data: pd.DataFrame) -> None:
        """
       Set the data for the class.
       :param data: New DataFrame to set.
       """
        self.data = data
