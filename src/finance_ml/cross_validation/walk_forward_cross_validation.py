"""
WalkForwardCrossValidation Class
--------------------------------
Author: Rabin BK (Matriculation Number: 23272000, IDM ID: yl03oxiq)
Author: Yatin Arora (Matriculation Number: 23014677, IDM ID: an79usew)

This class provides functionality for performing walk-forward cross-validation for time series data.

Usage:
------
1. Create an instance of WalkForwardCrossValidation with the number of folds.
2. Call the cross_validate method to perform cross-validation with a machine learning model and optional metrics.

Example:
--------
# Import necessary libraries
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Create an instance of WalkForwardCrossValidation with 5 folds
cv = WalkForwardCrossValidation(num_folds=5)

# Perform cross-validation with a machine learning model and metrics
results = cv.cross_validate(X, y, model, metrics=[metric1, metric2], plot=True)

# Optionally, visualize the data splits if plot is set to True

"""

from typing import List, Tuple, Callable, Optional, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


class WalkForwardCrossValidation:
    def __init__(self, num_folds: int):
        """
        Initialize the WalkForwardCrossValidation object.

        Parameters:
        -----------
        num_folds (int): The number of folds for cross-validation.
        """
        self.num_folds = num_folds

    def _walk_forward_splits(self, X: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Generate train and test splits for walk-forward cross-validation.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input time series data.

        Yields:
        -------
        train_end, test_start, test_end : int, int, int
            Indices for the training and test data splits.
        """
        num_samples = len(X)
        fold_size = int(num_samples / self.num_folds)

        for fold_index in range(self.num_folds):
            train_end = fold_index * fold_size
            test_start = train_end
            test_end = test_start + fold_size

            if test_end > num_samples:
                test_end = num_samples

            yield train_end, test_start, test_end

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, model: object,
                       metrics: List[Callable] = None, plot: bool = False) -> Dict[str, List[Union[float, int]]]:
        """
        Perform walk-forward cross-validation with a machine learning model.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input time series data.
        y : pandas.Series
            The target values.
        model : object
            The machine learning model with fit and predict methods.
        metrics : list of callable functions, optional
            Evaluation metrics to be computed for each fold.
        plot : bool, optional
            Whether to plot the data splits for each fold.

        Returns:
        --------
        evaluation_results : dict
            A dictionary containing evaluation metric values for each fold.
        """
        self._check_parameters(model, metrics)
        evaluation_results = {metric.__name__: [] for metric in metrics}

        try:
            for train_end, test_start, test_end in self._walk_forward_splits(X):
                X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
                X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

                if X_train.empty or X_test.empty:
                    continue

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                for metric in metrics:
                    metric_value = metric(y_test, predictions)
                    evaluation_results[metric.__name__].append(metric_value)

                if plot:
                    self._plot_data_splits(X, y, X_train, y_train, X_test, y_test)
        except Exception as e:
            print(f"An error occurred during cross-validation: {str(e)}")

        return evaluation_results

    def _plot_data_splits(self, X: pd.DataFrame, y: pd.Series,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot the data splits for visualization.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input time series data.
        y : pandas.Series
            The target values.
        X_train : pandas.DataFrame
            The training data.
        y_train : pandas.Series
            The target values for the training data.
        X_test : pandas.DataFrame
            The test data.
        y_test : pandas.Series
            The target values for the test data.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(X.index, y, label='Data', linestyle='-', color='grey', alpha=0.3)
        plt.fill_between(X_train.index, y_train, color='skyblue', alpha=0.3, label='Training Data')
        plt.fill_between(X_test.index, y_test, color='salmon', alpha=0.3, label='Test Data')

        y_min, y_max = y.min(), y.max()
        plt.ylim(y_min - 0.1 * y_min, y_max + 0.1 * y_max)

        plt.title('Walk-Forward Cross-Validation')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def _check_parameters(self, model: object, metrics: list) -> None:
        """
        Check if the model and metrics parameters are valid.

        Parameters:
        -----------
        model : object
            The machine learning model.
        metrics : list of callable functions
            Evaluation metrics.

        Raises:
        -------
        ValueError: If metrics is not a list of callable functions.
        AttributeError: If the model does not have fit and predict methods.
        """
        if metrics is not None:
            for metric in metrics:
                if not callable(metric):
                    raise ValueError("Metrics must be a list of callable functions.")
        if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
            raise AttributeError("Model must implement fit and predict methods.")
