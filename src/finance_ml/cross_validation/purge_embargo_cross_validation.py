"""
PurgeEmbargoCrossValidation Class
---------------------------------
Author: Rabin BK (Matriculation Number: 23272000, IDM ID: yl03oxiq)
Author: Yatin Arora (Matriculation Number: 23014677, IDM ID: an79usew)

This class provides functionality for performing cross-validation with a purge and embargo period.

Usage:
------
1. Create an instance of PurgeEmbargoCrossValidation with the number of folds and embargo ratio.
2. Call the cross_validate method to perform cross-validation with a machine learning model and optional metrics.

Example:
--------
# Import necessary libraries
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

# Create an instance of PurgeEmbargoCrossValidation with 5 folds and an embargo ratio of 0.2
cv = PurgeEmbargoCrossValidation(num_folds=5, embargo_ratio=0.2)

# Perform cross-validation with a machine learning model and metrics
results = cv.cross_validate(X, y, model, metrics=[metric1, metric2], plot=True)

# Optionally, visualize the data splits if plot is set to True

"""
import pandas as pd
from typing import List, Callable, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


class PurgeEmbargoCrossValidation:
    """
    Perform cross-validation with a purge and embargo period.

    Attributes:
        num_folds (int): The number of cross-validation folds.
        embargo_ratio (float): The ratio of the embargo period.
    """

    def __init__(self, num_folds: int, embargo_ratio: float):
        """
        Initialize the PurgeEmbargoCrossValidation instance.

        Args:
            num_folds (int): The number of cross-validation folds.
            embargo_ratio (float): The ratio of the embargo period.
        """
        self.num_folds = num_folds
        self.embargo_ratio = embargo_ratio

    def _purge_embargo_splits(self, X: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Generate cross-validation splits with a purge and embargo period.

        Args:
            X (pd.DataFrame): The feature DataFrame.

        Yields:
            Tuple[int, int, int]: A tuple containing the start index of the test data, end index of the test data,
            and the embargo period size.
        """
        num_samples = len(X)
        fold_size = int(num_samples / self.num_folds)

        for fold_index in range(self.num_folds):
            train_end = fold_index * fold_size
            test_start = train_end
            test_end = test_start + fold_size

            embargo_size = int(self.embargo_ratio * fold_size)

            if test_end + embargo_size > num_samples:
                test_end = num_samples - embargo_size

            yield train_end, test_start, test_end, embargo_size

    def cross_validate(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   model: object,
                   metrics: Optional[List[Callable]] = None,
                   plot: bool = False) -> dict:
        """
        Perform cross-validation with a purge and embargo period and compute multiple evaluation metrics.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target variable.
            model: The machine learning model to evaluate.
            metrics (list of callable, optional): A list of evaluation metric functions. Defaults to None.
            plot (bool, optional): Whether to plot the data splits. Defaults to False.

        Returns:
            dict: A dictionary containing evaluation metric values for each fold and metric name.
        """
        self._check_parameters(model, metrics)
        evaluation_results = {metric.__name__: [] for metric in metrics}

        try:
            for fold_index, (train_end, test_start, test_end, embargo_size) in enumerate(self._purge_embargo_splits(X),
                                                                                         1):
                X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
                X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

                if X_train.empty or X_test.empty:
                    continue

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                for metric in metrics:
                    metric_value = metric(y_test, predictions)
                    evaluation_results[metric.__name__].append(metric_value)
                print(plot)
                if plot:
                    self._plot_data_splits(X, y, X_train, y_train, X_test, y_test, test_start, test_end, embargo_size,
                                           fold_index)
        except Exception as e:
            print(f"An error occurred during cross-validation: {str(e)}")

        return evaluation_results

    def _plot_data_splits(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      test_start: int,
                      test_end: int,
                      embargo_size: int,
                      fold_index: int):
        """
        Plot the cross-validation splits with a purge and embargo period.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target variable.
            X_train (pd.DataFrame): The training feature data.
            y_train (pd.Series): The training target data.
            X_test (pd.DataFrame): The test feature data.
            y_test (pd.Series): The test target data.
            test_start (int): The start index of the test data.
            test_end (int): The end index of the test data.
            embargo_size (int): The size of the embargo period.
            fold_index (int): The fold number for the plot.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(X.index, y, label='Data', linestyle='-', color='grey', alpha=0.3)
        plt.fill_between(X_train.index, y_train, color='skyblue', alpha=0.3, label='Training Data')
        plt.fill_between(X_test.index, y_test, color='salmon', alpha=0.3, label='Test Data')

        # Embargo period
        if embargo_size > 0 and test_end + embargo_size <= len(X):
            embargo_indices = X.index[test_end:test_end + embargo_size]
            plt.fill_between(embargo_indices, y[test_end:test_end + embargo_size], color='orange',
                             alpha=0.5, label='Embargo Period')

        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.title(f'Purge-Embargo Cross-Validation - Fold {fold_index}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def _check_parameters(self, model: object, metrics: List[Callable]):
        """
        Check if the model and metrics are valid.

        Args:
            model: The machine learning model to evaluate.
            metrics (list of callable): The evaluation metric functions.

        Raises:
            ValueError: If metrics are not callable.
            AttributeError: If model does not implement fit and predict methods.
        """
        if metrics is not None:
            for metric in metrics:
                if not callable(metric):
                    raise ValueError("Metrics must be a list of callable functions.")
        if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
            raise AttributeError("Model must implement fit and predict methods.")
