"""
TimeGapCrossValidation Class
----------------------------
Author: Rabin BK (Matriculation Number: 23272000, IDM ID: yl03oxiq)
Author: Yatin Arora (Matriculation Number: 23014677, IDM ID: an79usew)

This class provides functionality for performing time-based cross-validation with a gap between train and test data.

Usage:
------
1. Create an instance of TimeGapCrossValidation with the number of folds.
2. Call the cross_validate method to perform cross-validation with a machine learning model and optional metrics.

Example:
--------
# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

# Create an instance of TimeGapCrossValidation with 5 folds
cv = TimeGapCrossValidation(num_folds=5)

# Perform cross-validation with a machine learning model and metrics
results = cv.cross_validate(X, y, model, metrics=[metric1, metric2], gap_percentage=0.1, plot=True)

# Optionally, visualize the data splits if plot is set to True

"""

from typing import List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter


class TimeGapCrossValidation:
    """
    Perform time-based cross-validation with a gap between train and test data.

    Attributes:
        num_folds (int): The number of cross-validation folds.
    """

    def __init__(self, num_folds: int = 5):
        """
        Initialize the TimeGapCrossValidation instance.

        Args:
            num_folds (int, optional): The number of cross-validation folds. Defaults to 5.
        """
        self._validate_init_params(num_folds)
        self.num_folds = num_folds

    def _validate_init_params(self, num_folds: int) -> None:
        """
        Validate the initialization parameters.

        Args:
            num_folds (int): The number of cross-validation folds.

        Raises:
            ValueError: If num_folds is not a positive integer.
        """
        if not isinstance(num_folds, int) or num_folds <= 0:
            raise ValueError("num_folds must be a positive integer.")

    def _time_gap_splits(self, X: pd.DataFrame, gap_percentage: float) -> Tuple[int, int, int]:
        """
        Generate time-based cross-validation splits with a gap.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            gap_percentage (float): The percentage of gap between train and test data.

        Yields:
            Tuple[int, int, int]: A tuple containing the start index of the test data, end index of the test data,
            and the gap size.
        """
        num_samples = len(X)
        fold_size = int(num_samples / self.num_folds)

        for fold_index in range(self.num_folds):
            half_fold_size = fold_size / 2
            gap_size = half_fold_size * gap_percentage
            train_data_size = half_fold_size - gap_size
            test_data_size = half_fold_size + gap_size

            test_start = (fold_index * fold_size) + train_data_size
            test_end = test_start + test_data_size

            test_start = int(test_start)
            test_end = int(test_end)

            if fold_index == self.num_folds - 1:
                test_end = num_samples
            if test_start >= num_samples:
                break

            yield test_start, test_end, int(gap_size)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, model: object,
                       metrics: Optional[List[Callable]] = None,
                       gap_percentage: float = 0, plot: bool = False) -> dict:
        """
        Perform time-based cross-validation with a gap between train and test data and compute multiple evaluation metrics.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target variable.
            model: The machine learning model to evaluate.
            metrics (list of callable, optional): A list of evaluation metric functions. Defaults to None.
            gap_percentage (float, optional): The percentage of gap between train and test data. Defaults to 0.
            plot (bool, optional): Whether to plot the data splits. Defaults to False.

        Returns:
            dict: A dictionary containing evaluation metric values for each fold and metric name.
        """
        self._check_parameters(model, metrics)
        evaluation_results = {metric.__name__: [] for metric in metrics}

        try:
            for test_start, test_end, gap_size in self._time_gap_splits(X, gap_percentage):
                X_train, y_train = X.iloc[:test_start], y.iloc[:test_start]
                X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

                if X_train.empty or X_test.empty:
                    continue

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                for metric in metrics:
                    metric_value = metric(y_test, predictions)
                    evaluation_results[metric.__name__].append(metric_value)

                if plot:
                    self._plot_data_splits(X, y, X_train, y_train, X_test, y_test, test_start, test_end, gap_size * 2)
        except Exception as e:
            print(f"An error occurred during cross-validation: {str(e)}")

        return evaluation_results

    def _plot_data_splits(self, X: pd.DataFrame, y: pd.Series,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          test_start: int, test_end: int, gap_size: int) -> None:
        """
        Plot the time-based cross-validation splits.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target variable.
            X_train (pd.DataFrame): The training feature data.
            y_train (pd.Series): The training target data.
            X_test (pd.DataFrame): The test feature data.
            y_test (pd.Series): The test target data.
            test_start (int): The start index of the test data.
            test_end (int): The end index of the test data.
            gap_size (int): The size of the gap between train and test data.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(X.index, y, label='Data', linestyle='-', color='grey', alpha=0.3)
        plt.fill_between(X_train.index, y_train, color='skyblue', alpha=0.3, label='Training Data')
        plt.fill_between(X_test.index, y_test, color='salmon', alpha=0.3, label='Test Data')

        if gap_size > 0 and test_start > gap_size:
            gap_indices = X.index[test_start - gap_size:test_start]
            plt.fill_between(gap_indices, y[test_start - gap_size:test_start], color='orange', alpha=0.5, label='Gap')

        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.title(f'Time Gap Cross-Validation')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def _check_parameters(self, model: object, metrics: Optional[List[Callable]]) -> None:
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
