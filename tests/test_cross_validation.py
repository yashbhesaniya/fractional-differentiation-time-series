from unittest.mock import MagicMock, patch

import pandas as pd

from src.finance_ml.cross_validation.purge_embargo_cross_validation import PurgeEmbargoCrossValidation
from src.finance_ml.cross_validation.time_gap_cross_validation import TimeGapCrossValidation
from src.finance_ml.cross_validation.walk_forward_cross_validation import WalkForwardCrossValidation

# Sample data for testing
X = pd.DataFrame({'feature': range(100)})
y = pd.Series(range(100))

# Mock model and metrics for testing
mock_model = MagicMock()
mock_model.fit = MagicMock()
mock_model.predict = MagicMock(return_value=y)
mock_metric = MagicMock(return_value=0.5)


# Testing WalkForwardCrossValidation
class TestWalkForwardCrossValidation:
    def test_initialization(self):
        validator = WalkForwardCrossValidation(num_folds=5)
        assert validator.num_folds == 5

    @patch('matplotlib.pyplot.show')
    def test_cross_validate_with_plot(self, mock_show):
        validator = WalkForwardCrossValidation(num_folds=5)
        results = validator.cross_validate(X, y, mock_model, [mock_metric], plot=True)
        mock_show.assert_called()
        assert 'mock_metric' in results

    def test_walk_forward_splits(self):
        validator = WalkForwardCrossValidation(num_folds=5)
        splits = list(validator._walk_forward_splits(X))
        assert len(splits) == 5
        assert all(len(split) == 3 for split in splits)


# Testing TimeGapCrossValidation
class TestTimeGapCrossValidation:
    def test_initialization(self):
        validator = TimeGapCrossValidation(num_folds=5)
        assert validator.num_folds == 5

    @patch('matplotlib.pyplot.show')
    def test_cross_validate_with_plot(self, mock_show):
        validator = TimeGapCrossValidation(num_folds=5)
        results = validator.cross_validate(X, y, mock_model, [mock_metric], plot=True)
        mock_show.assert_called()
        assert 'mock_metric' in results

    def test_time_gap_splits(self):
        validator = TimeGapCrossValidation(num_folds=5)
        splits = list(validator._time_gap_splits(X, gap_percentage=0.2))
        assert len(splits) == 5
        assert all(len(split) == 3 for split in splits)


# Testing PurgeEmbargoCrossValidation
class TestPurgeEmbargoCrossValidation:
    def test_initialization(self):
        validator = PurgeEmbargoCrossValidation(num_folds=5, embargo_ratio=0.1)
        assert validator.num_folds == 5
        assert validator.embargo_ratio == 0.1

    @patch('matplotlib.pyplot.show')
    def test_cross_validate_with_plot(self, mock_show):
        validator = PurgeEmbargoCrossValidation(num_folds=5, embargo_ratio=0.1)
        results = validator.cross_validate(X, y, mock_model, [mock_metric], plot=True)
        mock_show.assert_called()
        assert 'mock_metric' in results

    def test_purge_embargo_splits(self):
        validator = PurgeEmbargoCrossValidation(num_folds=5, embargo_ratio=0.1)
        splits = list(validator._purge_embargo_splits(X))
        assert len(splits) == 5
        assert all(len(split) == 4 for split in splits)
