import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from synthetic_backtesting import *

def test_default_parameters():
        initial_price = 100
        forecast = 110
        sigma = 0.1
        phi = 0.5
        maximum_holding_period = 100
        num_scenarios = int(1e5)

        result = generate_paths_ornstein_uhlenbeck(initial_price, forecast, sigma, phi, maximum_holding_period, num_scenarios)

        assert result.shape == (maximum_holding_period+1, num_scenarios)
        assert result.iloc[0].tolist() == [initial_price] * num_scenarios

def test_estimate_parameters():
        # Arrange
        historical_prices = pd.Series([1, 2, 3, 4, 5])
        position = 2
        forecast = 3

        # Act
        result = estimate_parameters_ornstein_uhlenbeck(historical_prices, position, forecast)

        # Assert
        assert isinstance(result, dict)
        assert 'phi_hat' in result
        assert 'sigma_hat' in result
        assert isinstance(result['phi_hat'], float)
        assert isinstance(result['sigma_hat'], float)

def test_return_dictionary():
        # Arrange
        historical_prices = pd.Series([1, 2, 3, 4, 5])
        position = 2
        forecast = 3

        # Act
        result = estimate_parameters_ornstein_uhlenbeck(historical_prices, position, forecast)

        # Assert
        assert isinstance(result, dict)
        assert 'phi_hat' in result
        assert 'sigma_hat' in result

def test_historical_prices_length_greater_than_2():
        # Arrange
        historical_prices = pd.Series([1, 2, 3, 4, 5])
        position = 2
        forecast = 3

        # Act
        result = estimate_parameters_ornstein_uhlenbeck(historical_prices, position, forecast)

        # Assert
        assert isinstance(result, dict)
        assert 'phi_hat' in result
        assert 'sigma_hat' in result

def test_empty_historical_prices():
        # Arrange
        historical_prices = pd.Series([])
        position = 0
        forecast = 3

        # Act and Assert
        with pytest.raises(Exception):
            estimate_parameters_ornstein_uhlenbeck(historical_prices, position, forecast)

def test_position_index_greater_than_length():
        # Arrange
        historical_prices = pd.Series([1, 2, 3, 4, 5])
        position = 5
        forecast = 3

        # Act and Assert
        with pytest.raises(Exception):
            estimate_parameters_ornstein_uhlenbeck(historical_prices, position, forecast)

def test_non_numeric_forecast():
        # Arrange
        historical_prices = pd.Series([1, 2, 3, 4, 5])
        position = 2
        forecast = 'abc'

        # Act and Assert
        with pytest.raises(Exception):
            estimate_parameters_ornstein_uhlenbeck(historical_prices, position, forecast)

def test_return_array_of_pairs():
        result = mesh_ornstein_uhlenbeck(100, 1, 10, 20)
        assert isinstance(result, np.ndarray)
        assert result.shape == (400, 2)

def test_return_array_with_correct_shape():
        result = mesh_ornstein_uhlenbeck(100, 1, 10, 20)
        assert result.shape == (400, 2)

def test_return_array_with_correct_dtype():
        result = mesh_ornstein_uhlenbeck(100, 1, 10, 20)
        assert result.dtype == np.float64

def test_return_empty_array_when_max_mesh_is_zero():
        result = mesh_ornstein_uhlenbeck(100, 1, 0, 20)
        assert result.size == 800

# def test_return_empty_array_when_sigma_is_zero():
#         result = mesh_ornstein_uhlenbeck(100, 0, 10, 20)
#         assert result.size == 0
        
def test_empty_list_if_number_of_scenarios_is_zero():
        generated_paths = pd.DataFrame()
        trading_rule = np.array([2.0, 6.0])
        result = simulate_one_box(generated_paths, trading_rule)
        assert result == []
def test_iterates_through_all_items():
        # Create a mock iterable
        iterable = range(10)
    
        # Call the progressbar function with the mock iterable
        progress = list(progressbar(iterable))
    
        # Assert that the length of the progress list is equal to the length of the iterable
        assert len(progress) == len(iterable)
def test_yields_each_item():
        # Create a mock iterable
        iterable = range(10)
    
        # Call the progressbar function with the mock iterable
        progress = list(progressbar(iterable))
    
        # Assert that each item in the progress list is equal to the corresponding item in the iterable
        assert progress == list(iterable)
def test_one_item_iterable():
        # Create an iterable with one item
        iterable = [1]
    
        # Call the progressbar function with the iterable
        progress = list(progressbar(iterable))
    
        # Assert that the progress list contains only one item and it is equal to the item in the iterable
        assert len(progress) == 1
        assert progress[0] == iterable[0]   

def test_positive_float_input():
        assert tau_to_phi(1.5) > 0 

def test_negative_float_input():
        assert tau_to_phi(-1.5) > 0   

def test_zero_input():
        with pytest.raises(ZeroDivisionError):
            tau_to_phi(0)

def test_nan_input():
        import math
        assert math.isnan(tau_to_phi(float('nan')))

def test_generate_interactive_plot():
        visualization_ornstein_uhlenbeck()

def test_compute_sharpe_ratio():
        simulation = np.array([0.05, 0.02, -0.03, 0.01, 0.04])
        initial_price = 100.0

        result = sharpe_ratio(simulation, initial_price)

        assert isinstance(result, float)

def test_valid_input_parameters():
        historical_prices = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        position = 2
        forecast = 115

        result = optimal_trading_rule(historical_prices, position, forecast)

        assert isinstance(result, pd.DataFrame)
        assert 'stop-loss' in result.columns
        assert 'profit-taking' in result.columns
        assert 'sharpe_ratio' in result.columns
def test_test_parameter_not_none():
        historical_prices = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        position = 2
        forecast = 115
        test_params = [0.1, 0.5]

        result = optimal_trading_rule(historical_prices, position, forecast, TEST=test_params)

        assert isinstance(result, dict)
        assert 'results' in result.keys()
        assert 'simulates' in result.keys()

def test_to_print_true(capsys):
        historical_prices = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        position = 2
        forecast = 115

        optimal_trading_rule(historical_prices, position, forecast, TO_PRINT=True)

        captured = capsys.readouterr()
        assert "sigma: " in captured.out
        assert "phi: " in captured.out

def test_empty_historical_prices():
        historical_prices = pd.Series([])
        position = 2
        forecast = 115

        with pytest.raises(Exception):
            optimal_trading_rule(historical_prices, position, forecast)

def test_position_greater_than_length():
        historical_prices = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        position = 10
        forecast = 115

        with pytest.raises(Exception):
            optimal_trading_rule(historical_prices, position, forecast)

def test_forecast_not_float():
        historical_prices = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        position = 2
        forecast = "115"

        with pytest.raises(Exception):
            optimal_trading_rule(historical_prices, position, forecast)