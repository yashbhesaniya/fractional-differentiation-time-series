
"""
Created on Mon Nov 27 12:33:00 2023
@Group:
    Ahmed Ibrahim
    Brijesh Mandaliya
    Moritz Link
"""
import pytest
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from numpy.testing import assert_almost_equal
from src.finance_ml.data_preparation.dataTransformer_class import DataTransformer

start_date = datetime.strptime("2022-10-17", "%Y-%m-%d")
end_date = datetime.strptime("2022-10-21", "%Y-%m-%d")
# difference between each date. D means one day
D = 'D'
date_list = pd.date_range(start_date, end_date, freq=D)

open = np.linspace(30, 300, 5)
low = open - 10
close = low + 30
hight = open + 50
volume = np.linspace(200, 1000, 5)
transactions  = np.linspace(5,50,5)

test_df = pd.DataFrame()
test_df["TEST_OPEN"] = open
test_df["TEST_LOW"] = low
test_df["TEST_CLOSE"] = close
test_df["TEST_HIGHT"] = hight
test_df["TEST_VOLUME"] = volume
test_df["TEST_TRANSACTIONS"] = transactions
test_df = test_df.set_index(date_list)

def test_DOLLAR_PRICE_COL_type():
    with pytest.raises(ValueError):
        assert DataTransformer({"TEST":""}, DOLLAR_PRICE_COL = 5)
def test_DOLLAR_PRICE_COL_param():
    with pytest.raises(ValueError):
        assert DataTransformer({"TEST":""},DOLLAR_PRICE_COL=  "Test")
def test_assets_df_type():
    dt = DataTransformer({"TEST":""})
    with pytest.raises(ValueError):
        assert dt.transform(assets_df="Test", bar_type="tick", bar_parameter=2)
def test_bar_type_type():
    dt = DataTransformer({"TEST":""})
    with pytest.raises(ValueError):
        assert dt.transform(assets_df=test_df, bar_type=1, bar_parameter=2)
def test_bar_type_param():
    dt = DataTransformer({"TEST":""})
    with pytest.raises(ValueError):
        assert dt.transform(assets_df=test_df, bar_type="test", bar_parameter=2)
def test_bar_parameter_float():
    dt = DataTransformer({"TEST":""})
    with pytest.raises(ValueError):
        assert dt.transform(assets_df=test_df, bar_type="tick", bar_parameter="Test")
def test_tick_bar_calculation_row0():
    volume_tick_solution_bar_parm_2_row_0 = 600
    open_tick_solution_bar_parm_2_row_0 = 30
    low_tick_solution_bar_parm_2_row_0 = 20.0
    close_tick_solution_bar_parm_2_row_0 = 117.5
    hight_tick_solution_bar_parm_2_row_0 = 147.5
    transactions_tick_solution_bar_parm_2_row_0 = 21.25
    tick_bars_row0 = np.array([open_tick_solution_bar_parm_2_row_0,hight_tick_solution_bar_parm_2_row_0, low_tick_solution_bar_parm_2_row_0,
                                close_tick_solution_bar_parm_2_row_0, volume_tick_solution_bar_parm_2_row_0,transactions_tick_solution_bar_parm_2_row_0])
    dt = DataTransformer({"TEST":""})
    r_df = dt.transform(assets_df=test_df, bar_type="tick", bar_parameter=2)
    result_row0 = r_df.iloc[0].values
    assert np.array_equal(tick_bars_row0, result_row0)
def test_tick_bar_calculation_row1():
    volume_tick_solution_bar_parm_2_row_1 = 1400
    open_tick_solution_bar_parm_2_row_1 = 165.0
    low_tick_solution_bar_parm_2_row_1 = 155.0
    close_tick_solution_bar_parm_2_row_1 = 252.5
    hight_tick_solution_bar_parm_2_row_1 = 282.5
    transactions_tick_solution_bar_parm_2_row_1 = 66.25
    tick_bars_row1 = np.array([open_tick_solution_bar_parm_2_row_1,hight_tick_solution_bar_parm_2_row_1, low_tick_solution_bar_parm_2_row_1,
                                close_tick_solution_bar_parm_2_row_1, volume_tick_solution_bar_parm_2_row_1,transactions_tick_solution_bar_parm_2_row_1])

    dt = DataTransformer({"TEST":""})
    r_df = dt.transform(assets_df=test_df, bar_type="tick", bar_parameter=2)
    result_row1 = r_df.iloc[1].values
    assert np.array_equal(tick_bars_row1, result_row1)
def test_volume_bar_calculation_row0():
    volume_volume_solution_bar_parm_1200_row_0 = 1200
    open_volume_solution_bar_parm_1200_row_0 = 30
    low_volume_solution_bar_parm_1200_row_0 = 20.0
    close_volume_solution_bar_parm_1200_row_0 = 185.0
    hight_volume_solution_bar_parm_1200_row_0 = 215.0
    transactions_volume_solution_bar_parm_1200_row_0 = 48.75
    volume_bars_row0 = np.array([open_volume_solution_bar_parm_1200_row_0,hight_volume_solution_bar_parm_1200_row_0, low_volume_solution_bar_parm_1200_row_0,
                                close_volume_solution_bar_parm_1200_row_0, volume_volume_solution_bar_parm_1200_row_0,transactions_volume_solution_bar_parm_1200_row_0])
    dt = DataTransformer({"TEST":""})
    r_df = dt.transform(assets_df=test_df, bar_type="volume", bar_parameter=1200)
    result_row0 = r_df.iloc[0].values

    assert np.array_equal(volume_bars_row0, result_row0)
def test_volume_bar_calculation_row1():
    volume_volume_solution_bar_parm_1200_row_1 = 1800
    open_volume_solution_bar_parm_1200_row_1 = 232.5
    low_volume_solution_bar_parm_1200_row_1 = 222.5
    close_volume_solution_bar_parm_1200_row_1 = 320.0
    hight_volume_solution_bar_parm_1200_row_1 = 350.0
    transactions_volume_solution_bar_parm_1200_row_1 = 88.75
    volume_bars_row1 = np.array([open_volume_solution_bar_parm_1200_row_1,hight_volume_solution_bar_parm_1200_row_1, low_volume_solution_bar_parm_1200_row_1,
                                close_volume_solution_bar_parm_1200_row_1, volume_volume_solution_bar_parm_1200_row_1,transactions_volume_solution_bar_parm_1200_row_1])
    dt = DataTransformer({"TEST":""})
    r_df = dt.transform(assets_df=test_df, bar_type="volume", bar_parameter=1200)
    result_row1 = r_df.iloc[1].values
    assert np.array_equal(volume_bars_row1, result_row1)
def test_dollar_bar_calculation_row0():
    volume_dollar_solution_bar_parm_1200_row_0 = 2000
    open_dollar_solution_bar_parm_1200_row_0 = 30
    low_dollar_solution_bar_parm_1200_row_0 = 20.0
    close_dollar_solution_bar_parm_1200_row_0 = 252.5
    hight_dollar_solution_bar_parm_1200_row_0 = 282.5
    transactions_dollar_solution_bar_parm_1200_row_0 = 48.75 + 38.75
    dollar_bars_row0 = np.array([open_dollar_solution_bar_parm_1200_row_0,hight_dollar_solution_bar_parm_1200_row_0, low_dollar_solution_bar_parm_1200_row_0,
                                close_dollar_solution_bar_parm_1200_row_0, volume_dollar_solution_bar_parm_1200_row_0,transactions_dollar_solution_bar_parm_1200_row_0])

    dt = DataTransformer({"TEST":""})
    r_df = dt.transform(assets_df=test_df, bar_type="dollar", bar_parameter=168001)
    result_row0 = r_df.iloc[0].values
    #print(np.array_equal(dollar_bars_row0, result_row0))
    assert np.array_equal(dollar_bars_row0, result_row0)
    #return np.array_equal(dollar_bars_row0, result_row0)
def test_dollar_bar_calculation_row1():
    volume_dollar_solution_bar_parm_1200_row_1 = 1000
    open_dollar_solution_bar_parm_1200_row_1 = 300
    low_dollar_solution_bar_parm_1200_row_1 = 290
    close_dollar_solution_bar_parm_1200_row_1 = 320.0
    hight_dollar_solution_bar_parm_1200_row_1 = 350.0
    transactions_dollar_solution_bar_parm_1200_row_1 = 50.0
    dollar_bars_row1 = np.array([open_dollar_solution_bar_parm_1200_row_1,hight_dollar_solution_bar_parm_1200_row_1, low_dollar_solution_bar_parm_1200_row_1,
                                close_dollar_solution_bar_parm_1200_row_1, volume_dollar_solution_bar_parm_1200_row_1,transactions_dollar_solution_bar_parm_1200_row_1])
    dt = DataTransformer({"TEST":""})
    r_df = dt.transform(assets_df=test_df, bar_type="dollar", bar_parameter=168001)
    result_row1 = r_df.iloc[1].values
    assert np.array_equal(dollar_bars_row1, result_row1)


