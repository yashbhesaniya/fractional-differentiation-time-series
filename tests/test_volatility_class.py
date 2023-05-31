"""
Created on Tue May 30 14:00:00 2023

@Group: 
    Coded by Luis Alvaro Correia
"""
import pytest

import numpy as np
import pandas as pd

from src.finance_ml.data_preparation.data_preparation import DataLoader
from src.finance_ml.volatility.volatility import Volatility

def test_volatility() -> pd.DataFrame(dtype=float):
    '''
    Tests consistency of Volatility on instantiation and the warning on inconsistent data entry.
    '''
    # Generating a random matrix
    # Defining time_index_col (must be the same column in all inputs) and 
    # keep_cols refering to the columns that will remain in the dataset
    dataloader = DataLoader(time_index_col= 'DATE', 
                    keep_cols = ['VOLUME','OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW','TRANSACTIONS'])
    fname_USDBRL = 'FX/USDBRL_2020-04-07_2022-04-06.parquet'

    # No. of Records from example dataset
    N = 10000
    
    # Dataset chosen in this simulation
    ticker = 'USDBRL'
    fname = fname_USDBRL
    
    df = dataloader.load_dataset({ticker:'data/'+fname}).iloc[:N]

    # Instanciate the Denoising transformer
    volatility_processor = Volatility(ticker = ticker)
    
    assert isinstance(volatility_processor._Volatility__atr_win, int)
    assert isinstance(volatility_processor._Volatility__bb_dev, int)
    assert isinstance(volatility_processor._Volatility__bb_win, int)
    assert isinstance(volatility_processor._Volatility__dc_off, int)
    assert isinstance(volatility_processor._Volatility__dc_win, int)
    assert isinstance(volatility_processor._Volatility__kc_mult, int)
    assert isinstance(volatility_processor._Volatility__kc_win, int)
    assert isinstance(volatility_processor._Volatility__kc_win_atr, int)
    assert isinstance(volatility_processor._Volatility__ticker, str)
    assert isinstance(volatility_processor._Volatility__ui_win, int)

    df = volatility_processor.fit_transform(df)
    
    # Test binary indicators - Bollinger Bands
    assert np.logical_and(df.filter(regex='BBHI').values >= 0.0, 
                          df.filter(regex='BBHI').values <= 1.0).all()
    assert np.logical_and(df.filter(regex='BBLI').values >= 0.0, 
                          df.filter(regex='BBLI').values <= 1.0).all()
    
    # Test binary indicators - Keltner Channel
    assert np.logical_and(df.filter(regex='KCHI').values >= 0.0, 
                          df.filter(regex='KCHI').values <= 1.0).all()
    assert np.logical_and(df.filter(regex='KCLI').values >= 0.0, 
                          df.filter(regex='KCLI').values <= 1.0).all()
    
def test_denoising_param_atr_win():
    with pytest.raises(ValueError):
        assert Volatility(atr_win = 1.5)

    with pytest.raises(ValueError):
        assert Volatility(atr_win = -1)
        
def test_denoising_param_bb_win():
    with pytest.raises(ValueError):
        assert Volatility(bb_win = 0.7)

    with pytest.raises(ValueError):
        assert Volatility(bb_win = -1)
        
def test_denoising_param_bb_dev():
    with pytest.raises(ValueError):
        assert Volatility(bb_dev = 4.7)

    with pytest.raises(ValueError):
        assert Volatility(bb_dev = -2)
        
    bb_win = 20
    with pytest.raises(ValueError):
        assert Volatility(bb_win = bb_win, bb_dev = bb_win + 2)
        
def test_denoising_param_dc_win():
    with pytest.raises(ValueError):
        assert Volatility(dc_win = 0.7)

    with pytest.raises(ValueError):
        assert Volatility(dc_win = -1)
        
def test_denoising_param_dc_off():
    with pytest.raises(ValueError):
        assert Volatility(dc_off = 3.1)

    with pytest.raises(ValueError):
        assert Volatility(dc_off = -6)
        
    dc_win = 20
    with pytest.raises(ValueError):
        assert Volatility(dc_win = dc_win, dc_off = dc_win)
        
def test_denoising_param_kc_win():
    with pytest.raises(ValueError):
        assert Volatility(kc_win = 1.7)

    with pytest.raises(ValueError):
        assert Volatility(kc_win = -1.4)
        
def test_denoising_param_kc_win_atr():
    with pytest.raises(ValueError):
        assert Volatility(kc_win_atr = 6.7)

    with pytest.raises(ValueError):
        assert Volatility(kc_win_atr = -0.9)
        
def test_denoising_param_kc_mult():
    with pytest.raises(ValueError):
        assert Volatility(kc_mult = 2.3)

    with pytest.raises(ValueError):
        assert Volatility(kc_mult = -4)
        
def test_denoising_param_ui_win():
    with pytest.raises(ValueError):
        assert Volatility(ui_win = 6.2)

    with pytest.raises(ValueError):
        assert Volatility(ui_win = -3.4)
        
        
        

        

        