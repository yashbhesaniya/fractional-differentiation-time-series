"""
Created on Tue May 16 13:00:00 2023

@Group: Luis Alvaro Correia

Updated: 10 November 2023 by
@Group:
    Priyanka Teja Ravi
    Ramiscan Yakar
    Tolga Keskinoglu
"""
import pytest

import numpy as np

from src.finance_ml.denoising.denoising import Denoising

from numpy.testing import assert_almost_equal

def test_denoising():
    '''
    Tests consistency of Denoising on instantiation and the warning on inconsistent data entry.
    '''
    # Generating a random matrix
    X = np.random.normal(size=(1000,100))

    # Instanciate the Denoising transformer
    denoise_processor = Denoising().run_denoising(X)    
    
    ## Test Types ##
    assert isinstance(denoise_processor.get_cov_original, np.ndarray)
    assert isinstance(denoise_processor.get_corr_original, np.ndarray)
    
    assert isinstance(denoise_processor.get_cov_denoised, np.ndarray)
    assert isinstance(denoise_processor.get_corr_denoised, np.ndarray)
    assert isinstance(denoise_processor.get_eval_denoised, np.ndarray)
    assert isinstance(denoise_processor.get_evec_denoised, np.ndarray)
    
    assert isinstance(denoise_processor, Denoising)
    
    # Test diagonal of correlation matrices
    assert_almost_equal(np.diag(denoise_processor.get_corr_original), 
                        np.ones(denoise_processor.get_corr_original.shape[0]), decimal=8)
    assert_almost_equal(np.diag(denoise_processor.get_corr_denoised), 
                        np.ones(denoise_processor.get_corr_denoised.shape[0]), decimal=8)

    # Test off-diagonal elements are in the [-1,+1] interval
    assert np.logical_and(denoise_processor.get_corr_original >= -1.0, 
                          denoise_processor.get_corr_original <= 1.0).all()
    assert np.logical_and(denoise_processor.get_corr_denoised >= -1.0, 
                          denoise_processor.get_corr_denoised <= 1.0).all()
    
def test_denoising_param_alpha():
    with pytest.raises(ValueError):
        assert Denoising(alpha = 1.5)
        
def test_denoising_param_bWidth():
    with pytest.raises(ValueError):
        assert Denoising(bWidth = -1.5)
        
def test_denoising_param_method():
    with pytest.raises(ValueError):
        assert Denoising(method = 'some invalid method')
        
def test_denoising_param_nFacts():
    with pytest.raises(ValueError):
        assert Denoising(nFacts = -11.5)
        
def test_denoising_param_q():
    with pytest.raises(ValueError):
        assert Denoising(q = 0.15)
        
def test_denoising_param_market_component():
    with pytest.raises(ValueError):
        assert Denoising(market_component = -1)
  
def test_denoising_param_detoning():
    with pytest.raises(ValueError):
        assert Denoising(detoning = 1)
