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
    denoise_processor = Denoising()    
    
    # Calculates non-denoised Covariance Matrix
    cov0 = np.cov(X,rowvar=0)
    corr0 = Denoising.cov2corr(cov0)

    # Calculates Correlation, Covariance, EigenValues and EigenVectors of denoised covariance matrix
    cov1, corr1, eVal1, eVec1 = denoise_processor.transform(X)
    
    ## Test Types ##
    print('Checking consistency of types...')
    assert isinstance(cov0, np.ndarray)
    assert isinstance(corr0, np.ndarray)
    
    assert isinstance(cov1, np.ndarray)
    assert isinstance(corr1, np.ndarray)
    assert isinstance(eVal1, np.ndarray)
    assert isinstance(eVec1, np.ndarray)
    
    assert isinstance(denoise_processor, Denoising)
    
    print('Consistency of types PASSED!')
    
    # Test diagonal of correlation matrices
    print('\nChecking correlation matrices produced...')
    assert_almost_equal(np.diag(corr0), np.ones(corr0.shape[0]), decimal=8)
    assert_almost_equal(np.diag(corr1), np.ones(corr1.shape[0]), decimal=8)

    # Test off-diagonal elements are in the [-1,+1] interval
    assert np.logical_and(corr0 >= -1.0, corr0 <= 1.0).all()
    assert np.logical_and(corr1 >= -1.0, corr1 <= 1.0).all()
    print('Correlation matrices produced PASSED!')
    
    # Teste Instantiation error handling (invalid alpha)
    print('\nTesting Error handling of Denoising class...')
    with pytest.warns(Warning):
        dp_error = Denoising(alpha = 1.5)
    assert dp_error
    
    # Teste Instantiation error handling (invalid bWidth)
    with pytest.warns(Warning):
        dp_error = Denoising(bWidth = 1.5)
    assert dp_error

    # Teste Instantiation error handling (invalid method)
    with pytest.warns(Warning):
        dp_error = Denoising(method = 'some invalid method')
    assert dp_error
    
    # Teste Instantiation error handling (invalid nFacts)
    with pytest.warns(Warning):
        dp_error = Denoising(nFacts = -11.5)
    assert dp_error

    # Teste Instantiation error handling (invalid q=T/N)
    with pytest.warns(Warning):
        dp_error = Denoising(q = 0.15)
    assert dp_error
