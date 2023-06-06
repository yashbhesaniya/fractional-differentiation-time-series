import pytest
import warnings

import numpy as np
import pandas as pd
import sys
sys.path.append('../src/finance_ml')

from fractional_differentiation.frac_diff_transf import FracDiff

def test_class_outputs():
    '''
    Tests if types of outputs are coherent expected outputs
    '''

    inputs = {'d': [1.],
              'w': np.empty(1),
              'weights': True,
              'rem_nan': True,
              'method': 'fww', 
              'thres': 1e-2,
              'minimum': True,
              'min_dict': {'interv': [0., 1.], 
                           'step': 1e-1, 
                           'c_val_idx': int(1)}}
    
    X = pd.Series(np.random.normal(0.0, 0.01, 1000))
   
    frac_transf = FracDiff(**inputs)
        
    assert isinstance(frac_transf.transform(X), pd.DataFrame)    

    assert isinstance(frac_transf.get_weights(d = 1.,
                                              size = 1,
                                              method = 'fww',
                                              thres = 1.e-2), np.ndarray)       
    
    assert isinstance(frac_transf.calc_frac_diff(X, 
                                                 d= 1.,
                                                 w =np.ones(1),
                                                 weights = True, 
                                                 method= 'fww', 
                                                 thres = 1e-2)[0], pd.Series)       


    assert isinstance(frac_transf.calc_frac_diff(X, 
                                                 d = 1.,
                                                 w = np.ones(1),
                                                 weights = True, 
                                                 method = 'fww', 
                                                 thres = 1e-2)[1], np.ndarray)    

    assert isinstance(frac_transf.min_fd_order(X,
                                               interv = [0., 1.], 
                                               step = 1e-1, 
                                               c_val_idx = 1,  
                                               method = 'fww', 
                                               thres = 1e-2)[0], pd.Series)
    
    assert isinstance(frac_transf.min_fd_order(X,
                                               interv = [0., 1.], 
                                               step = 1e-1, 
                                               c_val_idx = 1,  
                                               method = 'fww', 
                                               thres = 1e-2)[1], np.ndarray)
    
    assert isinstance(frac_transf.min_fd_order(X,
                                               interv = [0., 1.], 
                                               step = 1e-1, 
                                               c_val_idx = 1,  
                                               method = 'fww', 
                                               thres = 1e-2)[2], float)

    assert isinstance(frac_transf.fit(X), type(frac_transf))
    
    assert isinstance(frac_transf.transform(X), pd.DataFrame)

    assert isinstance(frac_transf.fit_transform(X), pd.DataFrame)

def test_class_inputs():

    inputs = {'d': [1.],
          'w': np.empty(1),
          'weights': True,
          'rem_nan': True,
          'method': 'fww', 
          'thres': 1e-2,
          'minimum': True,
          'min_dict': {'interv': [0., 1.], 
                       'step': 1e-1, 
                       'c_val_idx': int(1)}}
    
    X = pd.Series(np.random.normal(0.0, 0.01, 1000))

    # testing warings are raised
    inputs['d'] = [1.]
    inputs['w'] = 0.
    with pytest.warns(UserWarning):
        warnings.warn("Changed w from float to Numpy array.", UserWarning)
        warnings.warn("w array redimensioned to (n, 1).", UserWarning)

    inputs['w'] = 0.
    with pytest.warns(UserWarning):
        warnings.warn("w array redimensioned to (n, 1).", UserWarning)

    # testing errors are raised
    # argument types
    inputs['d'] = 1.
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'd' has incorrect type {float}. It should be {list}."

    inputs['d'] = [1.]
    inputs['weights'] = 'error'
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'weights' has incorrect type {str}. It should be {bool}."

    inputs['rem_nan'] = 'error'
    inputs['weights'] = True
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'rem_nan' has incorrect type {str}. It should be {bool}."

    inputs['rem_nan'] = True
    inputs['method'] = 1
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'method' has incorrect type {int}. It should be {str}."

    inputs['thres'] = [0.01]
    inputs['method'] = 'fww'
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'thres' has incorrect type {list}. It should be {float}."

    inputs['thres'] = 1e-2
    inputs['minimum'] = 'error'
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'minimum' has incorrect type {str}. It should be {bool}."

    inputs['min_dict'] = np.array([0., 1., 2.])
    inputs['minimum'] = True
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == f"Arg 'min_dict' has incorrect type {np.ndarray}. It should be {dict}."

    # input values
    inputs['min_dict'] = {'interv': [0., 1.], 
                          'step': 1e-1, 
                          'c_val_idx': int(1)}
    inputs['rem_nan'] = True
    inputs['method'] = 'other method'
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == "'method' argument must be either 'fww' or 'std'."

    inputs['thres'] = -1e-2
    inputs['method'] = 'fww'
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == "Tolerance level thresh must be greater than 0."

    inputs['thres'] = 1e-2
    inputs['min_dict']['interv'] = [1., 0.]
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == "Search interval for minimum order must be a list like [a, b], where a < b and a,b are type float."

    inputs['min_dict']['step'] = -1e-2
    inputs['min_dict']['interv'] = [0., 1.]
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == "Step must be greater than 0 and less or equal to (b - a), the interval extreme values in min_dict['interv']."

    inputs['min_dict']['step'] = 1e-2
    inputs['min_dict']['c_val_idx'] = 3
    with pytest.raises(Exception) as excinfo:   
        FracDiff(**inputs)
    assert str(excinfo.value) == "min_dict['c_val_idx'] argument must be either 0, 1 or 2."
            
def test_results():
    '''
    Test if for differentiation order d =1, 
    transformer returns the difference between original series and shifted by 1 step series.
    '''
    inputs = {'d': [1.],
              'w': np.empty(1),
              'weights': True,
              'rem_nan': True,
              'method': 'fww', 
              'thres': 1e-2,
              'minimum': False,
              'min_dict': {'interv': [0., 1.], 
                           'step': 1e-1, 
                           'c_val_idx': int(1)}}
    
    X = pd.Series(np.random.normal(0.0, 0.01, 1000))
    frac_transf = FracDiff(**inputs)
    X_transf = frac_transf.transform(X)

    assert all((X - X.shift(1))[1:-1].values.reshape(-1,1) == X_transf)
    
    '''
    Test if when forcing weights w=0, then transformer yields a null series.
    '''
    inputs['w'] = 0.
    inputs['weights'] = False

    assert all(X_transf[X_transf == 0])