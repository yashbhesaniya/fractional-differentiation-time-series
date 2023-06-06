import warnings

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller
from sklearn import set_config
set_config(transform_output="pandas")

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

class FracDiff(BaseEstimator, TransformerMixin):

    def __init__(self,
                 d: float = [1.],
                 w: float = np.empty(1),
                 weights: bool = True,
                 rem_nan: bool = True,
                 method: str = 'fww', 
                 thres: float = 1e-2,
                 minimum: bool = False,
                 min_dict: dict = {'interv': [0., 1.], 
                                   'step': 1e-1, 
                                   'c_val_idx': int(1)}):
            
        self.d = d 
        self.w = w
        self.weights= weights
        self.rem_nan = rem_nan 
        self.method = method
        self.thres = thres
        self.minimum = minimum
        self.min_dict = min_dict

        check_inputs, warn = self.inputs_check()

        if not check_inputs: 
            raise TypeError(warn)

    def get_weights(self,
                    d: float = 1.,
                    size: int = 1,
                    method: str = 'fww',
                    thres: float = 1.e-2) -> np.ndarray:
        '''    
        Compute weights w_k to be applied to the k-th lagged element of the series.
        Adapted from:
        https://www.ostirion.net/post/stock-price-fractional-differentiation-best-fraction-finder
        and book Advances in Financial Machine Learning - Marcos M. Lopez de Prado - Chapter 5 - 1st edition 
        
        Args:        
        -----            
        d: float 
            Differentiation order of the series.
          
        thres: float
            Tolerance level to define weights to be discarded.
            
        Returns: 
        -----
        w: Numpy float array
            k-sized array with weights w.
        '''
        
        w = [1.0]
        k = 1
 
        if method == 'fww':     # selects the differentiation method
            
            while True:
                w_ = -w[-1]/k*(d-k+1)   # while grater than tolerance level, computes weights until k-th order
                if abs(w_) <= thres:
                    break
                w.append(w_)
                k += 1
                
        elif method == 'std':
            
            for k in range(k, size):
                w_ = -w[-1]/k*(d-k+1)   # compute weights considering the whole sample from T to T-k
                w.append(w_)
    
        w = np.array(w[::-1]).reshape(-1, 1)

        return w
    
    def calc_frac_diff(self,
                       X: pd.Series(dtype = float), 
                       d: float = 1.,
                       w: float =np.ones(1),
                       weights: bool = True, 
                       method: str = 'fww', 
                       thres: float = 1e-2        ) -> tuple[pd.Series(dtype = float), np.ndarray]:
        '''    
        Compute fractionally differentiated series.
        Adapted from:
        https://www.ostirion.net/post/stock-price-fractional-differentiation-best-fraction-finder
        and book Advances in Financial Machine Learning - Marcos M. Lopez de Prado - Chapter 5 - 1st edition 
        
        Args:        
        -----
        X: Pandas Series float
            Series to be differentiated.        
                
        d: float 
            Differentiation order of the series.       
          
        method: string 
            Chosen method. When 'fww', it applies fixes width window method, 
            and when 'std', applies the expanding window method.
            
        thres: float
            Tolerance level to define weights to be discarded.
            
        Returns: 
        -----
        X_tilda: Pandas Series float      
            Fracitonally differentiated series.

        w: Numpy array float
            k-sized array with weights w.
        '''
        
        l = 1
        size = X.shape[0]
              
        if method == 'fww':
            if weights == True:
                w = self.get_weights(d, size, method, thres)     # compute weights to be applied to the series
            l = len(w)
            if l > size:
                method = 'std'
        
        elif method == 'std':
            if weights == True:
                w = self.get_weights(d, size, method, thres)     # compute weights to be applied to the series
            w_ = np.cumsum(abs(w))             
            w_ /= w_[-1]
            l = w_[w_>thres].shape[0]   # drops relative weighted-loss greater than tolerance level
        
        results = {}
        r = range(l, size)

        for idx in r:
            
            if not np.isfinite(X.iloc[idx]): continue   # drop NAs

            if method == 'std':         
                results[idx] = np.dot(w[-(idx):].T, 
                                      X.iloc[:idx].to_numpy(dtype=float))[0]    # differentiating series

            elif method == 'fww':
                results[idx] = np.dot(w.T, 
                                      X.iloc[(idx-l):idx].to_numpy(dtype=float))[0]     # differentiating series

        X_tilda = pd.Series(results)
        X_tilda = X_tilda.set_axis(list(X[l:].index), axis=0)

        return X_tilda, w

    def min_fd_order(self,
                     X: pd.Series(dtype = float),
                     interv: float = [0., 1.], 
                     step: float = 1e-1, 
                     c_val_idx: int = 1,  
                     method: str = 'fww', 
                     thres: float = 1e-2         ) -> tuple[pd.Series(dtype=float), np.ndarray, float]:
        '''    
        Compute the minimum order d, which passes the ADF test to a given confidence leve, and
        returns the fractionally differentiated series by this minimum order d, as the minimum order as well.

        Args:        
        -----
        X: Pandas Series float
            Series to be differentiated.
        
        interv: float list
            Order d interval to be used in the differentiation.      
                
        step: float 
            Differentiation order increment.        
          
        c_val_idx: int
            3 possible value: 0, 1 e 2, to define the ADF test confidence
            level. Case 0: 1%, 1: 5%, 2: 10%.
            
        method: string
            Chosen method. When 'fww', it applies fixes width window method,
            and when 'std', applies the expanding window method.

        thres: float
            Tolerance level to define weights to be discarded.     
            
        Returns: 
        -----
        X_tilda: Pandas DataFrame float     
            DataFrame with all series fractionally differentiated.

        w: Pandas DataFrame float     
            DataFrame with columns containing computed weights' series to differentiate each series.

        d: float
            Series' minimum differetiation order.  
        '''
         
        series_ = pd.Series(dtype=float) 
        d_ = interv[0]
        
        while d_ <= interv[1]:

            series_, w_ = self.calc_frac_diff(X, d_, None, True, method, thres)     # differentiating series for each order d
                          
            adf = adfuller(series_,     # ADF statistics for d-order differentiated series                                             
                           maxlag = 1,
                           regression = 'c',
                           autolag = None)
            
            # critical values = {'1%': -3.4369193380671, '5%': -2.864440383452517, '10%': -2.56831430323573}
            # ADF test approximated critical values as per MacKinnon (1994, 2010)
            if adf[0] < list(adf[4].items())[c_val_idx][1]:     # applying test
                break
                
            d_ += step
     
        X_tilda = series_
        w = w_
        d = d_
    
        return X_tilda, w, d

    # estimating parameters
    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
        self.X_ = self.transform(X)            

        return self
        
    # transforming data
    def transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ) -> pd.DataFrame(dtype=float):
        '''    
        Transform series into a fractionally differentiated one. 
        It can be chosen between fixed width or expanding window methods, and also between
        a given differentiatio norder or find the minimum differentiation order that passes
        ADF test at a certain confidence leve.

        Args:        
        -----            
        self: object
            Al inputs froom function __init__.        

        X: Pandas DataFrame float
            DataFrame containing columns with series to be differentiated.

        Returns: 
        -----
        X_tilda: Pandas DataFrame float     
            DataFrame with all fractionally differentiated columns.

        w: Pandas DataFrame float     
            DataFrame with computed weighted to differentiate X.

        d: list float
            List of each order used in the differentiation of eac series in X.        
        '''
        
        d = []
        w = pd.DataFrame(dtype = float)
        
        if isinstance(X, pd.Series):               
            X = X.to_frame('0')     # case X is a series, it swaps into a DataFrame

        X_tilda = pd.DataFrame(dtype=float)     # initializing DataFrame to be returned and naming its columns
        
        if len(self.d) > X.shape[1]:
            return print("d list must have less or equal number of elements than series being differentiated")
        elif len(self.d) < X.shape[1]:
            for i in range(len(self.d), X.shape[1]):
                self.d.append(1.)

        for col in X.columns:
            idx = X.columns.get_loc(col)
            series_ = pd.Series(dtype=float)
            d_ = self.d[idx]
            w_ = pd.Series(dtype=float)
            
            if self.minimum:           
                series_, w_, d_ = self.min_fd_order(X[col],     # computing the differentiated series for minimum d-order, according to ADF test
                                          self.min_dict['interv'], 
                                          self.min_dict['step'], 
                                          self.min_dict['c_val_idx'],  
                                          self.method, 
                                          self.thres)

            else:                                                                                   
                series_, w_ = self.calc_frac_diff(X[col],       # computing differentiated series for a given d-order
                                                  d_,
                                                  self.w[~np.isnan(self.w[:,idx])][:,idx].reshape(-1,1), 
                                                  self.weights, 
                                                  self.method, 
                                                  self.thres)          
            
            # allocating series to a DataFrame, assuring the integrality of index
            df_series = pd.DataFrame(pd.Series(series_.values), columns=[str(col)+'_fd'])
            df_series = df_series.set_axis(list(series_.index), axis=0)
            X_tilda = pd.concat([X_tilda, df_series], axis=1)

            w = pd.concat([w, pd.DataFrame(w_.reshape(-1), columns=[str(col)+'_w'])], axis=1)
            
            d.append(d_)    # d-order of each series

        self.w = w
        self.d = d
        
        if self.rem_nan:  X_tilda.dropna(inplace=True)  # dropping NaNs of the higher d-order series

        return X_tilda
    
    def fit_transform(self, 
                  X: pd.DataFrame, 
                  y=None) -> pd.DataFrame(dtype=float):
        self.fit(X)
        return self.transform(X)
    
    def inputs_check(self) -> tuple[bool, str]: 
        
        warnings.formatwarning = custom_formatwarning
        check = True
        warn = ""
        
        # checking inputs types
        if type(self.w) == float: 
            self.w= np.array([self.w])
            warnings.warn("Changed w from float to Numpy array.")

        if self.w.ndim == 1: 
            self.w = self.w.reshape(-1,1)
            warnings.warn("w array redimensioned to (n, 1).")

        args_types = {'d': list,
                      'w': np.ndarray,
                      'weights': bool,
                      'rem_nan': bool,
                      'method': str, 
                      'thres': float,
                      'minimum': bool,
                      'min_dict': dict}
        
        min_dict = {'interv': list,
                    'step': float,
                    'c_val_idx': int}
        
        args_dict = self.__dict__

        for arg, val in args_dict.items():
            if not isinstance(val, args_types[arg]):
                check = False
                warn = f"Arg '{arg}' has incorrect type {type(val)}. It should be {args_types[arg]}."
                return check, warn

        for arg, val in self.min_dict.items():
            if not isinstance(val, min_dict[arg]):
                check = False
                warn = f"Arg min_dict['{arg}'] has incorrect type {type(val)}. It should be {min_dict[arg]}."
                return check, warn

        # cehcking input values
        if self.method not in ['fww', 'std']: 
            check = False
            warn = "'method' argument must be either 'fww' or 'std'."
            return check, warn
        
        if self.thres <= 0: 
            check = False
            warn = "Tolerance level thresh must be greater than 0."
            return check, warn
        
        if len(self.min_dict['interv']) != 2 or \
           self.min_dict['interv'][0] > self.min_dict['interv'][1]:
            
            check = False
            warn = "Search interval for minimum order must be a list like [a, b], where a < b and a,b are type float."
            return check, warn
        
        if self.min_dict['step'] <= 0 or \
            self.min_dict['step'] > self.min_dict['interv'][1] - self.min_dict['interv'][0]: 
            check = False
            warn = "Step must be greater than 0 and less or equal to (b - a), the interval extreme values in min_dict['interv']."
            return check, warn

        if self.min_dict['c_val_idx'] not in [0, 1, 2]: 
            check = False
            warn = "min_dict['c_val_idx'] argument must be either 0, 1 or 2."
            return check, warn

        return check, warn