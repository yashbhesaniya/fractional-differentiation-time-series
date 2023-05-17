# Import required packages
import pandas as pd
import numpy as np

import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
set_config(transform_output="pandas")

class Volatility(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        pass

    def __getBeta(self, 
                  series: pd.Series(dtype=float),
                  sl: int):
        hl = series[['HIGHT','LOW']].values
        hl = np.log(hl[:,0]/hl[:,1])**2
        hl = pd.Series(hl,index=series.index)
        #beta = pd.stats.moments.rolling_sum(hl,window=2)
        #beta = pd.stats.moments.rolling_mean(beta,window=sl)
        beta = hl.rolling(window=2).sum()
        beta = beta.rolling(window=sl).mean()
        return beta.dropna()
    
    def __getGamma(self, 
                   series: pd.Series(dtype=float)):
        h2 = series['HIGHT'].rolling(window=2).max()
        #l2 = pd.stats.moments.rolling_min(series['Low'],window=2)
        #h2 = pd.stats.moments.rolling_max(series['High'],window=2)
        l2 = series['LOW'].rolling(window=2).min()
        gamma = np.log(h2.values/l2.values)**2
        gamma = pd.Series(gamma,index=h2.index)
        return gamma.dropna()
    
    def __getAlpha(self, 
                   beta: float, 
                   gamma: float):
        den = 3-2*2**.5
        alpha = (2**.5-1)*(beta**.5)/den
        alpha -= (gamma/den)**.5
        alpha[alpha<0] = 0 # set negative alphas to 0 (see p.727 of paper)
        return alpha.dropna()
    
    # ESTIMATING VOLATILITY FOR HIGH-LOW PRICES
    def __getSigma(self, 
                   beta: float, 
                   gamma: float):
        k2 = (8/np.pi)**.5
        den = 3-2*2**.5
        sigma = (2**-.5-1)*beta**.5/(k2*den)
        sigma += (gamma/(k2**2*den))**.5
        sigma[sigma<0] = 0
        return sigma
    
    # IMPLEMENTATION OF THE CORWIN-SCHULTZ Algorithm
    def corwinSchultz(self, 
                      series: pd.Series(dtype=float),
                      sl: int = 1):
        # Note: S<0 iif alpha<0
        beta = self.__getBeta(series,sl)
        gamma = self.__getGamma(series)
        alpha = self.__getAlpha(beta,gamma)
        spread = 2*(np.exp(alpha)-1)/(1+np.exp(alpha))
        startTime = pd.Series(series.index[0:spread.shape[0]],index=spread.index)
        spread = pd.concat([spread,startTime],axis=1)
        spread.columns = ['Spread','Start_Time'] # 1st loc used to compute beta
        return spread
    
    def get_GarmanKlaus(self,
                        price_data: pd.DataFrame(dtype=float), 
                        window: int = 30, 
                        trading_periods: int = 252, 
                        clean:bool = True):
    
        log_hl = (price_data['HIGHT'] / price_data['LOW']).apply(np.log)
        log_co = (price_data['CLOSE'] / price_data['OPEN']).apply(np.log)
    
        rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
        
        def f(v):
            return (trading_periods * v.mean())**0.5
        
        result = rs.rolling(window=window, center=False).apply(func=f)
        
        if clean:
            return result.dropna()
        else:
            return result
        
    def get_RogersSatchell(self, 
                           price_data: pd.DataFrame(dtype=float), 
                           window: int = 30, 
                           trading_periods: int = 252, 
                           clean:bool = True):
        
        log_ho = (price_data['HIGHT'] / price_data['OPEN']).apply(np.log)
        log_lo = (price_data['LOW'] / price_data['OPEN']).apply(np.log)
        log_co = (price_data['CLOSE'] / price_data['OPEN']).apply(np.log)
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
        def f(v):
            return trading_periods * v.mean()**0.5
        
        result = rs.rolling(
            window=window,
            center=False
        ).apply(func=f)
        
        if clean:
            return result.dropna()
        else:
            return result
        
    def get_YangZhang(self, 
                      price_data: pd.DataFrame(dtype=float), 
                      window: int = 30, 
                      trading_periods:int = 252, 
                      clean:bool = True):
    
        log_ho = (price_data['HIGHT'] / price_data['OPEN']).apply(np.log)
        log_lo = (price_data['LOW'] / price_data['OPEN']).apply(np.log)
        log_co = (price_data['CLOSE'] / price_data['OPEN']).apply(np.log)
        
        log_oc = (price_data['OPEN'] / price_data['CLOSE'].shift(1)).apply(np.log)
        log_oc_sq = log_oc**2
        
        log_cc = (price_data['CLOSE'] / price_data['CLOSE'].shift(1)).apply(np.log)
        log_cc_sq = log_cc**2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = log_cc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
    
        k = 0.34 / (1 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)
    
        if clean:
            return result.dropna()
        else:
            return result
        
    def get_HodgesTompkins(self, 
                           price_data: pd.DataFrame(dtype=float), 
                           window: int = 30, 
                           trading_periods: int = 252, 
                           clean:bool = True):
        
        log_return = (price_data['CLOSE'] / price_data['CLOSE'].shift(1)).apply(np.log)
    
        vol = log_return.rolling(
            window=window,
            center=False
        ).std() * math.sqrt(trading_periods)
    
        h = window
        n = (log_return.count() - h) + 1
    
        adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))
    
        result = vol * adj_factor
    
        if clean:
            return result.dropna()
        else:
            return result

    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        return self
        
    def transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ) -> pd.DataFrame(dtype=float):
        """
        Transforms the dataframe containing all variables of our financial series
            calculating the volatility indicators available.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate volatility indicators from.

        Returns:
            X_tilda (pd.DataFrame): Original Dataframe with volatility indicators

        """
 
        if isinstance(X,pd.Series):               
            X = X.to_frame('0')                                    

        X_tilda = X.copy()
        
        # Calculate Volatility Indicators
        X_tilda['CorwinSchultz'] = self.corwinSchultz(X_tilda)['Spread']
        X_tilda['HodgesTompkins'] = self.get_HodgesTompkins(X_tilda)
        X_tilda['ZhangYang'] = self.get_YangZhang(X_tilda)
        X_tilda['RogersSatchell'] = self.get_RogersSatchell(X_tilda)
        X_tilda['GarmanKlaus'] = self.get_GarmanKlaus(X_tilda)

        return X_tilda
