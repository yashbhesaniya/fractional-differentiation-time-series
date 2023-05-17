# Import required packages
import pandas as pd
import numpy as np

import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import set_config
set_config(transform_output="pandas")


class Indicators(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 norm_data: bool = False,
                 scale_method: str = 'minmax',
                 sr_days: int = 20,
                 r_f: float = 0.02  ):
        
        self.__norm_data = norm_data
        self.__scale_method = scale_method
        self.__sr_days = sr_days
        self.__r_f = r_f
            
    def scale_data(self, 
                   data, 
                   col_Name:str) -> pd.Series(dtype=float):
        values = data[col_Name].values.reshape(-1,1)
        if self.__scale_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1,1))
        else:
            scaler = StandardScaler()
        return scaler.fit_transform(values)

    def cal_price_change(self, 
                         data: pd.DataFrame(dtype=float), 
                         col_name:str) -> None:
        price_change = data[col_name].diff()
        data[col_name+"_price_change"] = pd.Series(price_change, index = data.index)
    
    def cal_pct_change(self, 
                       data: pd.DataFrame(dtype=float),
                       col_name:str) -> None:
        pct_change = data[col_name].pct_change()
        data[col_name+"_pct_change"] = pd.Series(pct_change, index = data.index)
    
    def cal_return(self, 
                   data: pd.DataFrame(dtype=float),
                   col_name:str) -> None:
        values = data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = values[idx]/values[idx-1]
        data[col_name+"_returns"] = pd.Series(log_returns, index = data.index)
    
    def cal_log_return(self, 
                       data: pd.DataFrame(dtype=float),
                       col_name:str) -> None:
        values = data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        data[col_name+"_log_returns"] = pd.Series(log_returns, index = data.index)
    
    def cal_Diff(self, 
                 data: pd.DataFrame(dtype=float)) -> None:
        values = data[['OPEN','HIGHT','LOW','CLOSE']].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['open','high','low','close']
        data['AMPL'] = df_wrk['high']-df_wrk['low']
        data['OPNCLS'] = df_wrk['close']-df_wrk['open']
    
    def cal_RA(self, 
               data: pd.DataFrame(dtype=float)) -> None:
        values = data['CLOSE'].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['close']
        data['RA_5'] = df_wrk['close'].rolling(window=5).std()
        data['RA_10'] = df_wrk['close'].rolling(window=10).std()
    
    def cal_MA(self, 
               data: pd.DataFrame(dtype=float)) -> None:
        values = data['CLOSE'].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['close']
        data['MA_5'] = df_wrk['close'].rolling(5, min_periods=1).mean()
        data['MA_10'] = df_wrk['close'].rolling(10, min_periods=1).mean()
    
    def cal_VMA(self, 
                data: pd.DataFrame(dtype=float)) -> None:
        values = data['VOLUME'].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['volume']
        data['V_MA_5'] = df_wrk['volume'].rolling(5, min_periods=1).mean()
        data['V_MA_10'] = df_wrk['volume'].rolling(10, min_periods=1).mean()
        data['V_MA_20'] = df_wrk['volume'].rolling(20, min_periods=1).mean()

    def cal_SharpRatio (self, 
                        data: pd.DataFrame(dtype=float), 
                        col_name: str) -> None:
        values = data[col_name+'_returns'].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = [col_name+'_returns']
        data[col_name+'_SharpeRatio'] = df_wrk[col_name].rolling(self.__sr_days, min_periods=1).mean()
        data['V_MA_10'] = df_wrk['volume'].rolling(10, min_periods=1).mean()
        data['V_MA_20'] = df_wrk['volume'].rolling(20, min_periods=1).mean()

    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        return self
        
    def transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ) -> pd.DataFrame(dtype=float):
        """
        Transforms the dataframe containing all variables of our financial series
            calculating the indicators.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate indicators from.

        Returns:
            X_tilda (pd.DataFrame): Original Dataframe with indicators

        """
 
        if isinstance(X,pd.Series):               
            X = X.to_frame('0')                                    

        X_tilda = X.copy()
        
        self.cal_return(X_tilda, "CLOSE")
        self.cal_log_return(X_tilda, "CLOSE")
        self.cal_price_change(X_tilda, "CLOSE")
        self.cal_pct_change(X_tilda, "CLOSE")
        self.cal_RA(X_tilda)
        self.cal_Diff(X_tilda)
        self.cal_MA(X_tilda)
        self.cal_VMA(X_tilda)
        
        if self.__norm_data:
            # Normalize Data
            X_tilda['N_OPEN'] = self.scale_data(X_tilda,'OPEN')
            X_tilda['N_HIGHT'] = self.scale_data(X_tilda,'HIGHT')
            X_tilda['N_LOW'] = self.scale_data(X_tilda,'LOW')
            X_tilda['N_CLOSE'] = self.scale_data(X_tilda,'CLOSE')
            X_tilda['N_VW'] = self.scale_data(X_tilda,'VW')
            X_tilda['N_VOLUME'] = self.scale_data(X_tilda,'VOLUME')
            
            X_tilda['N_CLOSE_returns'] = self.scale_data(X_tilda,'CLOSE_returns')
            X_tilda['N_CLOSE_log_returns'] = self.scale_data(X_tilda,'CLOSE_log_returns')
            X_tilda['N_CLOSE_price_change'] = self.scale_data(X_tilda,'CLOSE_price_change')
            X_tilda['N_CLOSE_pct_change'] = self.scale_data(X_tilda,'CLOSE_pct_change')
            X_tilda['N_RA_5'] = self.scale_data(X_tilda,'RA_5')
            X_tilda['N_RA_10'] = self.scale_data(X_tilda,'RA_10')
            X_tilda['N_AMPL'] = self.scale_data(X_tilda,'AMPL')
            X_tilda['N_OPNCLS'] = self.scale_data(X_tilda,'OPNCLS')
            X_tilda['N_MA_5'] = self.scale_data(X_tilda,'MA_5')
            X_tilda['N_MA_10'] = self.scale_data(X_tilda,'MA_10')
            X_tilda['N_V_MA_5'] = self.scale_data(X_tilda,'V_MA_5')
            X_tilda['N_V_MA_10'] = self.scale_data(X_tilda,'V_MA_10')
            X_tilda['N_V_MA_20'] = self.scale_data(X_tilda,'V_MA_20')
    
        return X_tilda
