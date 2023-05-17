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
        """
        Scale data according with the scale_method selected.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variables 
                to be scaled.

        Returns:
            (pd.DataFrame): scaled data

        """
        values = data[col_Name].values.reshape(-1,1)
        if self.__scale_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1,1))
        else:
            scaler = StandardScaler()
        return scaler.fit_transform(values)

    def cal_price_change(self, 
                         data: pd.DataFrame(dtype=float), 
                         col_name:str) -> None:
        """
        Calculates the price change of a column passed as parameter and creates
            a new column whose name is composed by col_name+"_price_change" in 
            the data-frame passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            col_name (str): Column name.    

        Returns:
            None.

        """
        price_change = data[col_name].diff()
        data[col_name+"_price_change"] = pd.Series(price_change, index = data.index)
    
    def cal_pct_change(self, 
                       data: pd.DataFrame(dtype=float),
                       col_name:str) -> None:
        """
        Calculates the percentual price change of a column passed as parameter 
            and creates a new column whose name is composed by col_name+"_price_change" 
            in the data-frame passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            col_name (str): Column name.    

        Returns:
            None.

        """
        pct_change = data[col_name].pct_change()
        data[col_name+"_pct_change"] = pd.Series(pct_change, index = data.index)
    
    def cal_return(self, 
                   data: pd.DataFrame(dtype=float),
                   col_name:str) -> None:
        """
        Calculates the return of a column passed as parameter and creates a new 
            column whose name is composed by col_name+"_returns" in the data-frame 
            passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            col_name (str): Column name.    

        Returns:
            None.

        """
        values = data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = values[idx]/values[idx-1]
        data[col_name+"_returns"] = pd.Series(log_returns, index = data.index)
    
    def cal_log_return(self, 
                       data: pd.DataFrame(dtype=float),
                       col_name:str) -> None:
        """
        Calculates the log-return of a column passed as parameter and creates a new 
            column whose name is composed by col_name+"_log_returns" in the data-frame 
            passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            col_name (str): Column name.    

        Returns:
            None.

        """
        values = data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        data[col_name+"_log_returns"] = pd.Series(log_returns, index = data.index)
    
    def cal_Diff(self, 
                 data: pd.DataFrame(dtype=float),
                 col_open: str,
                 col_high: str,
                 col_low: str,
                 col_close ) -> None:
        """
        Calculates the amplitude between 'col_high' and 'col_low' and 'col_open' 
            and 'col_close', passed as parameter and creates a new columns named 
            'AMPL' and 'OPNCLS', respectively, in the data-frame passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            col_open (str): name of the column with the 'OPEN' data prices
            col_high (str): name of the column with the 'HIGH' data prices
            col_low (str): name of the column with the 'LOW' data prices
            col_close (str): name of the column with the 'CLOSE' data prices

        Returns:
            None.

        """
        values = data[[col_open,col_high,col_low,col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['open','high','low','close']
        data['AMPL'] = df_wrk['high']-df_wrk['low']
        data['OPNCLS'] = df_wrk['close']-df_wrk['open']
    
    def cal_RA(self, 
               data: pd.DataFrame(dtype=float),
               rol_win1: int,
               rol_win2: int,
               col_close: str ) -> None:
        """
        Calculates the rolling standard deviations of closing prices passed in 
            'col_close', considering the windoes 'rol_win1' and 'rol_win2' passed 
            as arguments and creates new columns named 'RA'+rol_win1 and 
            'RA'+rol_win2, respectively, in the data-frame passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            rol_win1 (int): number rolling of days to calculate the first std() 
            rol_win2 (int): number rolling of days to calculate the second std() 
            col_close (str): name of the column with the 'CLOSE' data prices

        Returns:
            None.

        """
        values = data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['close']
        data['RA_'+str(rol_win1)] = df_wrk['close'].rolling(window=rol_win1).std()
        data['RA_'+str(rol_win2)] = df_wrk['close'].rolling(window=rol_win2).std()
    
    def cal_MA(self, 
               data: pd.DataFrame(dtype=float),
               rol_win1: int,
               rol_win2: int,
               col_close: str ) -> None:
        """
        Calculates the moving average of 'CLOSE' prices passed in 'col_close', 
            considering the windoes 'rol_win1' and 'rol_win2' passed as arguments 
            and creates new columns named 'MA'+rol_win1 and 'MA'+rol_win2, 
            respectively, in the data-frame passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            rol_win1 (int): number rolling of days to calculate the first mean() 
            rol_win2 (int): number rolling of days to calculate the second mean() 
            col_close (str): name of the column with the 'CLOSE' data prices

        Returns:
            None.

        """
        values = data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['close']
        data['MA_'+str(rol_win1)] = df_wrk['close'].rolling(rol_win1, min_periods=1).mean()
        data['MA_'+str(rol_win2)] = df_wrk['close'].rolling(rol_win2, min_periods=1).mean()
    
    def cal_VMA(self, 
                data: pd.DataFrame(dtype=float),
                rol_win1: int,
                rol_win2: int,
                rol_win3: int,
                col_volume: str ) -> None:
        """
        Calculates the moving average of 'VOLUME' passed in 'col_volume', 
            considering the windoes 'rol_win1', 'rol_win2' and 'rol_win3' passed 
            as arguments and creates new columns named 'V_MA'+rol_win1, 
            'V_MA'+rol_win2 and 'V_MA'+rol_win3, respectively, in the data-frame 
            passed as argument.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variable 
                to be processed.
            rol_win1 (int): number rolling of days to calculate the first mean() 
            rol_win2 (int): number rolling of days to calculate the second mean() 
            rol_win3 (int): number rolling of days to calculate the third mean() 
            col_volume (str): name of the column with the 'VOLUME' data

        Returns:
            None.

        """
        values = data[col_volume].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ['volume']
        data['V_MA_'+str(rol_win1)] = df_wrk['volume'].rolling(rol_win1, min_periods=1).mean()
        data['V_MA_'+str(rol_win2)] = df_wrk['volume'].rolling(rol_win2, min_periods=1).mean()
        data['V_MA_'+str(rol_win3)] = df_wrk['volume'].rolling(rol_win3, min_periods=1).mean()

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
 
        if isinstance(X, pd.Series):               
            X = X.to_frame('0')                                    

        X_tilda = X.copy()
        
        self.cal_return(X_tilda, "CLOSE")
        self.cal_log_return(X_tilda, "CLOSE")
        self.cal_price_change(X_tilda, "CLOSE")
        self.cal_pct_change(X_tilda, "CLOSE")
        self.cal_RA(X_tilda, 5, 10, "CLOSE")
        self.cal_Diff(X_tilda, "OPEN", "HIGHT", "LOW", "CLOSE" )
        self.cal_MA(X_tilda, 5, 10, "CLOSE")
        self.cal_VMA(X_tilda, 5, 10, 20, "VOLUME")
        
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
