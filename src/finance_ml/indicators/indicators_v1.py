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
                 data: pd.DataFrame(dtype=float),
                 ticker: str = '',
                 norm_data: bool = False,
                 scale_method: str = "minmax",
                 sr_days: int = 20,
                 r_f: float = 0.02  ):
        
        """
        Initialize data.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variables 
                to be scaled.

        Returns:
            None

        """
        self.__data = data
        self.__ticker = (ticker+'_' if ticker!='' else '')
        self.__norm_data = norm_data
        self.__scale_method = scale_method
        self.__sr_days = sr_days
        self.__r_f = r_f
            
    @property
    def data(self):
        return self.__data

    @property
    def ticker(self):
        return self.__ticker

    @property
    def norm_data(self):
        return self.__norm_data

    @property
    def scale_method(self):
        return self.__scale_method

    @property
    def sr_days(self):
        return self.__sr_days

    @property
    def r_f(self):
        return self.__r_f

    def __scale_data(self, 
                   col_Name:str) -> pd.Series(dtype=float):
        """
        Scale data according with the scale_method selected.
            
        Args:
            self: object
                All entries in function __init__.        
            colname (str): Column of dataframe containing the variable 
                to be scaled.

        Returns:
            (np.array[]): scaled data

        """
        values = self.__data[col_Name].values.reshape(-1,1)
        if self.__scale_method == "minmax":
            scaler = MinMaxScaler(feature_range=(-1,1))
        else:
            scaler = StandardScaler()
        return scaler.fit_transform(values).values

    def __cal_price_change(self, 
                         col_name:str) -> None:
        """
        Calculates the price change of a column passed as parameter and creates
            a new column whose name is composed by col_name+"_price_change" in 
            the data-frame passed as argument. For compatibility purposes, it 
            was added the ticker label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_name (str): Column name.    

        Returns:
            None.

        """
        price_change = self.__data[col_name].diff()
        self.__data[col_name+"_price_change"] = pd.Series(price_change, index = self.__data.index)
    
    def __cal_pct_change(self, 
                       col_name:str) -> None:
        """
        Calculates the percentual price change of a column passed as parameter 
            and creates a new column whose name is composed by col_name+"_price_change" 
            in the data-frame passed as argument. For compatibility purposes, 
            it was added the ticker label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_name (str): Column name.    

        Returns:
            None.

        """
        pct_change = self.__data[col_name].pct_change()
        self.__data[col_name+"_pct_change"] = pd.Series(pct_change, index = self.__data.index)
    
    def __cal_return(self, 
                   col_name:str) -> None:
        """
        Calculates the return of a column passed as parameter and creates a new 
            column whose name is composed by col_name+"_returns" in the data-frame 
            passed as argument. For compatibility purposes, it was added the ticker
            label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_name (str): Column name.    

        Returns:
            None.

        """
        values = self.__data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = values[idx]/values[idx-1]
        self.__data[col_name+"_returns"] = pd.Series(log_returns, index = self.__data.index)
    
    def __cal_log_return(self, 
                       col_name:str) -> None:
        """
        Calculates the log-return of a column passed as parameter and creates a new 
            column whose name is composed by col_name+"_log_returns" in the data-frame 
            passed as argument. For compatibility purposes, it was added the ticker
            label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_name (str): Column name.    

        Returns:
            None.

        """
        values = self.__data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        self.__data[col_name+"_log_returns"] = pd.Series(log_returns, index = self.__data.index)
    
    def __cal_Diff(self, 
                 col_open: str,
                 col_high: str,
                 col_low: str,
                 col_close: str ) -> None:
        """
        Calculates the amplitude between "col_high" and "col_low" and "col_open" 
            and "col_close", passed as parameter and creates a new columns named 
            "AMPL" and "OPNCLS", respectively, in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_open (str): name of the column with the "OPEN" data prices
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[[col_open,col_high,col_low,col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["open","high","low","close"]
        self.__data[self.__ticker+"AMPL"] = (df_wrk["high"]-df_wrk["low"]).values
        self.__data[self.__ticker+"OPNCLS"] = (df_wrk["close"]-df_wrk["open"]).values
    
    def __cal_RA(self, 
               rol_win1: int,
               rol_win2: int,
               col_close: str ) -> None:
        """
        Calculates the rolling standard deviations of closing prices passed in 
            "col_close", considering the windoes "rol_win1" and "rol_win2" passed 
            as arguments and creates new columns named "RA"+rol_win1 and 
            "RA"+rol_win2, respectively, in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            rol_win1 (int): number rolling of days to calculate the first std() 
            rol_win2 (int): number rolling of days to calculate the second std() 
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        self.__data[self.__ticker+"RA_"+str(rol_win1)] = df_wrk["close"].rolling(window=rol_win1).std().values
        self.__data[self.__ticker+"RA_"+str(rol_win2)] = df_wrk["close"].rolling(window=rol_win2).std().values
    
    def __cal_MA(self, 
               rol_win1: int,
               rol_win2: int,
               col_close: str ) -> None:
        """
        Calculates the moving average of "CLOSE" prices passed in "col_close", 
            considering the windoes "rol_win1" and "rol_win2" passed as arguments 
            and creates new columns named "MA"+rol_win1 and "MA"+rol_win2, 
            respectively, in the data-frame passed as argument. For compatibility 
            purposes, it was added the ticker label in front of all columns 
            created.
            
        Args:
            self: object
                All entries in function __init__.        
            rol_win1 (int): number rolling of days to calculate the first mean() 
            rol_win2 (int): number rolling of days to calculate the second mean() 
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        self.__data[self.__ticker+"MA_"+str(rol_win1)] = df_wrk["close"].rolling(rol_win1, min_periods=1).mean().values
        self.__data[self.__ticker+"MA_"+str(rol_win2)] = df_wrk["close"].rolling(rol_win2, min_periods=1).mean().values
    
    def __cal_VMA(self, 
                rol_win1: int,
                rol_win2: int,
                rol_win3: int,
                col_volume: str ) -> None:
        """
        Calculates the moving average of "VOLUME" passed in "col_volume", 
            considering the windoes "rol_win1", "rol_win2" and "rol_win3" passed 
            as arguments and creates new columns named "V_MA"+rol_win1, 
            "V_MA"+rol_win2 and "V_MA"+rol_win3, respectively, in the data-frame 
            passed as argument. For compatibility purposes, it was added the ticker
            label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            rol_win1 (int): number rolling of days to calculate the first mean() 
            rol_win2 (int): number rolling of days to calculate the second mean() 
            rol_win3 (int): number rolling of days to calculate the third mean() 
            col_volume (str): name of the column with the "VOLUME" data

        Returns:
            None.

        """
        values = self.__data[col_volume].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["volume"]
        self.__data[self.__ticker+"V_MA_"+str(rol_win1)] = df_wrk["volume"].rolling(rol_win1, min_periods=1).mean().values
        self.__data[self.__ticker+"V_MA_"+str(rol_win2)] = df_wrk["volume"].rolling(rol_win2, min_periods=1).mean().values
        self.__data[self.__ticker+"V_MA_"+str(rol_win3)] = df_wrk["volume"].rolling(rol_win3, min_periods=1).mean().values
        
    #------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------
    def calculate_indicators (self):
        """
        Calculates the indicators of the dataframe provided. For compatibility 
            purposes, it was added the ticker label in front of all columns 
            created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        self.__cal_return(self.__ticker+"CLOSE")
        self.__cal_log_return(self.__ticker+"CLOSE")
        self.__cal_price_change(self.__ticker+"CLOSE")
        self.__cal_pct_change(self.__ticker+"CLOSE")
        self.__cal_RA(5, 10, self.__ticker+"CLOSE")
        self.__cal_Diff(self.__ticker+"OPEN",
                      self.__ticker+"HIGHT", 
                      self.__ticker+"LOW", 
                      self.__ticker+"CLOSE" )
        self.__cal_MA(5, 10, self.__ticker+"CLOSE")
        self.__cal_VMA(5, 10, 20, self.__ticker+"VOLUME")

    def normalize_data(self):
        """
        Noprmalizes the indicators of the dataframe provided. For compatibility 
            purposes, it was added the ticker label in front of all columns 
            created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        self.__data[self.__ticker+"N_OPEN"] = self.__scale_data(self.__ticker+"OPEN")
        self.__data[self.__ticker+"N_HIGHT"] = self.__scale_data(self.__ticker+"HIGHT")
        self.__data[self.__ticker+"N_LOW"] = self.__scale_data(self.__ticker+"LOW")
        self.__data[self.__ticker+"N_CLOSE"] = self.__scale_data(self.__ticker+"CLOSE")
        self.__data[self.__ticker+"N_VW"] = self.__scale_data(self.__ticker+"VW")
        self.__data[self.__ticker+"N_VOLUME"] = self.__scale_data(self.__ticker+"VOLUME")
        
        self.__data[self.__ticker+"N_CLOSE_returns"] = self.__scale_data(self.__ticker+"CLOSE_returns")
        self.__data[self.__ticker+"N_CLOSE_log_returns"] = self.__scale_data(self.__ticker+"CLOSE_log_returns")
        self.__data[self.__ticker+"N_CLOSE_price_change"] = self.__scale_data(self.__ticker+"CLOSE_price_change")
        self.__data[self.__ticker+"N_CLOSE_pct_change"] = self.__scale_data(self.__ticker+"CLOSE_pct_change")
        self.__data[self.__ticker+"N_RA_5"] = self.__scale_data(self.__ticker+"RA_5")
        self.__data[self.__ticker+"N_RA_10"] = self.__scale_data(self.__ticker+"RA_10")
        self.__data[self.__ticker+"N_AMPL"] = self.__scale_data(self.__ticker+"AMPL")
        self.__data[self.__ticker+"N_OPNCLS"] = self.__scale_data(self.__ticker+"OPNCLS")
        self.__data[self.__ticker+"N_MA_5"] = self.__scale_data(self.__ticker+"MA_5")
        self.__data[self.__ticker+"N_MA_10"] = self.__scale_data(self.__ticker+"MA_10")
        self.__data[self.__ticker+"N_V_MA_5"] = self.__scale_data(self.__ticker+"V_MA_5")
        self.__data[self.__ticker+"N_V_MA_10"] = self.__scale_data(self.__ticker+"V_MA_10")
        self.__data[self.__ticker+"N_V_MA_20"] = self.__scale_data(self.__ticker+"V_MA_20")
                

    def fit(self, 
            X: pd.DataFrame(dtype=float) = data, 
            y = None):
               
        """
        Method defined for compatibility purposes.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate indicators.

        Returns:
            self (objext)

        """
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
            self.__data (pd.DataFrame): Dataframe processed with indicators

        """
 
        if isinstance(X, pd.Series):               
            X = X.to_frame("0")

        self.calculate_indicators()                                  

        if self.__norm_data:
            self.normalize_data()

        return self.__data
