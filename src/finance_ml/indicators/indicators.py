"""
Created on Thu May 11 19:00:00 2023

@author: Luis Alvaro Correia
"""
# Import required packages
import pandas as pd
import numpy as np

import math

from ta.momentum import KAMAIndicator, PercentagePriceOscillator, PercentageVolumeOscillator, \
                ROCIndicator, RSIIndicator, StochRSIIndicator, StochasticOscillator, \
                AwesomeOscillatorIndicator, TSIIndicator, UltimateOscillator, WilliamsRIndicator
from ta.volume import AccDistIndexIndicator



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import set_config
set_config(transform_output="pandas")

class Indicators(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 ticker: str = '',
                 norm_data: bool = False,
                 scale_method: str = "minmax",
                 sr_days: int = 20,
                 r_f: float = 0.02,
                 KAMA_win: int = 10,
                 KAMA_pow1: int = 2,
                 KAMA_pow2: int = 30,
                 PPO_win_slow: int = 26,
                 PPO_win_fast: int = 12,
                 PPO_win_sign: int = 9,
                 PVO_win_slow: int = 26,
                 PVO_win_fast: int = 12,
                 PVO_win_sign: int = 9,
                 ROC_win: int = 12,
                 RSI_win: int = 14,
                 StRSI_win: int = 14,
                 StRSI_sm1: int = 3,
                 StRSI_sm2: int = 3,
                 SO_win: int = 14,
                 SO_sm: int = 3,
                 AOI_win1: int = 5,
                 AOI_win2: int = 34,
                 TSI_win_slow: int = 25,
                 TSI_win_fast: int = 13,
                 UO_win1: int = 7,
                 UO_win2: int = 14,
                 UO_win3: int = 28,
                 UO_weight1: float = 4.0,
                 UO_weight2: float = 2.0,
                 UO_weight3: float = 1.0,
                 WRI_lbp: int = 14,
                 ):
        
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
        self.__ticker = (ticker+'_' if ticker!='' else '')
        self.__norm_data = norm_data
        self.__scale_method = scale_method
        self.__sr_days = sr_days
        self.__r_f = r_f
        self.__KAMA_win = KAMA_win
        self.__KAMA_pow1 = KAMA_pow1
        self.__KAMA_pow2 = KAMA_pow2
        self.__PPO_win_slow = PPO_win_slow
        self.__PPO_win_fast = PPO_win_fast
        self.__PPO_win_sign = PPO_win_sign
        self.__PVO_win_slow = PVO_win_slow
        self.__PVO_win_fast = PVO_win_fast
        self.__PVO_win_sign = PVO_win_sign
        self.__ROC_win = ROC_win
        self.__RSI_win = RSI_win
        self.__StRSI_win = StRSI_win
        self.__StRSI_sm1 = StRSI_sm1
        self.__StRSI_sm2 = StRSI_sm2    
        self.__SO_win = SO_win
        self.__SO_sm = SO_sm
        self.__AOI_win1 = AOI_win1
        self.__AOI_win2 = AOI_win2
        self.__TSI_win_slow = TSI_win_slow
        self.__TSI_win_fast = TSI_win_fast
        self.__UO_win1 = UO_win1
        self.__UO_win2 = UO_win2
        self.__UO_win3 = UO_win3
        self.__UO_weight1 = UO_weight1
        self.__UO_weight2 = UO_weight2
        self.__UO_weight3 = UO_weight3
        self.__WRI_lbp = WRI_lbp
            
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
        
    #---------------------------- TECHNICAL INDICATORS ----------------------------------
    def __cal_KAMA(self,
               col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Kaufmann Adaptative Moving Average (KAMA) moving average 
            designed to account for market noise or volatility. "CLOSE" prices 
            passed in "col_close" and creates new columns named "KAMA"+KAMA_win, 
            in the data-frame passed as argument. For compatibility 
            purposes, it was added the ticker label in front of all columns 
            created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize KAMA Indicator
        indicator_KAMA = KAMAIndicator(close=df_wrk["close"], window = self.__KAMA_win, 
                                       pow1 = self.__KAMA_pow1, pow2 = self.__KAMA_pow2)

        self.__data[self.__ticker+"KAMA_"+str(self.__KAMA_win)] = indicator_KAMA.kama().values
    
    def __cal_PPO(self,
               col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Percentage Price Oscilator (PPO) which is a momentum oscillator 
            that measures the difference between two moving averages as a percentage 
            of the larger moving average over "CLOSE" prices passed in "col_close",
            "win_fast", "win_slow" and "win_sign", creating new columns named 
            "PPO"+win_slow+"_"+win_fast in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Percentage Price Oscilator Indicator
        indicator_PPO = PercentagePriceOscillator(close=df_wrk["close"], window_slow = self.__PPO_win_slow, 
                                       window_fast = self.__PPO_win_fast, window_sign = self.__PPO_win_sign)

        self.__data[self.__ticker+"PPO_"+str(self.__PPO_win_slow)+"_"+str(self.__PPO_win_fast)] = \
                indicator_PPO.ppo().values
    
    def __cal_PVO(self,
               col_volume: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Percentage Volume Oscilator (PVO) which is a momentum oscillator 
            that measures the difference between two moving averages as a percentage 
            of the larger moving average over "VOLUME" prices passed in "col_close",
            "win_fast", "win_slow" and "win_sign", creating new columns named 
            "PVO"+win_slow+"_"+win_fast in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_volume (str): name of the column with the "VOLUME" data

        Returns:
            None.

        """
        values = self.__data[col_volume].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["volume"]
        
        # Initialize Percentage Volume Oscilator Indicator
        indicator_PVO = PercentageVolumeOscillator(volume = df_wrk["volume"], window_slow = self.__PVO_win_slow, 
                                       window_fast = self.__PVO_win_fast, window_sign = self.__PVO_win_sign)

        self.__data[self.__ticker+"PVO_"+str(self.__PVO_win_slow)+"_"+str(self.__PVO_win_fast)] = \
                indicator_PVO.pvo().values
        self.__data[self.__ticker+"PVOH_"+str(self.__PVO_win_slow)+"_"+str(self.__PVO_win_fast)] = \
                indicator_PVO.pvo_hist().values
        self.__data[self.__ticker+"PVOsgn_"+str(self.__PVO_win_slow)+"_"+str(self.__PVO_win_fast)] = \
                indicator_PVO.pvo_signal().values
    
    def __cal_ROC(self,
               col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates The Rate-of-Change (ROC) indicator, which is also referred to 
            as simply Momentum, is a pure momentum oscillator that measures the 
            percent change in price from one period to the next over "CLOSE" prices 
            passed in "col_close" and "roc_win", creating new columns named 
            "ROC"+roc_win in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize ROC Indicator
        indicator_ROC = ROCIndicator(close=df_wrk["close"], window = self.__ROC_win)

        self.__data[self.__ticker+"ROC_"+str(self.__ROC_win)] = indicator_ROC.roc().values
    
    def __cal_RSI(self,
               col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Relative Strength Index (RSI) which compares the magnitude of 
            recent gains and losses over a specified time period to measure speed 
            and change of price movements of a security over "CLOSE" prices 
            passed in "col_close" and "rsi_win", creating new columns named 
            "RSI"+roc_win in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize RSI Indicator
        indicator_RSI = RSIIndicator(close=df_wrk["close"], window = self.__RSI_win)

        self.__data[self.__ticker+"RSI_"+str(self.__RSI_win)] = indicator_RSI.rsi().values
    
    def __cal_StochRSI(self,
               col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates The StochRSI oscillator which was developed to take advantage of both 
            momentum indicators in order to create a more sensitive indicator that 
            is attuned to a specific security’s historical performance rather than 
            a generalized analysis of price change. It is calculated over "CLOSE" prices 
            passed in "col_close", "strsi_win", "smooth1" and "smooth2" creating 
            new columns named "StRSI"+strsi_win in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_StRSI = StochRSIIndicator(close=df_wrk["close"], window = self.__StRSI_win,
                                            smooth1 = self.__StRSI_sm1, smooth2 = self.__StRSI_sm2)

        self.__data[self.__ticker+"StRSI_"+str(self.__StRSI_win)] = indicator_StRSI.stochrsi().values
        self.__data[self.__ticker+"StRSId_"+str(self.__StRSI_win)] = indicator_StRSI.stochrsi_d().values
        self.__data[self.__ticker+"StRSIk_"+str(self.__StRSI_win)] = indicator_StRSI.stochrsi_k().values
    
    def __cal_SO(self,
                 col_high: str,
                 col_low: str,
                 col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates The Stochastic Oscillator which Developed in the late 1950s by 
            George Lane. The stochastic oscillator presents the location of the 
            closing price of a stock in relation to the high and low range of the 
            price of a stock over a period of time, typically a 14-day period. 
            It is calculated over "col_high" and "col_low" and "col_close", passed 
            as parameter in "col_close", "strsi_win", "smooth1" and "smooth2" creating 
            new columns named "StRSI"+strsi_win in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[[col_high,col_low,col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Stochastic Indicator
        indicator_SO = StochasticOscillator(high = df_wrk["high"], low = df_wrk["low"], 
                                            close=df_wrk["close"], window = self.__SO_win,
                                            smooth_window = self.__SO_sm)

        self.__data[self.__ticker+"SO_"+str(self.__SO_win)] = indicator_SO.stoch().values
        self.__data[self.__ticker+"SOsgn_"+str(self.__SO_win)] = indicator_SO.stoch_signal().values    
     
    def __cal_AOI(self,
                 col_high: str,
                 col_low: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates The Awesome Oscillator is an indicator used to measure market momentum. 
            AO calculates the difference of a 34 Period and 5 Period Simple Moving Averages. 
            It is calculated over "col_high" and "col_low", passed as parameter, 
            besides "AOI_win1" and "AOI_win2" creating new columns named 
            "AOI_"+AOI_win1+AOI_win2 in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices

        Returns:
            None.

        """
        values = self.__data[[col_high,col_low]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low"]
        
        # Initialize Awesome Oscillator Indicator
        indicator_AOI = AwesomeOscillatorIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                                   window1 = self.__AOI_win1, window2 = self.__AOI_win2)

        self.__data[self.__ticker+"AOI_"+str(self.__AOI_win1)+"_"+str(self.__AOI_win2)] = \
                    indicator_AOI.awesome_oscillator().values   
     
    def __cal_TSI(self,
               col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates True strength index (TSI) which shows both trend direction 
            and overbought/oversold conditions over "CLOSE" prices passed in "col_close",
            "win_fast" and "win_slow", creating new columns named 
            "TSI"+win_slow+"_"+win_fast in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Percentage Price Oscilator Indicator
        indicator_TSI = TSIIndicator(close=df_wrk["close"], window_slow = self.__TSI_win_slow, 
                                       window_fast = self.__TSI_win_fast )

        self.__data[self.__ticker+"TSI_"+str(self.__TSI_win_slow)+"_"+str(self.__TSI_win_fast)] = \
                indicator_TSI.tsi().values

    def __cal_UO(self,
                 col_high: str,
                 col_low: str,
                 col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Ultimate Oscillator created by Larry Williams’ (1976) signal, 
            a momentum oscillator designed to capture momentum across three different 
            timeframes. It is calculated over "col_high" and "col_low" and "col_close", 
            passed as parameter besides 03 pairs of (window, weight) parameters creating 
            new columns named "UO"+str(window, weight) in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[[col_high,col_low,col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Ultimate Oscillator
        indicator_UO = UltimateOscillator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], window1 = self.__UO_win1,
                                          window2 = self.__UO_win2, window3 = self.__UO_win3,
                                          weight1 = self.__UO_weight1, weight2 = self.__UO_weight2, 
                                          weight3 = self.__UO_weight3 )
        field_nm = "UO_"+f'{self.__UO_win1:02d}_{self.__UO_win2:02d}_{self.__UO_win3:02d}'
        self.__data[self.__ticker+field_nm] = indicator_UO.ultimate_oscillator().values   
        
    def __cal_WRI(self,
                 col_high: str,
                 col_low: str,
                 col_close: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates William %R Indicator, developed by Larry Williams, Williams %R is 
            a momentum indicator that is the inverse of the Fast Stochastic Oscillator. 
            Also referred to as %R, Williams %R reflects the level of the close relative
            to the highest high for the look-back period. In contrast, the Stochastic
            Oscillator reflects the level of the close relative to the lowest low. 
            %R corrects for the inversion by multiplying the raw value by -100. 
            As a result, the Fast Stochastic Oscillator and Williams %R produce the
            exact same lines, only the scaling is different. Williams %R oscillates 
            from 0 to -100. Readings from 0 to -20 are considered overbought. 
            Readings from -80 to -100 are considered oversold. Unsurprisingly, 
            signals derived from the Stochastic Oscillator are also applicable 
            to Williams %R.

        	%R = (Highest High - Close)/(Highest High - Lowest Low) * -100

            Lowest Low = lowest low for the look-back period Highest High = highest 
            high for the look-back period %R is multiplied by -100 correct the 
            inversion and move the decimal. It is calculated over "col_high" 
            and "col_low" and "col_close", passed as parameter and the "lbp" which 
            represents the looking back period and creates a new columns named 
            "WRI"+str(lbp) in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[[col_high,col_low,col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Bollinger Bands Indicator
        indicator_WRI = WilliamsRIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], lbp = self.__WRI_lbp )
        self.__data[self.__ticker+"WRI_"+str(self.__WRI_lbp)] = indicator_WRI.williams_r().values   

    #####------ VOLUME Indicators -------
    def __cal_ADI(self,
                 col_high: str,
                 col_low: str,
                 col_close: str, 
                 col_volume: str ) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Accumulation/Distribution Index (ADI) which acts as leading 
            indicator of price movements. It is calculated over "col_high" 
            and "col_low", "col_close" and "col_volume" passed as parameter and 
            creates a new columns named "ADI" in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices
            col_volume (str): name of the column with the "VOLUME" data

        Returns:
            None.

        """
        values = self.__data[[col_high,col_low,col_close, col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close", "volume"]
        
        # Initialize Bollinger Bands Indicator
        indicator_ADI = AccDistIndexIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], volume = df_wrk["volume"], )
        self.__data[self.__ticker+"ADI"] = indicator_ADI.acc_dist_index().values   
     
        
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
        self.__cal_KAMA(self.__ticker+"CLOSE")
        self.__cal_PPO(self.__ticker+"CLOSE")
        self.__cal_PVO(self.__ticker+"VOLUME")
        self.__cal_ROC(self.__ticker+"CLOSE")
        self.__cal_RSI(self.__ticker+"CLOSE")
        self.__cal_StochRSI(self.__ticker+"CLOSE")
        self.__cal_SO(self.__ticker+"HIGHT", self.__ticker+"LOW", 
                      self.__ticker+"CLOSE" )
        self.__cal_AOI(self.__ticker+"HIGHT", self.__ticker+"LOW")
        self.__cal_TSI(self.__ticker+"CLOSE")
        self.__cal_UO(self.__ticker+"HIGHT", self.__ticker+"LOW", 
                      self.__ticker+"CLOSE" )
        self.__cal_WRI(self.__ticker+"HIGHT", self.__ticker+"LOW", 
                      self.__ticker+"CLOSE" )
        self.__cal_ADI(self.__ticker+"HIGHT", self.__ticker+"LOW", 
                      self.__ticker+"CLOSE", self.__ticker+"VOLUME"  )


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
        if isinstance(X, pd.Series):               
            X = X.to_frame('0')                                    
        
        self.__data = X

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
    
    def fit_transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None   ) -> pd.DataFrame(dtype=float):
        """
        Fit and Transforms the dataframe containing all variables of our financial series
            calculating the indicators available.
            (This routine is intented to maintaing sklearn Pipeline compatibility)
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate the covariance matrix.

        Returns:
            self (object)

        """

        self.fit(X)
        self.transform(X)
        
        return self.__data

