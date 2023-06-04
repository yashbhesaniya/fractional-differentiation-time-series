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
from ta.volume import AccDistIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, \
                ForceIndexIndicator, MFIIndicator, NegativeVolumeIndexIndicator, \
                OnBalanceVolumeIndicator, VolumePriceTrendIndicator, VolumeWeightedAveragePrice
from ta.trend import ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator, EMAIndicator, \
                     IchimokuIndicator, KSTIndicator, MACD, MassIndex, PSARIndicator, SMAIndicator, \
                     STCIndicator, TRIXIndicator, VortexIndicator, WMAIndicator 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import set_config
set_config(transform_output="pandas")

class Indicators(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 col_open: str,
                 col_high: str,
                 col_low: str,
                 col_close: str,
                 col_volume: str,
                 ticker: str = '',
                 norm_data: bool = False,
                 scale_method: str = "minmax",
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
                 CMF_win: int = 20,
                 EOM_win: int = 14,
                 FI_win: int = 13,
                 MFI_win: int = 14,
                 VWAP_win: int = 14,
                 ADX_win: int = 14,
                 AROON_win: int = 25,
                 CCI_win: int = 20,
                 CCI_const: float = 0.015,
                 DPO_win: int = 20,
                 EMA_win: int = 14,
                 ICHI_win1: int = 9,
                 ICHI_win2: int = 26,
                 ICHI_win3: int = 52,
                 ICHI_visual: bool = False,
                 KST_roc1: int = 10,
                 KST_roc2: int = 15,
                 KST_roc3: int = 20,
                 KST_roc4: int = 30,
                 KST_win1: int = 10,
                 KST_win2: int = 10,
                 KST_win3: int = 10,
                 KST_win4: int = 15,
                 KST_nsig: int = 9,
                 MACD_win_slow: int = 26,
                 MACD_win_fast: int = 12,
                 MACD_win_sign: int = 9,                 
                 MI_win_fast: int = 9,
                 MI_win_slow: int = 25,                 
                 PSAR_step: float = 0.02,
                 PSAR_max_step: float = 0.2,                 
                 STC_win_slow: int = 50,
                 STC_win_fast: int = 23,
                 STC_cycle: int = 10,
                 STC_sm1: int = 3,
                 STC_sm2: int = 3,
                 TRIX_win: int = 15,
                 VI_win: int = 14,
                 WMA_win: int = 9,
                 ):
        
        """
        Initialize data.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variables 
                to be scaled.
            col_open (str): column containing the "OPEN" data
            col_high (str): column containing the "HIGH" data
            col_low (str): column containing the "LOW" data
            col_close (str): column containing the "CLOSE" data
            col_volume (str): column containing the "VOLUME" data
            ticker (str): ticker of the stock
            norm_data (bool): indicate to normalize data after calculating indicators
            scale_method (str): indicates the method for scaling (minmax, standard)
            KAMA_win (int):
            KAMA_pow1 (int): 
            KAMA_pow2 (int):
            PPO_win_slow (int):
            PPO_win_fast (int):
            PPO_win_sign (int):
            PVO_win_slow (int):
            PVO_win_fast (int):
            PVO_win_sign (int):
            ROC_win (int):
            RSI_win (int):
            StRSI_win (int):
            StRSI_sm1 (int):
            StRSI_sm2 (int):
            SO_win (int):
            SO_sm (int):
            AOI_win1 (int):
            AOI_win2 (int):
            TSI_win_slow (int):
            TSI_win_fast (int):
            UO_win1 (int):
            UO_win2 (int):
            UO_win3 (int):
            UO_weight1 (float):
            UO_weight2 (float):
            UO_weight3 (float):
            WRI_lbp (int):
            CMF_win (int):
            EOM_win (int):
            FI_win (int):
            MFI_win (int):
            VWAP_win (int):
            ADX_win (int):
            AROON_win (int):
            CCI_win (int):
            CCI_const (float):
            DPO_win (int):
            EMA_win (int):
            ICHI_win1 (int):
            ICHI_win2 (int):
            ICHI_win3 (int):
            ICHI_visual (bool):
            KST_roc1 (int):
            KST_roc2 (int):
            KST_roc3 (int):
            KST_roc4 (int):
            KST_win1 (int):
            KST_win2 (int):
            KST_win3 (int):
            KST_win4 (int):
            KST_nsig (int):
            MACD_win_slow (int):
            MACD_win_fast (int):
            MACD_win_sign (int):                 
            MI_win_fast (int):
            MI_win_slow (int):                 
            PSAR_step (float):
            PSAR_max_step (float):                 
            STC_win_slow (int):
            STC_win_fast (int):
            STC_cycle (int):
            STC_sm1 (int):
            STC_sm2 (int):
            TRIX_win (int):
            VI_win (int):
            WMA_win (int):

        Returns:
            None

        """
        if (type(col_open) != str):
            raise ValueError('Indicators Class - Parameter col_open must be a valid str')

        if (type(col_high) != str):
            raise ValueError('Indicators Class - Parameter col_high must be a valid str')

        if (type(col_low) != str):
            raise ValueError('Indicators Class - Parameter col_low must be a valid str')

        if (type(col_close) != str):
            raise ValueError('Indicators Class - Parameter col_close must be a valid str')

        if (type(col_volume) != str):
            raise ValueError('Indicators Class - Parameter col_volume must be a valid str')

        self.__ticker = (ticker+'_' if ticker!='' else '')
        self.__col_open = self.__ticker + col_open
        self.__col_high = self.__ticker + col_high
        self.__col_low = self.__ticker + col_low
        self.__col_close = self.__ticker + col_close
        self.__col_volume = self.__ticker + col_volume
        self.__norm_data = norm_data
        self.__scale_method = scale_method
        
        # Loading parameters for Momentum Indicators
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
        self.__CMF_win = CMF_win
        
        # Loading parameters for Volume Indicators
        self.__EOM_win = EOM_win
        self.__FI_win = FI_win
        self.__MFI_win = MFI_win
        self.__VWAP_win = VWAP_win
        
        # Loading parameters for Trend Indicators
        self.__ADX_win = ADX_win
        self.__AROON_win = AROON_win
        self.__CCI_win = CCI_win
        self.__CCI_const = CCI_const
        self.__DPO_win = DPO_win
        self.__EMA_win = EMA_win
        self.__ICHI_win1 = ICHI_win1
        self.__ICHI_win2 = ICHI_win2
        self.__ICHI_win3 = ICHI_win3
        self.__ICHI_visual = ICHI_visual
        self.__KST_roc1 = KST_roc1
        self.__KST_roc2 = KST_roc2
        self.__KST_roc3 = KST_roc3
        self.__KST_roc4 = KST_roc4
        self.__KST_win1 = KST_win1
        self.__KST_win2 = KST_win2
        self.__KST_win3 = KST_win3
        self.__KST_win4 = KST_win4
        self.__KST_nsig = KST_nsig
        self.__MACD_win_slow = MACD_win_slow
        self.__MACD_win_fast = MACD_win_fast
        self.__MACD_win_sign = MACD_win_sign
        self.__MI_win_slow = MI_win_slow
        self.__MI_win_fast = MI_win_fast
        self.__PSAR_step = PSAR_step
        self.__PSAR_max_step = PSAR_max_step                 
        self.__STC_win_slow = STC_win_slow
        self.__STC_win_fast = STC_win_fast
        self.__STC_cycle = STC_cycle
        self.__STC_sm1 = STC_sm1
        self.__STC_sm2 = STC_sm2
        self.__TRIX_win = TRIX_win
        self.__VI_win = VI_win
        self.__WMA_win = WMA_win
            
    @property
    def data(self):
        return self.__data

    @property
    def ticker(self):
        return self.__ticker

    @property
    def scale_method(self):
        return self.__scale_method

    def __scale_data(self, 
                   col_name:str) -> pd.Series(dtype=float):
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
        values = self.__data[col_name].values.reshape(-1,1)
        if self.scale_method == "minmax":
            scaler = MinMaxScaler(feature_range=(-1,1))
        else:
            scaler = StandardScaler()
        return scaler.fit_transform(values).values

    def __cal_price_change(self) -> None:
        """
        Calculates the price change of a column passed as parameter and creates
            a new column whose name is composed by col_name+"_price_change" in 
            the data-frame passed as argument. For compatibility purposes, it 
            was added the ticker label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        price_change = self.__data[self.__col_close].diff()
        self.__data[self.__col_close+"_price_change"] = pd.Series(price_change, index = self.__data.index)
    
    def __cal_pct_change(self) -> None:
        """
        Calculates the percentual price change of a column passed as parameter 
            and creates a new column whose name is composed by col_name+"_price_change" 
            in the data-frame passed as argument. For compatibility purposes, 
            it was added the ticker label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        pct_change = self.__data[self.__col_close].pct_change()
        self.__data[self.__col_close+"_pct_change"] = pd.Series(pct_change, index = self.__data.index)
    
    def __cal_return(self) -> None:
        """
        Calculates the return of a column passed as parameter and creates a new 
            column whose name is composed by col_name+"_returns" in the data-frame 
            passed as argument. For compatibility purposes, it was added the ticker
            label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = values[idx]/values[idx-1]
        self.__data[self.__col_close+"_returns"] = pd.Series(log_returns, index = self.__data.index)
    
    def __cal_log_return(self) -> None:
        """
        Calculates the log-return of a column passed as parameter and creates a new 
            column whose name is composed by col_name+"_log_returns" in the data-frame 
            passed as argument. For compatibility purposes, it was added the ticker
            label in front of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        self.__data[self.__col_close+"_log_returns"] = pd.Series(log_returns, index = self.__data.index)
    
    def __cal_Diff(self) -> None:
        """
        Calculates the amplitude between "col_high" and "col_low" and "col_open" 
            and "col_close", passed as parameter and creates a new columns named 
            "AMPL" and "OPNCLS", respectively, in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_open,self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["open","high","low","close"]
        self.__data[self.__ticker+"AMPL"] = (df_wrk["high"]-df_wrk["low"]).values
        self.__data[self.__ticker+"OPNCLS"] = (df_wrk["close"]-df_wrk["open"]).values
    
    def __cal_RA(self, 
               rol_win1: int,
               rol_win2: int ) -> None:
        """
        Calculates the rolling standard deviations of closing prices considering 
            the windoes "rol_win1" and "rol_win2" passed as arguments and creates 
            new columns named "RA"+rol_win1 and "RA"+rol_win2, respectively, in 
            the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            rol_win1 (int): number rolling of days to calculate the first std() 
            rol_win2 (int): number rolling of days to calculate the second std() 

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        self.__data[self.__ticker+"RA_"+str(rol_win1)] = df_wrk["close"].rolling(window=rol_win1).std().values
        self.__data[self.__ticker+"RA_"+str(rol_win2)] = df_wrk["close"].rolling(window=rol_win2).std().values
    
    def __cal_MA(self, 
               rol_win1: int,
               rol_win2: int ) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        self.__data[self.__ticker+"MA_"+str(rol_win1)] = df_wrk["close"].rolling(rol_win1, min_periods=1).mean().values
        self.__data[self.__ticker+"MA_"+str(rol_win2)] = df_wrk["close"].rolling(rol_win2, min_periods=1).mean().values
    
    def __cal_VMA(self, 
                rol_win1: int,
                rol_win2: int,
                rol_win3: int ) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_volume].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["volume"]
        self.__data[self.__ticker+"V_MA_"+str(rol_win1)] = df_wrk["volume"].rolling(rol_win1, min_periods=1).mean().values
        self.__data[self.__ticker+"V_MA_"+str(rol_win2)] = df_wrk["volume"].rolling(rol_win2, min_periods=1).mean().values
        self.__data[self.__ticker+"V_MA_"+str(rol_win3)] = df_wrk["volume"].rolling(rol_win3, min_periods=1).mean().values
        
    #================================ TECHNICAL INDICATORS ====================================
    #
    # -------------------------------- Momentum Indicators ------------------------------------
    #
    def __cal_KAMA(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize KAMA Indicator
        indicator_KAMA = KAMAIndicator(close=df_wrk["close"], window = self.__KAMA_win, 
                                       pow1 = self.__KAMA_pow1, pow2 = self.__KAMA_pow2)

        field_nm = f'w{self.__KAMA_win:02d}p({self.__KAMA_pow1:02d},{self.__KAMA_pow2:02d})'
        self.__data[self.__ticker+"KAMA_"+field_nm] = indicator_KAMA.kama().values
    
    def __cal_PPO(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Percentage Price Oscilator Indicator
        indicator_PPO = PercentagePriceOscillator(close=df_wrk["close"], window_slow = self.__PPO_win_slow, 
                                       window_fast = self.__PPO_win_fast, window_sign = self.__PPO_win_sign)

        field_nm = f'w({self.__PPO_win_slow:02d},{self.__PPO_win_fast:02d})'
        self.__data[self.__ticker+"PPO_"+field_nm] = indicator_PPO.ppo().values
    
    def __cal_PVO(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_volume].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["volume"]
        
        # Initialize Percentage Volume Oscilator Indicator
        indicator_PVO = PercentageVolumeOscillator(volume = df_wrk["volume"], window_slow = self.__PVO_win_slow, 
                                       window_fast = self.__PVO_win_fast, window_sign = self.__PVO_win_sign)

        field_nm = f'w({self.__PVO_win_slow:02d},{self.__PVO_win_fast:02d})'+ \
                    f's{self.__PVO_win_sign:02d}'
        self.__data[self.__ticker+"PVO_"+field_nm] = indicator_PVO.pvo().values
        self.__data[self.__ticker+"PVOH_"+field_nm] = indicator_PVO.pvo_hist().values
        self.__data[self.__ticker+"PVOsgn_"+field_nm] = indicator_PVO.pvo_signal().values
    
    def __cal_ROC(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize ROC Indicator
        indicator_ROC = ROCIndicator(close=df_wrk["close"], window = self.__ROC_win)

        self.__data[self.__ticker+"ROC_"+str(self.__ROC_win)] = indicator_ROC.roc().values
    
    def __cal_RSI(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize RSI Indicator
        indicator_RSI = RSIIndicator(close=df_wrk["close"], window = self.__RSI_win)

        self.__data[self.__ticker+"RSI_"+str(self.__RSI_win)] = indicator_RSI.rsi().values
    
    def __cal_StochRSI(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_StRSI = StochRSIIndicator(close=df_wrk["close"], window = self.__StRSI_win,
                                            smooth1 = self.__StRSI_sm1, smooth2 = self.__StRSI_sm2)

        field_nm = f'w{self.__StRSI_win:02d}'+ \
                    f's({self.__StRSI_sm1:02d},{self.__StRSI_sm2:02d})'
        self.__data[self.__ticker+"StRSI_"+field_nm] = indicator_StRSI.stochrsi().values
        self.__data[self.__ticker+"StRSId_"+field_nm] = indicator_StRSI.stochrsi_d().values
        self.__data[self.__ticker+"StRSIk_"+field_nm] = indicator_StRSI.stochrsi_k().values
    
    def __cal_SO(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Stochastic Indicator
        indicator_SO = StochasticOscillator(high = df_wrk["high"], low = df_wrk["low"], 
                                            close=df_wrk["close"], window = self.__SO_win,
                                            smooth_window = self.__SO_sm)

        field_nm = f'w{self.__SO_win:02d}s{self.__SO_sm:02d}'
        self.__data[self.__ticker+"SO_"+field_nm] = indicator_SO.stoch().values
        self.__data[self.__ticker+"SOsgn_"+field_nm] = indicator_SO.stoch_signal().values    
     
    def __cal_AOI(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low"]
        
        # Initialize Awesome Oscillator Indicator
        indicator_AOI = AwesomeOscillatorIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                                   window1 = self.__AOI_win1, window2 = self.__AOI_win2)

        field_nm = f'w({self.__AOI_win1:02d},{self.__AOI_win2:02d})'
        self.__data[self.__ticker+"AOI_"+field_nm] = indicator_AOI.awesome_oscillator().values   
     
    def __cal_TSI(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Percentage Price Oscilator Indicator
        indicator_TSI = TSIIndicator(close=df_wrk["close"], window_slow = self.__TSI_win_slow, 
                                       window_fast = self.__TSI_win_fast )

        field_nm = f'w({self.__TSI_win_slow:02d},{self.__TSI_win_fast:02d})'
        self.__data[self.__ticker+"TSI_"+field_nm] = indicator_TSI.tsi().values

    def __cal_UO(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Ultimate Oscillator
        indicator_UO = UltimateOscillator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], window1 = self.__UO_win1,
                                          window2 = self.__UO_win2, window3 = self.__UO_win3,
                                          weight1 = self.__UO_weight1, weight2 = self.__UO_weight2, 
                                          weight3 = self.__UO_weight3 )
        field_nm = f'wi({self.__UO_win1:02d},{self.__UO_win2:02d},{self.__UO_win3:02d})'+ \
                    f'wg({self.__UO_weight1:.1f},{self.__UO_weight2:.1f},{self.__UO_weight3:.1f})'
        self.__data[self.__ticker+"UO_"+field_nm] = indicator_UO.ultimate_oscillator().values   
        
    def __cal_WRI(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Bollinger Bands Indicator
        indicator_WRI = WilliamsRIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], lbp = self.__WRI_lbp )
        self.__data[self.__ticker+"WRI_"+str(self.__WRI_lbp)] = indicator_WRI.williams_r().values   

    #
    # -------------------------------- Volume Indicators ------------------------------------
    #
    def __cal_ADI(self) -> None:
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

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close", "volume"]
        
        # Initialize Accumulation/Distribution Index Indicator
        indicator_ADI = AccDistIndexIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], volume = df_wrk["volume"] )
        self.__data[self.__ticker+"ADI"] = indicator_ADI.acc_dist_index().values   
     
    def __cal_CMF(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Chaikin Money Flow (CMF) which measures the amount of Money 
            Flow Volume over a specific period. It is calculated over "col_high" 
            and "col_low", "col_close" and "col_volume" passed as parameter and 
            creates a new columns named "CMF" in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close", "volume"]
        
        # Initialize Chaikin Money Flow Indicator
        indicator_CMF = ChaikinMoneyFlowIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], volume = df_wrk["volume"],
                                          window = self.__CMF_win)
        self.__data[self.__ticker+"CMF_"+str(self.__CMF_win)] = indicator_CMF.chaikin_money_flow().values   
     
    def __cal_EOM(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Ease of movement (EoM, EMV) which relate an asset’s price 
            change to its volume and is particularly useful for assessing the 
            strength of a trend. It is calculated over "col_high" and "col_low" 
            and "col_volume" passed as parameter and creates new columns named 
            "EOM" and "EMV" in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","volume"]
        
        # Initialize Ease of Movement Indicator
        indicator_EOM = EaseOfMovementIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          volume = df_wrk["volume"], window = self.__EOM_win)
        self.__data[self.__ticker+"EOM_"+str(self.__EOM_win)] = indicator_EOM.ease_of_movement().values   
        self.__data[self.__ticker+"EMV_"+str(self.__EOM_win)] = indicator_EOM.sma_ease_of_movement().values   
     
    def __cal_FI(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Force Index (FI) which illustrates how strong the actual buying 
            or selling pressure is. High positive values mean there is a strong 
            rising trend, and low values signify a strong downward trend. 
            It is calculated over "CLOSE" prices "VOLUME" passed in "col_close" 
            and "col_volume", respectively, besides "FI_win" which is the period
            desired. The function creates a new columns named "FI_"+FI_window
            in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close","volume"]
        
        # Initialize Force Index Indicator
        indicator_FI = ForceIndexIndicator(close=df_wrk["close"], 
                                           volume = df_wrk["volume"], 
                                           window = self.__FI_win )

        self.__data[self.__ticker+"FI_"+str(self.__FI_win)] = indicator_FI.force_index().values

    def __cal_MFI(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Money Flow Index (MFI) which uses both price and volume 
            to measure buying and selling pressure. It is positive when the 
            typical price rises (buying pressure) and negative when the typical 
            price declines (selling pressure). A ratio of positive and negative 
            money flow is then plugged into an RSI formula to create an oscillator 
            that moves between zero and one hundred. It is calculated over "col_high" 
            and "col_low", "col_close" and "col_volume" passed as parameter and 
            creates a new columns named "MFI_"+MFI_window in the data-frame 
            passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close", "volume"]
        
        # Initialize Money Flow Index Indicator
        indicator_MFI = MFIIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], volume = df_wrk["volume"],
                                          window = self.__MFI_win)
        self.__data[self.__ticker+"MFI_"+str(self.__MFI_win)] = indicator_MFI.money_flow_index().values   
     
    def __cal_NVI(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Negative Volume Index (NVI) is a cumulative indicator 
            that uses the change in volume to decide when the smart money is active. 
            The function creates a new columns named "NVI" in the data-frame 
            passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices
            col_volume (str): name of the column with the "VOLUME" data

        Returns:
            None.

        """
        values = self.__data[[self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close","volume"]
        
        # Initialize Negative Volume Index Indicator
        indicator_NVI = NegativeVolumeIndexIndicator(close=df_wrk["close"], 
                                           volume = df_wrk["volume"] )

        self.__data[self.__ticker+"NVI"] = indicator_NVI.negative_volume_index().values
        
    def __cal_OBV(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the On-balance volume (OBV) which relates price and volume in 
            the stock market. OBV is based on a cumulative total volume. 
            The function creates a new columns named "OBV" in the data-frame 
            passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close","volume"]
        
        # Initialize On-balance volume Indicator
        indicator_OBV = OnBalanceVolumeIndicator(close=df_wrk["close"], 
                                                 volume = df_wrk["volume"] )

        self.__data[self.__ticker+"OBV"] = indicator_OBV.on_balance_volume().values
        
    def __cal_VPT(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Volume-price trend (VPT) which is based on a running 
            cumulative volume that adds or substracts a multiple of the percentage 
            change in share price trend and current volume, depending upon the 
            investment’s upward or downward movements. The function creates a 
            new columns named "VPT" in the data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close","volume"]
        
        # Initialize Volume-price trend Indicator
        indicator_VPT = VolumePriceTrendIndicator(close=df_wrk["close"], 
                                                 volume = df_wrk["volume"] )

        self.__data[self.__ticker+"VPT"] = indicator_VPT.volume_price_trend().values
        
    def __cal_VWAP(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Volume Weighted Average Price (VWAP) which is equals the 
            dollar value of all trading periods divided by the total trading volume 
            for the current day. The calculation starts when trading opens and 
            ends when it closes. Because it is good for the current trading day 
            only, intraday periods and data are used in the calculation. It is 
            calculated over "col_high", "col_low", "col_close" and "col_volume" 
            passed as parameter and creates a new columns named "VWAP" in the 
            data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close, self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close", "volume"]
        
        # Initialize Volume Weighted Average Price Indicator
        indicator_VWAP = VolumeWeightedAveragePrice(high = df_wrk["high"], low = df_wrk["low"], 
                                          close=df_wrk["close"], volume = df_wrk["volume"],
                                          window = self.__VWAP_win)
        self.__data[self.__ticker+"VWAP_"+str(self.__VWAP_win)] = \
                indicator_VWAP.volume_weighted_average_price().values   
     
    #
    # -------------------------------- Trend Indicators ------------------------------------
    #
    def __cal_ADX(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Average Directional Movement Index (ADX). The Plus Directional 
            Indicator (+DI) and Minus Directional Indicator (-DI) are derived from 
            smoothed averages of these differences, and measure trend direction 
            over time. These two indicators are often referred to collectively 
            as the Directional Movement Indicator (DMI). The Average Directional 
            Index (ADX) is in turn derived from the smoothed averages of the 
            difference between +DI and -DI, and measures the strength of the trend 
            (regardless of direction) over time. Using these three indicators 
            together, chartists can determine both the direction and strength of 
            the trend. It is calculated over "col_high", "col_low" and "col_close" 
            passed as parameter and creates corresponding "ADX"+ADX_window columns 
            in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Average Directional Movement Index Indicator
        indicator_ADX = ADXIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     close=df_wrk["close"], window = self.__ADX_win)
        # self.__data[self.__ticker+"ADX_"+str(self.__ADX_win)] = indicator_ADX.adx().values   
        self.__data[self.__ticker+"ADXP_"+str(self.__ADX_win)] = indicator_ADX.adx_pos().values   
        self.__data[self.__ticker+"ADXN_"+str(self.__ADX_win)] = indicator_ADX.adx_neg().values   
    
    def __cal_Aroon(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Aroon Indicator which identify when trends are likely to 
            change direction. 
            - Aroon Up = ((N - Days Since N-day High) / N) x 100 
            - Aroon Down = ((N - Days Since N-day Low) / N) x 100 
            - Aroon Indicator = Aroon Up - Aroon Down. 
            It is calculated over "col_close" passed as parameter and creates 
            corresponding "AROON"+AROON_window columns in the data-frame passed 
            as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Aroon Indicator
        indicator_AROON = AroonIndicator(close=df_wrk["close"], window = self.__AROON_win)
        self.__data[self.__ticker+"AROOND_"+str(self.__AROON_win)] = indicator_AROON.aroon_down().values   
        self.__data[self.__ticker+"AROON_"+str(self.__AROON_win)] = indicator_AROON.aroon_indicator().values   
        self.__data[self.__ticker+"AROONU_"+str(self.__AROON_win)] = indicator_AROON.aroon_up().values   
    
    def __cal_CCI(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Commodity Channel Index (CCI) which measures the difference 
            between a security’s price change and its average price change. High 
            positive readings indicate that prices are well above their average, 
            which is a show of strength. Low negative readings indicate that prices 
            are well below their average, which is a show of weakness. It is 
            calculated over "col_high", "col_low" and "col_close" passed as 
            parameter and creates corresponding "CCI"+CCI_window columns in the 
            data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Commodity Channel Index Indicator
        indicator_CCI = CCIIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     close=df_wrk["close"], window = self.__CCI_win,
                                     constant = self.__CCI_const)
        self.__data[self.__ticker+"CCI_"+str(self.__CCI_win)] = indicator_CCI.cci().values   
    
    def __cal_DPO(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Detrended Price Oscillator (DPO) which is an indicator 
            designed to remove trend from price and make it easier to identify cycles. 
            It is calculated over "col_close" passed as parameter and creates 
            corresponding "DPO_"+DPO_window columns in the data-frame passed 
            as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Detrended Price Oscillator Indicator
        indicator_DPO = DPOIndicator(close=df_wrk["close"], window = self.__DPO_win)
        self.__data[self.__ticker+"DPO_"+str(self.__DPO_win)] = indicator_DPO.dpo().values   
    
    def __cal_EMA(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Exponential Moving Average (EMA). It is calculated over 
            "col_close" passed as parameter and creates corresponding 
            "EMA_"+EMA_window columns in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Exponential Moving Average Indicator
        indicator_EMA = EMAIndicator(close=df_wrk["close"], window = self.__EMA_win)
        self.__data[self.__ticker+"EMA_"+str(self.__EMA_win)] = indicator_EMA.ema_indicator().values   
    
    def __cal_SMA(self,
                 SMA_win: int) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the SMA - Simple Moving Average. It is calculated over 
            "col_close" passed as parameter and creates corresponding 
            "SMA_"+SMA_window columns in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Simple Moving Average Indicator
        indicator_SMA = SMAIndicator(close=df_wrk["close"], window = SMA_win)
        self.__data[self.__ticker+"SMA_"+str(SMA_win)] = indicator_SMA.sma_indicator().values   
    
    def __cal_Ichimoku(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Ichimoku Cloud, also known as Ichimoku Kinko Hyo, which is 
            a versatile indicator that defines support and resistance, identifies 
            trend direction, gauges momentum and provides trading signals. 
            Ichimoku Kinko Hyo translates into “one look equilibrium chart”. 
            With one look, chartists can identify the trend and look for potential 
            signals within that trend. It is calculated over "col_high" and "col_low", 
            and 03 window parameters, and creates corresponding "ICHI"+ICHI_window(1,2,3)
            columns in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low"]
        
        # Initialize Ichimoku Cloud Indicator
        indicator_ICHI = IchimokuIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     window1 = self.__ICHI_win1, window2 = self.__ICHI_win2,  
                                     window3 = self.__ICHI_win3, visual = self.__ICHI_visual)
        field_nm = f'w({self.__ICHI_win1:02d},{self.__ICHI_win2:02d},{self.__ICHI_win3:02d})'
        self.__data[self.__ticker+"ICHIA_"+field_nm] = indicator_ICHI.ichimoku_a().values   
        self.__data[self.__ticker+"ICHIB_"+field_nm] = indicator_ICHI.ichimoku_b().values   
        self.__data[self.__ticker+"ICHIBL_"+field_nm] = indicator_ICHI.ichimoku_base_line().values   
        self.__data[self.__ticker+"ICHICL_"+field_nm] = indicator_ICHI.ichimoku_conversion_line().values   
    
    def __cal_KST(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the KST Oscillator (KST Signal) which is useful to identify 
            major stock market cycle junctures because its formula is weighed to 
            be more greatly influenced by the longer and more dominant time spans, 
            in order to better reflect the primary swings of stock market cycle. 
            It is calculated over "col_close" passed as parameter, 04 roc parameters, 
            04 window parameters and a "sig" parameter, creating a corresponding 
            "KST"+KST_roc(1,2,3,4)+KST_window(1,2,3,4) columns in the data-frame 
            passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize KST Oscillator Indicator
        indicator_KST = KSTIndicator(close=df_wrk["close"], roc1 = self.__KST_roc1,
                                     roc2 = self.__KST_roc2, roc3 = self.__KST_roc3,
                                     roc4 = self.__KST_roc4, window1 = self.__KST_win1,
                                     window2 = self.__KST_win2, window3 = self.__KST_win3,
                                     window4 = self.__KST_win4, nsig = self.__KST_nsig)
        field_rocnm = f'r({self.__KST_roc1:02d},{self.__KST_roc2:02d},{self.__KST_roc3:02d},{self.__KST_roc4:02d})'
        field_winnm = f'w({self.__KST_win1:02d},{self.__KST_win2:02d},{self.__KST_win3:02d},{self.__KST_win4:02d})'
        self.__data[self.__ticker+"KST_"+field_rocnm+"_"+field_winnm] = indicator_KST.kst().values   
        self.__data[self.__ticker+"KSTD_"+field_rocnm+"_"+field_winnm] = indicator_KST.kst_diff().values   
        self.__data[self.__ticker+"KSTS_"+field_rocnm+"_"+field_winnm] = indicator_KST.kst_sig().values   
    
    def __cal_MACD(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Moving Average Convergence Divergence (MACD) which is a 
            trend-following momentum indicator that shows the relationship between 
            two moving averages of prices. It is calculated over "col_close" 
            passed as parameter, 02 window parameters (slow, fast) and a "sign" 
            parameter, creating a corresponding "MACD"+MACD_window(slow, fast) 
            columns in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Moving Average Convergence Divergence (MACD) Indicator
        indicator_MACD = MACD(close=df_wrk["close"], window_slow = self.__MACD_win_slow,
                                     window_fast = self.__MACD_win_fast, window_sign = self.__MACD_win_sign )
        field_nm = f'w({self.__MACD_win_slow:02d},{self.__MACD_win_fast:02d},{self.__MACD_win_sign:02d})'
        self.__data[self.__ticker+"MACD_"+field_nm] = indicator_MACD.macd().values   
        self.__data[self.__ticker+"MACDD_"+field_nm] = indicator_MACD.macd_diff().values   
        self.__data[self.__ticker+"MACDS_"+field_nm] = indicator_MACD.macd_signal().values   
    
    def __cal_MI(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Mass Index (MI) whch uses the high-low range to identify 
            trend reversals based on range expansions. It identifies range bulges 
            that can foreshadow a reversal of the current trend. It is calculated 
            over "col_high" and "col_low", and 02 window parameters, creating corresponding 
            "MI_"+MI_window(fast, slow) columns in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low"]
        
        # Initialize Mass Index Indicator
        indicator_MI = MassIndex(high = df_wrk["high"], low = df_wrk["low"], 
                                     window_fast = self.__MI_win_fast, 
                                     window_slow = self.__MI_win_slow)
        field_nm = f'w({self.__MI_win_fast:02d},{self.__MI_win_slow:02d})'
        self.__data[self.__ticker+"MI_"+field_nm] = indicator_MI.mass_index().values   
    
    def __cal_PSAR(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Parabolic Stop and Reverse (Parabolic SAR). The Parabolic 
            Stop and Reverse, more commonly known as the Parabolic SAR,is a 
            trend-following indicator developed by J. Welles Wilder. The Parabolic 
            SAR is displayed as a single parabolic line (or dots) underneath the 
            price bars in an uptrend, and above the price bars in a downtrend. It is 
            calculated over "col_high", "col_low" and "col_close" passed as 
            parameter and creates corresponding "ADX"+ADX_window columns in the 
            data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Parabolic Stop and Reverse Indicator
        indicator_PSAR = PSARIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     close=df_wrk["close"], step = self.__PSAR_step,
                                     max_step = self.__PSAR_max_step, fillna = True)
        field_nm = f's({self.__PSAR_step:.2f},{self.__PSAR_max_step:.1f})'
        self.__data[self.__ticker+"PSAR_"+field_nm] = indicator_PSAR.psar().values   
        self.__data[self.__ticker+"PSARD_"+field_nm] = indicator_PSAR.psar_down().values   
        self.__data[self.__ticker+"PSARDI_"+field_nm] = indicator_PSAR.psar_down_indicator().values   
        self.__data[self.__ticker+"PSARU_"+field_nm] = indicator_PSAR.psar_up().values   
        self.__data[self.__ticker+"PSARUI_"+field_nm] = indicator_PSAR.psar_up_indicator().values   
    
    def __cal_STC(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Schaff Trend Cycle (STC) which is a charting indicator that 
            is commonly used to identify market trends and provide buy and sell 
            signals to traders. Developed in 1999 by noted currency trader Doug 
            Schaff, STC is a type of oscillator and is based on the assumption 
            that, regardless of time frame, currency trends accelerate and decelerate 
            in cyclical patterns. It is calculated over "CLOSE" prices passed in 
            "col_close", "STC_win_slow", "STC_win_fast", "STC_sm1" and "STC_sm2" 
            creating new columns named "STC"+STC_(slow, fast)_(sm1, sm2) in the 
            data-frame passed as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_STC = STCIndicator(close=df_wrk["close"], window_slow = self.__STC_win_slow,
                                     window_fast = self.__STC_win_fast, cycle = self.__STC_cycle, 
                                     smooth1 = self.__STC_sm1, smooth2 = self.__STC_sm2)

        field_nm = f'w({self.__STC_win_slow:02d},{self.__STC_win_fast:02d})'+ \
                    f'c{self.__STC_cycle:1d}'+ \
                    f's({self.__STC_sm1:02d},{self.__STC_sm2:02d})'
        self.__data[self.__ticker+"STC_"+field_nm] = indicator_STC.stc().values
    
    def __cal_TRIX(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Trix (TRIX) that shows the percent rate of change of a triple 
            exponentially smoothed moving average. It is designed to filter out 
            insignificant price movements. Chartists can use TRIX to generate 
            signals similar to MACD. A signal line can be applied to look for signal 
            line crossovers. A directional bias can be determined with the absolute 
            level. Bullish and bearish divergences can be used to anticipate reversals.
            It is calculated over "CLOSE" prices passed in "col_close", "TRIX_win" 
            creating new columns named "TRIX_"+TRIX_win in the data-frame passed 
            as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        
            col_close (str): name of the column with the "CLOSE" data prices

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_TRIX = TRIXIndicator(close=df_wrk["close"], window = self.__TRIX_win )

        field_nm = f'w({self.__TRIX_win:02d})'
        self.__data[self.__ticker+"TRIX_"+field_nm] = indicator_TRIX.trix().values
        
    def __cal_VI(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the Vortex Indicator (VI) which consists of two oscillators 
            that capture positive and negative trend movement. A bullish signal 
            triggers when the positive trend indicator crosses above the negative 
            trend indicator or a key level. It is calculated over "col_high", 
            "col_low" and "col_close", and "VI_window" passed as parameter and 
            creates corresponding "VI_"+VI_window columns in the data-frame 
            passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Vortex Indicator
        indicator_VI = VortexIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     close=df_wrk["close"], window = self.__VI_win )
        field_nm = f'w({self.__VI_win:02d})'
        self.__data[self.__ticker+"VI_"+field_nm] = indicator_VI.vortex_indicator_diff().values   
        self.__data[self.__ticker+"VIN_"+field_nm] = indicator_VI.vortex_indicator_neg().values   
        self.__data[self.__ticker+"VIP_"+field_nm] = indicator_VI.vortex_indicator_pos().values   

    def __cal_WMA(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates WMA - Weighted Moving Average. It is calculated over "CLOSE" 
            prices passed in "col_close", "WMA_win" creating new columns named 
            "WMA_"+WMA_win in the data-frame passed  as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_WMA = WMAIndicator(close=df_wrk["close"], window = self.__WMA_win )

        field_nm = f'w({self.__WMA_win:02d})'
        self.__data[self.__ticker+"WMA_"+field_nm] = indicator_WMA.wma().values
        
    #========================================================================================
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
        self.__cal_return()
        self.__cal_log_return()
        self.__cal_price_change()
        self.__cal_pct_change()
        self.__cal_RA(5, 10)
        self.__cal_Diff()
        self.__cal_MA(5, 10)
        self.__cal_VMA(5, 10, 20)
        self.__cal_KAMA()
        self.__cal_PPO()
        self.__cal_PVO()
        self.__cal_ROC()
        self.__cal_RSI()
        self.__cal_StochRSI()
        self.__cal_SO()
        self.__cal_AOI()
        self.__cal_TSI()
        self.__cal_UO()
        self.__cal_WRI()
        self.__cal_ADI()
        self.__cal_CMF()
        self.__cal_EOM()
        self.__cal_FI()
        self.__cal_MFI()
        self.__cal_NVI()
        self.__cal_OBV()
        self.__cal_VPT()
        self.__cal_VWAP()
        self.__cal_ADX()
        self.__cal_Aroon()
        self.__cal_CCI()
        self.__cal_DPO()
        self.__cal_EMA()
        self.__cal_Ichimoku()
        self.__cal_KST()
        self.__cal_MACD()
        self.__cal_MI()
        self.__cal_PSAR()
        self.__cal_STC()
        self.__cal_TRIX()
        self.__cal_VI()
        self.__cal_WMA()

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
        col_names = self.__data.columns
        for col in col_names:
            self.__data["N_"+col] = self.__scale_data(col)
                

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

