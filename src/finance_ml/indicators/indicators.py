"""
Created on Thu May 11 19:00:00 2023

@author: Luis Alvaro Correia

Updated: July 27th
    1. Removed dscription of 'self' parameter on function's documentation
    2. Updated function '__cal_SMA()' to receive parameter via 'self'
    3. Updated function 'calculate_indicators()' to calculate SMA indicator.
    4. Optimized function 'normalize_data()' to process the whole block of indicators at once.
    5. Removed the function '__scale_data()' transferring its duties to 'normalize_data()'
    
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

import pandas_ta as pta

class Indicators(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 col_open: str = 'OPEN',
                 col_high: str = 'HIGHT',
                 col_low: str = 'LOW',
                 col_close: str = 'CLOSE',
                 col_volume: str = 'VOLUME',
                 ticker: str = '',
                 calc_all: bool = True,
                 list_ind: list = [],
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
                 SMA_win: int = 14,
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
                 HMA_win: int = 9
                 ):
        
        """
        Initialize data.
            
        Args:
            data (pd.DataFrame): Columns of dataframe containing the variables 
                to be scaled.
            col_open (str): column containing the "OPEN" data
            col_high (str): column containing the "HIGH" data
            col_low (str): column containing the "LOW" data
            col_close (str): column containing the "CLOSE" data
            col_volume (str): column containing the "VOLUME" data
            ticker (str): ticker of the stock
            calc_all (bool): if True, calc all indicators
            list_ind (list): list of indicators do calculate if calc_all is False
            norm_data (bool): indicate to normalize data after calculating indicators
            scale_method (str): indicates the method for scaling (minmax, standard)
            KAMA_win (int): n period for Kaufman's Adaptative Moving Average
            KAMA_pow1 (int): number of periods for the fastest EMA constant
            KAMA_pow2 (int): number of periods for the slowest EMA constant
            PPO_win_slow (int): n period long-term for Percentage Price Oscillator
            PPO_win_fast (int): n period short-term for Percentage Price Oscillator
            PPO_win_sign (int): n period to signal for Percentage Price Oscillator
            PVO_win_slow (int): n period long-term for Percentage Volume Oscillator
            PVO_win_fast (int): n period short-term for Percentage Volume Oscillator
            PVO_win_sign (int): n period to signal for Percentage Volume Oscillator
            ROC_win (int): n period for Rate of Change 
            RSI_win (int): n period for Relative Strength Index
            StRSI_win (int): n period for Stochastic RSI
            StRSI_sm1 (int): moving average of Stochastic RSI
            StRSI_sm2 (int): moving average of %K
            SO_win (int): n period for Stochastic Oscillator
            SO_sm (int): sma period over 
            AOI_win1 (int): short period for Awesome Oscillator
            AOI_win2 (int): long period for Awesome Oscillator
            TSI_win_slow (int): high period of True Strength Index
            TSI_win_fast (int): low period of True Strength Index
            UO_win1 (int): short period of Ultimate Oscillator
            UO_win2 (int): medium period of Ultimate Oscillator
            UO_win3 (int): long period of Ultimate Oscillator
            UO_weight1 (float): weight of short BP average for Ultimate Oscillator
            UO_weight2 (float): weight of medium BP average for Ultimate Oscillator
            UO_weight3 (float): weight of long BP average for Ultimate Oscillator
            WRI_lbp (int): lookback period for William %R
            CMF_win (int): n period for Chaikin Money Flow
            EOM_win (int): n period for Ease Of Movement
            FI_win (int): n period for Force Index
            MFI_win (int): n period for Money Flow Index
            VWAP_win (int): n period for Volume Weighted Average Price 
            ADX_win (int): n period for Average Directional Movement Index
            AROON_win (int): n period for Aroon Indicator
            CCI_win (int): n period for Commodity Channel Index 
            CCI_const (float): constant for Commodity Channel Index
            DPO_win (int): n period for Detrended Price Oscillator
            EMA_win (int): n period for EMA - Exponential Moving Average
            SMA_win (int): n period for SMA - Simple Moving Average
            ICHI_win1 (int): n1 low period for Ichimoku Kinko Hyo indicator
            ICHI_win2 (int): n2 medium period for Ichimoku Kinko Hyo indicator
            ICHI_win3 (int): n3 high period for Ichimoku Kinko Hyo indicator
            ICHI_visual (bool): if True, shift n2 values
            KST_roc1 (int): roc1 period for KST Oscillator
            KST_roc2 (int): roc2 period for KST Oscillator
            KST_roc3 (int): roc3 period for KST Oscillator
            KST_roc4 (int): roc4 period for KST Oscillator
            KST_win1 (int): n1 smoothed period for KST Oscillator
            KST_win2 (int): n2 smoothed period for KST Oscillator
            KST_win3 (int): n3 smoothed period for KST Oscillator
            KST_win4 (int): n4 smoothed period for KST Oscillator
            KST_nsig (int): n period to signal for KST Oscillator
            MACD_win_slow (int): n period short-term for Moving Average Convergence Divergence
            MACD_win_fast (int): n period long-term for Moving Average Convergence Divergence
            MACD_win_sign (int): n period signal for Moving Average Convergence Divergence
            MI_win_fast (int): fast period value for Mass index
            MI_win_slow (int): slow period value for Mass index
            PSAR_step (float): the Acceleration Factor used to compute the SAR
            PSAR_max_step (float): the maximum value allowed for the Acceleration Factor
            STC_win_slow (int): n period long-term for Schaff Trend Cycle
            STC_win_fast (int): n period short-term for Schaff Trend Cycle
            STC_cycle (int): cycle size for Schaff Trend Cycle
            STC_sm1 (int): ema period over stoch_k for Shaff Trend Cycle
            STC_sm2 (int): ema period over stoch_kd for Shaff Trend Cycle
            TRIX_win (int): n period for Trix Indicator
            VI_win (int): n period for Vortex Indicator
            WMA_win (int): n period for Weighted Moving Average
            HMA_win (int): n period for Hull Moving Average

        Returns:
            None

        """
        IND_LIST = ['RET', 'LRET', 'PCHG','PCTCHG', 'RA', 'DIFF', 'MA', 'VMA', \
                    'KAMA', 'PPO', 'PVO', 'ROC', 'RSI', 'STRSI', 'SO', 'AOI', \
                    'TSI', 'UO', 'WRI', 'ADI', 'CMF', 'EOM', 'FI', 'MFI', 'NVI', \
                    'OBV', 'VPT', 'VWAP', 'ADX', 'AROON', 'CCI', 'DPO', 'EMA', 'SMA', \
                    'ICHI', 'KST', 'MACD', 'MI', 'PSAR', 'STC', 'TRIX', 'VI', 'WMA', 'HMA']
        SCALE_METHODS = ['MINMAX', 'STANDARD']

        if (type(calc_all) != bool):
            raise ValueError('Indicators Class - Parameter calc_all must be True or False')
        
        if (type(list_ind) != list):
            raise ValueError('Indicators Class - Parameter list_ind must be a list')
            
        list_ind = [l.upper() for l in list_ind]
        
        if (not set(list_ind).issubset(IND_LIST)):
            raise ValueError(f'Indicators Class - Invalid Indicator {set(list_ind)-set(IND_LIST)}')
        
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

        if (type(norm_data) != bool):
            raise ValueError('Indicators Class - Parameter norm_data must be True or False')

        if (scale_method.upper() not in SCALE_METHODS):
            raise ValueError(f'Indicators Class - Invalid Scale Method ({scale_method})')

        if (type(KAMA_win) != int) | (KAMA_win <= 0):
            raise ValueError('Indicators Class - Parameter KAMA_win must be int, positive')

        if (type(KAMA_pow1) != int) | (KAMA_pow1 <= 0):
            raise ValueError('Indicators Class - Parameter KAMA_pow1 must be int, positive')

        if (type(KAMA_pow2) != int) | (KAMA_pow2 <= 0) | (KAMA_pow2 <= KAMA_pow1):
            raise ValueError('Indicators Class - Parameter KAMA_pow2 must be int, positive, greater than KAMA_pow1')

        if (type(PPO_win_slow) != int) | (PPO_win_slow <= 0):
            raise ValueError('Indicators Class - Parameter PPO_win_slow must be int, positive')

        if (type(PPO_win_fast) != int) | (PPO_win_fast <= 0) | (PPO_win_fast >= PPO_win_slow):
            raise ValueError('Indicators Class - Parameter PPO_win_fast must be int, positive, less than PPO_win_slow')

        if (type(PPO_win_sign) != int) | (PPO_win_sign <= 0):
            raise ValueError('Indicators Class - Parameter PPO_win_sign must be int, positive')

        if (type(PVO_win_slow) != int) | (PVO_win_slow <= 0):
            raise ValueError('Indicators Class - Parameter PVO_win_slow must be int, positive')

        if (type(PVO_win_fast) != int) | (PVO_win_fast <= 0) | (PVO_win_fast >= PVO_win_slow):
            raise ValueError('Indicators Class - Parameter PVO_win_fast must be int, positive, less than PVO_win_slow')

        if (type(PVO_win_sign) != int) | (PVO_win_sign <= 0):
            raise ValueError('Indicators Class - Parameter PVO_win_sign must be int, positive')

        if (type(ROC_win) != int) | (ROC_win <= 0):
            raise ValueError('Indicators Class - Parameter ROC_win must be int, positive')

        if (type(RSI_win) != int) | (RSI_win <= 0):
            raise ValueError('Indicators Class - Parameter RSI_win must be int, positive')

        if (type(StRSI_win) != int) | (StRSI_win <= 0):
            raise ValueError('Indicators Class - Parameter StRSI_win must be int, positive')

        if (type(StRSI_sm1) != int) | (StRSI_sm1 <= 0):
            raise ValueError('Indicators Class - Parameter StRSI_sm1 must be int, positive')

        if (type(StRSI_sm2) != int) | (StRSI_sm2 <= 0):
            raise ValueError('Indicators Class - Parameter RSI_win must be int, positive')

        if (type(SO_win) != int) | (SO_win <= 0):
            raise ValueError('Indicators Class - Parameter SO_win must be int, positive')

        if (type(SO_sm) != int) | (SO_sm <= 0):
            raise ValueError('Indicators Class - Parameter SO_sm must be int, positive')

        if (type(AOI_win1) != int) | (AOI_win1 <= 0):
            raise ValueError('Indicators Class - Parameter AOI_win1 must be int, positive')

        if (type(AOI_win2) != int) | (AOI_win2 <= 0) | (AOI_win2 <= AOI_win1):
            raise ValueError('Indicators Class - Parameter AOI_win2 must be int, positive, greater than AOI_win1')

        if (type(TSI_win_slow) != int) | (TSI_win_slow <= 0):
            raise ValueError('Indicators Class - Parameter TSI_win_slow must be int, positive')

        if (type(TSI_win_fast) != int) | (TSI_win_fast <= 0) | (TSI_win_fast >= TSI_win_slow):
            raise ValueError('Indicators Class - Parameter TSI_win_fast must be int, positive, less than TSI_win_slow')

        if (type(UO_win1) != int) | (UO_win1 <= 0):
            raise ValueError('Indicators Class - Parameter UO_win1 must be int, positive')

        if (type(UO_win2) != int) | (UO_win2 <= 0) | (UO_win2 <= UO_win1):
            raise ValueError('Indicators Class - Parameter UO_win2 must be int, positive, greater than UO_win1')

        if (type(UO_win3) != int) | (UO_win3 <= 0) | (UO_win3 <= UO_win2):
            raise ValueError('Indicators Class - Parameter UO_win3 must be int, positive, greater than UO_win2')

        if (type(UO_weight1) != float) | (UO_weight1 <= 0.0):
            raise ValueError('Indicators Class - Parameter UO_weight1 must be float, positive')

        if (type(UO_weight2) != float) | (UO_weight2 <= 0.0):
            raise ValueError('Indicators Class - Parameter UO_weight2 must be float, positive')

        if (type(UO_weight3) != float) | (UO_weight3 <= 0.0):
            raise ValueError('Indicators Class - Parameter UO_weight3 must be float, positive')

        if (type(WRI_lbp) != int) | (WRI_lbp <= 0):
            raise ValueError('Indicators Class - Parameter WRI_lbp must be int, positive')

        if (type(CMF_win) != int) | (CMF_win <= 0):
            raise ValueError('Indicators Class - Parameter CMF_win must be int, positive')

        if (type(EOM_win) != int) | (EOM_win <= 0):
            raise ValueError('Indicators Class - Parameter EOM_win must be int, positive')

        if (type(FI_win) != int) | (FI_win <= 0):
            raise ValueError('Indicators Class - Parameter FI_win must be int, positive')

        if (type(MFI_win) != int) | (MFI_win <= 0):
            raise ValueError('Indicators Class - Parameter MFI_win must be int, positive')

        if (type(VWAP_win) != int) | (VWAP_win <= 0):
            raise ValueError('Indicators Class - Parameter VWAP_win must be int, positive')

        if (type(ADX_win) != int) | (ADX_win <= 0):
            raise ValueError('Indicators Class - Parameter ADX_win must be int, positive')

        if (type(AROON_win) != int) | (AROON_win <= 0):
            raise ValueError('Indicators Class - Parameter AROON_win must be int, positive')

        if (type(CCI_win) != int) | (CCI_win <= 0):
            raise ValueError('Indicators Class - Parameter CCI_win must be int, positive')

        if (type(CCI_const) != float) | (CCI_const <= 0.0):
            raise ValueError('Indicators Class - Parameter CCI_const must be float, positive')

        if (type(DPO_win) != int) | (DPO_win <= 0):
            raise ValueError('Indicators Class - Parameter DPO_win must be int, positive')

        if (type(EMA_win) != int) | (EMA_win <= 0):
            raise ValueError('Indicators Class - Parameter EMA_win must be int, positive')

        if (type(SMA_win) != int) | (SMA_win <= 0):
            raise ValueError('Indicators Class - Parameter SMA_win must be int, positive')

        if (type(ICHI_win1) != int) | (ICHI_win1 <= 0):
            raise ValueError('Indicators Class - Parameter ICHI_win1 must be int, positive')

        if (type(ICHI_win2) != int) | (ICHI_win2 <= 0) | (ICHI_win2 <= ICHI_win1):
            raise ValueError('Indicators Class - Parameter ICHI_win2 must be int, positive, greater than ICHI_win1')

        if (type(ICHI_win3) != int) | (ICHI_win3 <= 0) | (ICHI_win3 <= ICHI_win2):
            raise ValueError('Indicators Class - Parameter ICHI_win3 must be int, positive, greater than ICHI_win2')

        if (type(ICHI_visual) != bool):
            raise ValueError('Indicators Class - Parameter ICHI_visual must be True or False')

        if (type(KST_roc1) != int) | (KST_roc1 <= 0):
            raise ValueError('Indicators Class - Parameter KST_roc1 must be int, positive')

        if (type(KST_roc2) != int) | (KST_roc2 <= 0) | (KST_roc2 <= KST_roc1):
            raise ValueError('Indicators Class - Parameter KST_roc2 must be int, positive, greater than KST_roc1')

        if (type(KST_roc3) != int) | (KST_roc3 <= 0) | (KST_roc3 <= KST_roc2):
            raise ValueError('Indicators Class - Parameter KST_roc3 must be int, positive, greater than KST_roc2')

        if (type(KST_roc4) != int) | (KST_roc4 <= 0) | (KST_roc4 <= KST_roc3):
            raise ValueError('Indicators Class - Parameter KST_roc4 must be int, positive, greater than KST_roc3')

        if (type(KST_win1) != int) | (KST_win1 <= 0):
            raise ValueError('Indicators Class - Parameter KST_win1 must be int, positive')

        if (type(KST_win2) != int) | (KST_win2 <= 0):
            raise ValueError('Indicators Class - Parameter KST_win2 must be int, positive')

        if (type(KST_win3) != int) | (KST_win3 <= 0):
            raise ValueError('Indicators Class - Parameter KST_win3 must be int, positive')

        if (type(KST_win4) != int) | (KST_win4 <= 0):
            raise ValueError('Indicators Class - Parameter KST_win4 must be int, positive')

        if (type(KST_nsig) != int) | (KST_nsig <= 0):
            raise ValueError('Indicators Class - Parameter KST_nsig must be int, positive')

        if (type(MACD_win_slow) != int) | (MACD_win_slow <= 0):
            raise ValueError('Indicators Class - Parameter MACD_win_slow must be int, positive')

        if (type(MACD_win_fast) != int) | (MACD_win_fast <= 0) | (MACD_win_fast >= MACD_win_slow):
            raise ValueError('Indicators Class - Parameter MACD_win_fast must be int, positive, less than MACD_win_slow')

        if (type(MACD_win_sign) != int) | (MACD_win_sign <= 0):
            raise ValueError('Indicators Class - Parameter MACD_win_sign must be int, positive')

        if (type(MI_win_fast) != int) | (MI_win_fast <= 0):
            raise ValueError('Indicators Class - Parameter MI_win_fast must be int, positive')

        if (type(MI_win_slow) != int) | (MI_win_slow <= 0) | (MI_win_slow <= MI_win_fast):
            raise ValueError('Indicators Class - Parameter MI_win_slow must be int, positive, greater than MI_win_fast')

        if (type(PSAR_step) != float) | (PSAR_step <= 0.0):
            raise ValueError('Indicators Class - Parameter PSAR_step must be float, positive')

        if (type(PSAR_max_step) != float) | (PSAR_max_step <= 0.0):
            raise ValueError('Indicators Class - Parameter PSAR_max_step must be float, positive')

        if (type(STC_win_slow) != int) | (STC_win_slow <= 0):
            raise ValueError('Indicators Class - Parameter STC_win_slow must be int, positive')

        if (type(STC_win_fast) != int) | (STC_win_fast <= 0) | (STC_win_fast >= STC_win_slow):
            raise ValueError('Indicators Class - Parameter STC_win_fast must be int, positive, less than STC_win_slow')

        if (type(STC_cycle) != int) | (STC_cycle <= 0):
            raise ValueError('Indicators Class - Parameter STC_cycle must be int, positive')

        if (type(STC_sm1) != int) | (STC_sm1 <= 0):
            raise ValueError('Indicators Class - Parameter STC_sm1 must be int, positive')

        if (type(STC_sm2) != int) | (STC_sm2 <= 0):
            raise ValueError('Indicators Class - Parameter STC_sm2 must be int, positive')

        if (type(TRIX_win) != int) | (TRIX_win <= 0):
            raise ValueError('Indicators Class - Parameter TRIX_win must be int, positive')

        if (type(VI_win) != int) | (VI_win <= 0):
            raise ValueError('Indicators Class - Parameter VI_win must be int, positive')

        if (type(WMA_win) != int) | (WMA_win <= 0):
            raise ValueError('Indicators Class - Parameter WMA_win must be int, positive')
        
        if (type(HMA_win) != int) | (HMA_win <= 0):
            raise ValueError('Indicators Class - Parameter HMA_win must be int, positive')

        self.__ticker = (ticker+'_' if ticker!='' else '')
        self.__col_open = self.__ticker + col_open
        self.__col_high = self.__ticker + col_high
        self.__col_low = self.__ticker + col_low
        self.__col_close = self.__ticker + col_close
        self.__col_volume = self.__ticker + col_volume
        self.__norm_data = norm_data
        self.__scale_method = scale_method
        self.__calc_all = calc_all
        self.__list_ind = list_ind
        
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
        self.__SMA_win = SMA_win
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
        self.__HMA_win = HMA_win
            
    @property
    def data(self) -> pd.DataFrame(dtype=float):
        return self.__data

    @property
    def data_norm(self) -> pd.DataFrame(dtype=float):
        return self.__data_norm

    @property
    def ticker(self) -> str:
        return self.__ticker

    @property
    def scale_method(self):
        return self.__scale_method

    def __cal_price_change(self) -> None:
        """
        Calculates the price change of a column passed as parameter and creates
            a new column whose name is composed by col_name+"_price_change" in 
            the data-frame passed as argument. For compatibility purposes, it 
            was added the ticker label in front of all columns created.
            
        Args:
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize ROC Indicator
        indicator_ROC = ROCIndicator(close=df_wrk["close"], window = self.__ROC_win)
        
        field_nm = f'w{self.__ROC_win:02d}'
        self.__data[self.__ticker+"ROC_"+field_nm] = indicator_ROC.roc().values
    
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
            None.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize RSI Indicator
        indicator_RSI = RSIIndicator(close=df_wrk["close"], window = self.__RSI_win)

        field_nm = f'w{self.__RSI_win:02d}'
        self.__data[self.__ticker+"RSI_"+field_nm] = indicator_RSI.rsi().values
    
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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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

        field_nm = f'w{self.__CMF_win:02d}'
        self.__data[self.__ticker+"CMF_"+field_nm] = indicator_CMF.chaikin_money_flow().values   
     
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
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_volume]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","volume"]
        
        # Initialize Ease of Movement Indicator
        indicator_EOM = EaseOfMovementIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                          volume = df_wrk["volume"], window = self.__EOM_win)

        field_nm = f'w{self.__EOM_win:02d}'
        self.__data[self.__ticker+"EOM_"+field_nm] = indicator_EOM.ease_of_movement().values   
        self.__data[self.__ticker+"EMV_"+field_nm] = indicator_EOM.sma_ease_of_movement().values   
     
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
            None.        

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

        field_nm = f'w{self.__FI_win:02d}'
        self.__data[self.__ticker+"FI_"+field_nm] = indicator_FI.force_index().values

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
            None.        

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

        field_nm = f'w{self.__MFI_win:02d}'
        self.__data[self.__ticker+"MFI_"+field_nm] = indicator_MFI.money_flow_index().values   
     
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
            None.        

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
            None.        

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
            None.        

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
        field_nm = f'w{self.__VWAP_win:02d}'
        self.__data[self.__ticker+"VWAP_"+field_nm] = indicator_VWAP.volume_weighted_average_price().values   
     
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
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Average Directional Movement Index Indicator
        indicator_ADX = ADXIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     close=df_wrk["close"], window = self.__ADX_win)

        field_nm = f'w{self.__ADX_win:02d}'
        # self.__data[self.__ticker+"ADX_"+field_nm] = indicator_ADX.adx().values   
        self.__data[self.__ticker+"ADXP_"+field_nm] = indicator_ADX.adx_pos().values   
        self.__data[self.__ticker+"ADXN_"+field_nm] = indicator_ADX.adx_neg().values   
    
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
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Aroon Indicator
        indicator_AROON = AroonIndicator(low=df_wrk["low"],high = df_wrk["high"], window = self.__AROON_win)

        field_nm = f'w{self.__AROON_win:02d}'
        self.__data[self.__ticker+"AROOND_"+field_nm] = indicator_AROON.aroon_down().values   
        self.__data[self.__ticker+"AROON_"+field_nm] = indicator_AROON.aroon_indicator().values   
        self.__data[self.__ticker+"AROONU_"+field_nm] = indicator_AROON.aroon_up().values   
    
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
            None.        

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

        field_nm = f'w{self.__CCI_win:02d}'
        self.__data[self.__ticker+"CCI_"+field_nm] = indicator_CCI.cci().values   
    
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
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Detrended Price Oscillator Indicator
        indicator_DPO = DPOIndicator(close=df_wrk["close"], window = self.__DPO_win)

        field_nm = f'w{self.__DPO_win:02d}'
        self.__data[self.__ticker+"DPO_"+field_nm] = indicator_DPO.dpo().values   
    
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
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Exponential Moving Average Indicator
        indicator_EMA = EMAIndicator(close=df_wrk["close"], window = self.__EMA_win)

        field_nm = f'w{self.__EMA_win:02d}'
        self.__data[self.__ticker+"EMA_"+field_nm] = indicator_EMA.ema_indicator().values   
    
    def __cal_SMA(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates the SMA - Simple Moving Average. It is calculated over 
            "col_close" passed as parameter and creates corresponding 
            "SMA_"+SMA_window columns in the data-frame passed as argument.
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            None.

        Returns:
            None.

        """
        values = self.__data[[self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Simple Moving Average Indicator
        indicator_SMA = SMAIndicator(close=df_wrk["close"], window = self.__SMA_win)

        field_nm = f'w{self.__SMA_win:02d}'
        self.__data[self.__ticker+"SMA_"+field_nm] = indicator_SMA.sma_indicator().values   
    
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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.        

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
            None.

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
            None.

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_TRIX = TRIXIndicator(close=df_wrk["close"], window = self.__TRIX_win )

        field_nm = f'w{self.__TRIX_win:02d}'
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
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        
        # Initialize Vortex Indicator
        indicator_VI = VortexIndicator(high = df_wrk["high"], low = df_wrk["low"], 
                                     close=df_wrk["close"], window = self.__VI_win )
        field_nm = f'w{self.__VI_win:02d}'
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
            None.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_WMA = WMAIndicator(close=df_wrk["close"], window = self.__WMA_win )

        field_nm = f'w{self.__WMA_win:02d}'
        self.__data[self.__ticker+"WMA_"+field_nm] = indicator_WMA.wma().values

    
    def __cal_HMA(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates HMA - Hull Moving Average. It is calculated over "CLOSE" 
            prices passed in "col_close", "HMA_win" creating new columns named 
            "HMA_"+HMA_win in the data-frame passed  as argument. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            None.        

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Stochastic RSI Indicator
        indicator_HMA = pta.hma(df_wrk['close'], length=self.__HMA_win, append=True)

        field_nm = f'w{self.__HMA_win:02d}'
        self.__data[self.__ticker+"HMA_"+field_nm] = indicator_HMA.values
        
    #========================================================================================
    def calculate_indicators (self):
        """
        Calculates the indicators of the dataframe provided as specified 
            by user. If 'calc_all' is True, then all indicators are calculated, 
            overriding the 'list_ind' list. Otherwise, only the indicators present
            in 'list_ind' will be calculated. For compatibility purposes, it was 
            added the ticker label in front of all columns created.
            
        Args:
            None.        

        Returns:
            None.

        """
        if (self.__calc_all) | ('RET' in self.__list_ind):
            self.__cal_return()
            
        if (self.__calc_all) | ('LRET' in self.__list_ind):
            self.__cal_log_return()

        if (self.__calc_all) | ('PCHG' in self.__list_ind):
            self.__cal_price_change()

        if (self.__calc_all) | ('PCTCHG' in self.__list_ind):
            self.__cal_pct_change()

        if (self.__calc_all) | ('RA' in self.__list_ind):
            self.__cal_RA(5, 10)

        if (self.__calc_all) | ('DIFF' in self.__list_ind):
            self.__cal_Diff()

        if (self.__calc_all) | ('MA' in self.__list_ind):
            self.__cal_MA(5, 10)

        if (self.__calc_all) | ('VMA' in self.__list_ind):
            self.__cal_VMA(5, 10, 20)

        if (self.__calc_all) | ('KAMA' in self.__list_ind):
            self.__cal_KAMA()

        if (self.__calc_all) | ('PPO' in self.__list_ind):
            self.__cal_PPO()

        if (self.__calc_all) | ('PVO' in self.__list_ind):
            self.__cal_PVO()

        if (self.__calc_all) | ('ROC' in self.__list_ind):
            self.__cal_ROC()

        if (self.__calc_all) | ('RSI' in self.__list_ind):
            self.__cal_RSI()

        if (self.__calc_all) | ('STRSI' in self.__list_ind):
            self.__cal_StochRSI()

        if (self.__calc_all) | ('SO' in self.__list_ind):
            self.__cal_SO()

        if (self.__calc_all) | ('AOI' in self.__list_ind):
            self.__cal_AOI()

        if (self.__calc_all) | ('TSI' in self.__list_ind):
            self.__cal_TSI()

        if (self.__calc_all) | ('UO' in self.__list_ind):
            self.__cal_UO()

        if (self.__calc_all) | ('WRI' in self.__list_ind):
            self.__cal_WRI()

        if (self.__calc_all) | ('ADI' in self.__list_ind):
            self.__cal_ADI()

        if (self.__calc_all) | ('CMF' in self.__list_ind):
            self.__cal_CMF()

        if (self.__calc_all) | ('EOM' in self.__list_ind):
            self.__cal_EOM()

        if (self.__calc_all) | ('FI' in self.__list_ind):
            self.__cal_FI()

        if (self.__calc_all) | ('MFI' in self.__list_ind):
            self.__cal_MFI()

        if (self.__calc_all) | ('NVI' in self.__list_ind):
            self.__cal_NVI()

        if (self.__calc_all) | ('OBV' in self.__list_ind):
            self.__cal_OBV()

        if (self.__calc_all) | ('VPT' in self.__list_ind):
            self.__cal_VPT()

        if (self.__calc_all) | ('VWAP' in self.__list_ind):
            self.__cal_VWAP()

        if (self.__calc_all) | ('ADX' in self.__list_ind):
            self.__cal_ADX()

        if (self.__calc_all) | ('AROON' in self.__list_ind):
            self.__cal_Aroon()

        if (self.__calc_all) | ('CCI' in self.__list_ind):
            self.__cal_CCI()

        if (self.__calc_all) | ('DPO' in self.__list_ind):
            self.__cal_DPO()

        if (self.__calc_all) | ('EMA' in self.__list_ind):
            self.__cal_EMA()

        if (self.__calc_all) | ('SMA' in self.__list_ind):
            self.__cal_SMA()

        if (self.__calc_all) | ('ICHI' in self.__list_ind):
            self.__cal_Ichimoku()

        if (self.__calc_all) | ('KST' in self.__list_ind):
            self.__cal_KST()

        if (self.__calc_all) | ('MACD' in self.__list_ind):
            self.__cal_MACD()

        if (self.__calc_all) | ('MI' in self.__list_ind):
            self.__cal_MI()

        if (self.__calc_all) | ('PSAR' in self.__list_ind):
            self.__cal_PSAR()

        if (self.__calc_all) | ('STC' in self.__list_ind):
            self.__cal_STC()

        if (self.__calc_all) | ('TRIX' in self.__list_ind):
            self.__cal_TRIX()

        if (self.__calc_all) | ('VI' in self.__list_ind):
            self.__cal_VI()

        if (self.__calc_all) | ('WMA' in self.__list_ind):
            self.__cal_WMA()

        if (self.__calc_all) | ('HMA' in self.__list_ind):
            self.__cal_HMA()

    def normalize_data(self):
        """
        Noprmalizes the indicators of the dataframe provided. For compatibility 
            purposes, it was added the ticker label in front of all columns 
            created.
            
        Args:
            None.        

        Returns:
            None.

        """
        col_names = self.__data.columns
        norm_cols = [col + '_norm' for col in col_names]

        values = self.__data[col_names].values
        
        # Select the appropriate scaler
        if self.__scale_method == "minmax":
            scaler = MinMaxScaler(feature_range=(-1,1))
        else:
            scaler = StandardScaler()
            
        self.__data_norm = pd.DataFrame(data = scaler.fit_transform(values), 
                                        columns = norm_cols, 
                                        index = self.__data.index)

    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None ):
               
        """
        Method defined for compatibility purposes.
            
        Args:
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
                  y = None ) -> pd.DataFrame(dtype=float):
        """
        Transforms the dataframe containing all variables of our financial series
            calculating the indicators.
            
        Args:
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
            return self.__data_norm
        else:
            return self.__data
    
    def fit_transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None   ) -> pd.DataFrame(dtype=float):
        """
        Fit and Transforms the dataframe containing all variables of our financial series
            calculating the indicators available.
            (This routine is intented to maintaing sklearn Pipeline compatibility)
            
        Args:
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate the covariance matrix.

        Returns:
            self.__data (pd.DataFrame): Dataframe processed with indicators

        """

        self.fit(X)
        self.transform(X)
        
        return self.__data

