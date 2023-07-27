"""
Created on Fri May 12 19:00:00 2023

@author: Luis Alvaro Correia

Updated: July 15th
    1. Included parameter 'sl' for CorwinSchultz high-low volatility index
    2. Set return to 'None' for functions cal_CorwinSchultz, cal_GarmanKlass, 
        cal_RogersSatchell, cal_YangZhang and cal_HodgesTompkins
Updated: July 27th
    1. Removed dscription of 'self' parameter on function's documentation

"""
# Import required packages
import pandas as pd
import numpy as np

import math

from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel, \
                            KeltnerChannel, UlcerIndex

from sklearn.base import BaseEstimator, TransformerMixin

class Volatility(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 col_open: str = 'OPEN',
                 col_high: str = 'HIGHT',
                 col_low: str = 'LOW',
                 col_close: str = 'CLOSE',
                 col_volume: str = 'VOLUME',
                 ticker: str = '',
                 calc_all: bool = True,
                 list_vol: list = [],
                 atr_win: int = 14,
                 bb_win: int = 20,
                 bb_dev: int = 2,
                 dc_win: int = 20,
                 dc_off: int = 0,
                 kc_win: int = 20,
                 kc_win_atr: int = 10,
                 kc_mult: int = 2,
                 ui_win: int = 14,
                 sl_win: int = 1,
                 window: int = 30,
                 trading_periods: int = 252,
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
            list_vol (list): list of indicators do calculate if calc_all is False
            atr_win (int): n ATR period
            bb_win (int): n period for Bollinger-Bands indicator
            bb_dev (int): n factor standard deviation
            dc_win (int): n period for Donchia-Channel indicator
            dc_off (int): offset for Donchia-Channel indicator
            kc_win (int): n period of Keltner-Channel indicator
            kc_win_atr (int): n ATR period. Only valid if original_version param is False.
            kc_mult (int): multiplier for Keltner-Channel indicator
            ui_win (int): n period of Ulcer Index
            sl_win (int): n periods of 2-day spread for Corwin-Schultz indicator
            window (int): no. of days for rolling prices
            trading_periods (int): no. of trading periods

        Returns:
            None

        """
        VOL_LIST = ['BB', 'ATR', 'DC', 'KC', 'UI', 'CS', 'HT', 'YZ', 'RZ', 'GK']
        
        if (type(calc_all) != bool):
            raise ValueError('Volatility Class - Parameter calc_all must be True or False')
        
        if (type(list_vol) != list):
            raise ValueError('Volatility Class - Parameter list_ind must be a list')
            
        list_vol = [l.upper() for l in list_vol]
        
        if (not set(list_vol).issubset(VOL_LIST)):
            raise ValueError(f'Volatility Class - Invalid Indicator {set(list_vol)-set(VOL_LIST)}')
        
        if (type(col_open) != str):
            raise ValueError('Volatility Class - Parameter col_open must be a valid str')

        if (type(col_high) != str):
            raise ValueError('Volatility Class - Parameter col_high must be a valid str')

        if (type(col_low) != str):
            raise ValueError('Volatility Class - Parameter col_low must be a valid str')

        if (type(col_close) != str):
            raise ValueError('Volatility Class - Parameter col_close must be a valid str')

        if (type(col_volume) != str):
            raise ValueError('Volatility Class - Parameter col_volume must be a valid str')

        if (type(atr_win) != int) | (atr_win <= 0):
            raise ValueError('Volatility Class - Parameter atr_win must be int, positive')

        if (type(bb_win) != int) | (bb_win <= 0):
            raise ValueError('Volatility Class - Parameter bb_win must be int, positive')

        if (type(bb_dev) != int) | (bb_dev <= 0) | (bb_dev >= bb_win):
            raise ValueError('Volatility Class - Parameter bb_dev must be int, positive, less than bb_win')

        if (type(dc_win) != int) | (dc_win <= 0):
            raise ValueError('Volatility Class - Parameter dc_win must be int, positive')

        if (type(dc_off) != int) | (dc_off < 0) | (dc_off >= dc_win):
            raise ValueError('Volatility Class - Parameter dc_off must be int, positive, less than dc_win')

        if (type(kc_win) != int) | (kc_win <= 0):
            raise ValueError('Volatility Class - Parameter kc_win must be int, positive')

        if (type(kc_win_atr) != int) | (kc_win_atr <= 0) | (kc_win_atr >= kc_win):
            raise ValueError('Volatility Class - Parameter kc_win_atr must be int, positive, less than kc_win')

        if (type(kc_mult) != int) | (kc_mult <= 0):
            raise ValueError('Volatility Class - Parameter kc_mult must be int, positive')

        if (type(ui_win) != int) | (ui_win <= 0):
            raise ValueError('Volatility Class - Parameter ui_win must be int, positive')

        if (type(sl_win) != int) | (ui_win <= 0):
            raise ValueError('Volatility Class - Parameter sl_win must be int, positive')

        if (type(window) != int) | (window <= 0):
            raise ValueError('Volatility Class - Parameter window must be int, positive')

        if (type(trading_periods) != int) | (trading_periods <= 0) | (trading_periods <= window):
            raise ValueError('Volatility Class - Parameter trading_periods must be int, positive, greater than window')
            
        self.__ticker = (ticker+'_' if ticker!='' else '')
        self.__col_open = self.__ticker + col_open
        self.__col_high = self.__ticker + col_high
        self.__col_low = self.__ticker + col_low
        self.__col_close = self.__ticker + col_close
        self.__col_volume = self.__ticker + col_volume
        self.__calc_all = calc_all
        self.__list_vol = list_vol
        self.__atr_win = atr_win
        self.__bb_win = bb_win
        self.__bb_dev = bb_dev
        self.__dc_win = dc_win
        self.__dc_off = dc_off
        self.__kc_win = kc_win
        self.__kc_win_atr = kc_win_atr
        self.__kc_mult = kc_mult
        self.__ui_win = ui_win
        self.__sl_win = sl_win
        self.__window = window
        self.__trading_periods = trading_periods

    @property
    def data(self) -> pd.DataFrame(dtype=float):
        return self.__data

    @property
    def ticker(self) -> str:
        return self.__ticker

    def __getBeta(self,
                  sl: int) -> pd.Series(dtype=float):
        """
        Calculates Beta, a measure of a stock's volatility in relation to the 
            overall market.
            
        Args:
            sl (int): no. of days for rolling prices

        Returns:
            (pd.Series): Beta values

        """
        hl = self.__data[[self.__col_high,self.__col_low]].values
        hl = np.log(hl[:,0] / hl[:,1])**2
        hl = pd.Series(hl,index = self.__data.index)
        beta = hl.rolling(window = 2).sum()
        beta = beta.rolling(window = sl).mean()
        return beta # beta.dropna()
    
    def __getGamma(self) -> pd.Series(dtype=float):
        """
        Calculates Gamma, the rate of change for an option's delta based on a 
            single-point move in the delta's price. It is a second-order risk 
            factor, sometimes known as the delta of the delta. Gamma is at its 
            highest when an option is at the money and is at its lowest when it 
            is further away from the money.
            
        Args:
            None.        

        Returns:
            (pd.Series): Gamma values

        """
        h2 = self.__data[self.__col_high].rolling(window = 2).max()
        l2 = self.__data[self.__col_low].rolling(window = 2).min()
        gamma = np.log(h2.values / l2.values)**2
        gamma = pd.Series(gamma, index = h2.index)
        return gamma # gamma.dropna()
    
    def __getAlpha(self, 
                   beta: pd.Series(dtype=float), 
                   gamma: pd.Series(dtype=float)) -> pd.Series(dtype=float):
        """
        Calculates Alpha is a measure of the active return on an investment, 
            the performance of that investment compared with a suitable market 
            index.
            
        Args:
            beta (pd.Series): beta of data prices
            gamma (pd.Series): gamma of data prices

        Returns:
            (pd.Series): Alpha values

        """
        den = 3 - 2 * 2 **.5
        alpha = (2 **.5 - 1) * (beta **.5) / den
        alpha -= (gamma / den) **.5
        alpha[alpha < 0] = 0 # set negative alphas to 0 (see p.727 of paper)
        return alpha # alpha.dropna()
    
    # ESTIMATING VOLATILITY FOR HIGH-LOW PRICES
    def __getSigma(self, 
                   beta: pd.Series(dtype=float), 
                   gamma: pd.Series(dtype=float)) -> pd.Series(dtype=float):
        """
        Calculates Sigma is a measure of the active return on an investment, 
            the performance of that investment compared with a suitable market 
            index.
            
        Args:
            beta (float): beta of data prices
            gamma (float): gamma of data prices

        Returns:
            (pd.Series): sigma values

        """
        k2 = (8 / np.pi) **.5
        den = 3 - 2 * 2**.5
        sigma = (2** - .5 - 1)  * beta **.5 / (k2 * den)
        sigma += (gamma / (k2 **2 * den)) **.5
        sigma[sigma < 0] = 0
        return sigma
    
    #------------------------ TECHNICAL INDICATORS -----------------------------
    def cal_BollingerBands(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Bollinger Bands of "CLOSE" prices passed in "col_close", 
            considering the windows "rol_win" and "win_dev" passed as arguments 
            and creates new columns named "BB_"+rol_win+win_dev in the data-frame 
            passed as argument. For compatibility purposes, it was added the 
            ticker label in front of all columns created.
            
        Args:
            None.       

        Returns:
            None.

        """
        values = self.__data[self.__col_close].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["close"]
        
        # Initialize Bollinger Bands Indicator
        indicator_bb = BollingerBands(close=df_wrk["close"], window=self.__bb_win, window_dev=self.__bb_dev)
        
        # Add Bollinger Bands features
        field_nm = f'w{self.__bb_win:02d}'
        self.__data[self.__ticker+"BBM_"+field_nm] = indicator_bb.bollinger_mavg().values
        self.__data[self.__ticker+"BBH_"+field_nm] = indicator_bb.bollinger_hband().values
        self.__data[self.__ticker+"BBL_"+field_nm] = indicator_bb.bollinger_lband().values
        
        # Add Bollinger Band high indicator
        self.__data[self.__ticker+"BBHI_"+field_nm] = indicator_bb.bollinger_hband_indicator().values
        
        # Add Bollinger Band low indicator
        self.__data[self.__ticker+"BBLI_"+field_nm] = indicator_bb.bollinger_lband_indicator().values
        
        # Add Width Size Bollinger Bands
        self.__data[self.__ticker+"BBW_"+field_nm] = indicator_bb.bollinger_wband().values
        
        # Add Percentage Bollinger Bands
        self.__data[self.__ticker+"BBP_"+field_nm] = indicator_bb.bollinger_pband().values
    
    def cal_ATR(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Average True Range (ATR) to capture volatility of stocks windows 
            "atr_win" passed as argument and creates new columns named "ATR_"+atr_win
            in the data-frame also passed as argument. For compatibility purposes, 
            it was added the ticker label in front of all columns created.
            
        Args:
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        indicator_atr = AverageTrueRange(high = df_wrk["high"], low = df_wrk["low"], 
                                         close = df_wrk["close"], window = self.__atr_win)
        
        # Add ATR features
        field_nm = f'w{self.__atr_win:02d}'
        self.__data[self.__ticker+"ATR_"+field_nm] = indicator_atr.average_true_range().values  
    
    def cal_DonchianChannel(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Donchian Channel index to capture volatility of stocks windows 
            "dc_win" passed as argument and creates new columns named "DC_"+dc_win
            in the data-frame also passed as argument. For compatibility purposes, 
            it was added the ticker label in front of all columns created.
            
        Args:
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        indicator_dc = DonchianChannel(high = df_wrk["high"], low = df_wrk["low"], 
                                         close = df_wrk["close"], window = self.__dc_win)
        
        # Add Donchian Channel Bands features
        field_nm = f'w{self.__dc_win:02d}'
        self.__data[self.__ticker+"DCM_"+field_nm] = indicator_dc.donchian_channel_mband().values
        self.__data[self.__ticker+"DCH_"+field_nm] = indicator_dc.donchian_channel_hband().values
        self.__data[self.__ticker+"DCL_"+field_nm] = indicator_dc.donchian_channel_lband().values
        
        # Add Width Size of Donchian Channel Bands
        self.__data[self.__ticker+"DCW_"+field_nm] = indicator_dc.donchian_channel_wband().values

    def cal_KeltnerChannel(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Keltner Channel index to capture volatility of stocks windows 
            "dc_win" passed as argument and creates new columns named "KC_"+kc_win
            in the data-frame also passed as argument. For compatibility purposes, 
            it was added the ticker label in front of all columns created.
            
        Args:
            None.        

        Returns:
            None.

        """
        values = self.__data[[self.__col_high,self.__col_low,self.__col_close]].values
        df_wrk = pd.DataFrame(values)
        df_wrk.columns = ["high","low","close"]
        indicator_kc = KeltnerChannel(high = df_wrk["high"], low = df_wrk["low"], 
                                         close = df_wrk["close"], window = self.__kc_win,
                                         window_atr = self.__kc_win_atr, 
                                         multiplier = self.__kc_mult)
        
        # Add Keltner Channel Bands features
        field_nm = f'w{self.__kc_win:02d}wa{self.__kc_win_atr:02d}m{self.__kc_mult:02d}'
        self.__data[self.__ticker+"KCH_"+field_nm] = indicator_kc.keltner_channel_hband().values
        self.__data[self.__ticker+"KCHI_"+field_nm] = indicator_kc.keltner_channel_hband_indicator().values
        self.__data[self.__ticker+"KCL_"+field_nm] = indicator_kc.keltner_channel_lband().values
        self.__data[self.__ticker+"KCLI_"+field_nm] = indicator_kc.keltner_channel_lband_indicator().values
        self.__data[self.__ticker+"KCM_"+field_nm] = indicator_kc.keltner_channel_mband().values

        # Add Keltner Channel Percentage Band
        # self.__data[self.__ticker+"KCP_"+str(self.__kc_win)] = indicator_kc.keltner_channel_pband().values
        
        # Add Keltner Channel Band Width
        self.__data[self.__ticker+"KCW_"+field_nm] = indicator_kc.keltner_channel_wband().values

    def cal_UlcerIndex(self) -> None:
        """
        Based on TA Technical Analysis Library in Python from Dario Lopez Padial (Bukosabino)
            https://github.com/bukosabino/ta/blob/master/docs/index.rst
        
        Calculates Ulcer Index of "CLOSE" prices passed in "col_close", 
            considering the windows "ui_win" passed as argument and creates 
            new columns named "UI_"+ui_win in the data-frame passed as argument. 
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
        
        # Initialize Bollinger Bands Indicator
        indicator_ui = UlcerIndex(close=df_wrk["close"], window=self.__ui_win)
        
        # Add Ulcer Index indicator
        field_nm = f'w{self.__ui_win:02d}'
        self.__data[self.__ticker+"UI_"+field_nm] = indicator_ui.ulcer_index().values
           
    #------------------------------------------------------------------------------------
    # IMPLEMENTATION OF THE CORWIN-SCHULTZ Algorithm
    def cal_CorwinSchultz(self) -> None:
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Corwin-Shultz volatility index over a series or stock prices.
            
        Args:
            None.        

        Returns:
            None.

        """
        # Note: S<0 iif alpha<0
        beta = self.__getBeta(self.__sl_win)
        gamma = self.__getGamma()
        alpha = self.__getAlpha(beta, gamma)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        startTime = pd.Series(self.__data.index[0:spread.shape[0]], index = spread.index)
        spread = pd.concat([spread,startTime], axis = 1)
        spread.columns = ['Spread','Start_Time'] # 1st loc used to compute beta

        field_nm = f'sl{self.__sl_win:02d}'
        self.__data[self.__ticker+'CorwinSchultz_'+field_nm] = spread['Spread'].values
    
    def cal_GarmanKlass(self,
                        clean:bool = False) -> None:
    
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Garman-Klass volatility index over a series or stock prices.
            
        Args:
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_hl = (self.__data[self.__col_high] / self.__data[self.__col_low]).apply(np.log)
        log_co = (self.__data[self.__col_close] / self.__data[self.__col_open]).apply(np.log)
    
        rs = 0.5 * log_hl **2 - (2 * math.log(2) - 1) * log_co **2
        
        def f(v):
            return (self.__trading_periods * v.mean()) **0.5
        
        result = rs.rolling(window = self.__window, center = False).apply(func=f)
        
        field_nm = f'w{self.__window:02d}tp{self.__trading_periods:03d}'
        if clean:
            self.__data[self.__ticker+'GarmanKlass_'+field_nm] = result.dropna().values
        else:
            self.__data[self.__ticker+'GarmanKlass_'+field_nm] = result.values
        
    def cal_RogersSatchell(self,
                           clean:bool = False) -> None:
        
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Rogers-Satchell volatility index over a series or stock prices.
            
        Args:
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_ho = (self.__data[self.__col_high] / self.__data[self.__col_open]).apply(np.log)
        log_lo = (self.__data[self.__col_low] / self.__data[self.__col_open]).apply(np.log)
        log_co = (self.__data[self.__col_close] / self.__data[self.__col_open]).apply(np.log)
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
        def f(v):
            return self.__trading_periods * v.mean()**0.5
        
        result = rs.rolling(window = self.__window, center = False).apply(func=f)
        
        field_nm = f'w{self.__window:02d}tp{self.__trading_periods:03d}'
        if clean:
            self.__data[self.__ticker+'RogersSatchell_'+field_nm] = result.dropna().values
        else:
            self.__data[self.__ticker+'RogersSatchell_'+field_nm] = result.values
        
    def cal_YangZhang(self,
                      clean:bool = False) -> None:
    
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Yang-Zang volatility index over a series or stock prices.
            
        Args:
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_ho = (self.__data[self.__col_high] / self.__data[self.__col_open]).apply(np.log)
        log_lo = (self.__data[self.__col_low] / self.__data[self.__col_open]).apply(np.log)
        log_co = (self.__data[self.__col_close] / self.__data[self.__col_open]).apply(np.log)
        
        log_oc = (self.__data[self.__col_open] / self.__data[self.__col_close].shift(1)).apply(np.log)
        log_oc_sq = log_oc **2
        
        log_cc = (self.__data[self.__col_close] / self.__data[self.__col_close].shift(1)).apply(np.log)
        log_cc_sq = log_cc **2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = log_cc_sq.rolling(window = self.__window, center = False).sum() * (1.0 / (self.__window - 1.0))
        open_vol = log_oc_sq.rolling(window = self.__window, center = False).sum() * (1.0 / (self.__window - 1.0))
        window_rs = rs.rolling(window = self.__window, center = False).sum() * (1.0 / (self.__window - 1.0))
    
        k = 0.34 / (1 + (self.__window + 1) / (self.__window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(self.__trading_periods)
    
        field_nm = f'w{self.__window:02d}tp{self.__trading_periods:03d}'
        if clean:
            self.__data[self.__ticker+'YangZhang_'+field_nm] = result.dropna().values
        else:
            self.__data[self.__ticker+'YangZhang_'+field_nm] = result.values
        
    def cal_HodgesTompkins(self,
                           clean:bool = False) -> None:
        
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Hodges-Tompkins volatility index over a series or stock prices.
            
        Args:
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_return = (self.__data[self.__col_close] / self.__data[self.__col_close].shift(1)).apply(np.log)
    
        vol = log_return.rolling(window = self.__window, center = False).std() * math.sqrt(self.__trading_periods)
    
        h = self.__window
        n = (log_return.count() - h) + 1
    
        adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))
    
        result = vol * adj_factor
    
        field_nm = f'w{self.__window:02d}tp{self.__trading_periods:03d}'
        if clean:
            self.__data[self.__ticker+'HodgesTompkins_'+field_nm] = result.dropna().values
        else:
            self.__data[self.__ticker+'HodgesTompkins_'+field_nm] = result.values
        

    def calculate_volatilities (self):
        """
        Calculates the volatility indicators of the dataframe provided as specified 
            by user. If 'calc_all' is True, then all indicators are calculated, 
            overriding the 'list_ind' list. Otherwise, only the indicators present
            in 'list_ind' will be calculated.  
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            None.        

        Returns:
            None.

        """
        
        # Calculate Technical Indicators
            
        if (self.__calc_all) | ('BB' in self.__list_vol):
            self.cal_BollingerBands ()
            
        if (self.__calc_all) | ('ATR' in self.__list_vol):
            self.cal_ATR()
            
        if (self.__calc_all) | ('DC' in self.__list_vol):
            self.cal_DonchianChannel ()
            
        if (self.__calc_all) | ('KC' in self.__list_vol):
            self.cal_KeltnerChannel ()
            
        if (self.__calc_all) | ('UI' in self.__list_vol):
            self.cal_UlcerIndex ()
            
        # Calculate Volatility Indicators
            
        if (self.__calc_all) | ('CS' in self.__list_vol):
            self.cal_CorwinSchultz()
            
        if (self.__calc_all) | ('HT' in self.__list_vol):
            self.cal_HodgesTompkins()
            
        if (self.__calc_all) | ('YZ' in self.__list_vol):
            self.cal_YangZhang()
            
        if (self.__calc_all) | ('RS' in self.__list_vol):
            self.cal_RogersSatchell()
            
        if (self.__calc_all) | ('GK' in self.__list_vol):
            self.cal_GarmanKlass()
        
    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        """
        Method defined for compatibility purposes.
            
        Args:
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate volatilities.

        Returns:
            self (object)

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
            calculating the volatility indicators available.
            
        Args:
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate volatility indicators from.

        Returns:
            X_tilda (pd.DataFrame): Original Dataframe with volatility indicators

        """

        self.calculate_volatilities()
        
        return self.__data
    
    def fit_transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None  ) -> pd.DataFrame(dtype=float):
        """
        Fit and Transforms the dataframe containing all variables of our financial series
            calculating the volatility indicators available.
            (This routine is intented to maintaing sklearn Pipeline compatibility)
            
        Args:
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate the covariance matrix.

        Returns:
            self (object)

        """

        self.fit(X)
        self.transform(X)
        
        return self.__data
