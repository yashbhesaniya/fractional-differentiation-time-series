# Import required packages
import pandas as pd
import numpy as np

import math

from ta.volatility import BollingerBands, AverageTrueRange

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
set_config(transform_output="pandas")

class Volatility(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 data: pd.DataFrame(dtype=float),
                 ticker: str = '',
                 atr_win: int = 14,
                 bb_win: int = 20,
                 bb_dev: int = 2,
                 ):
        """
        Initialize data.
            
        Args:
            self: object
                All entries in function __init__.        
            data (pd.DataFrame): Columns of dataframe containing the variables 
                to be scaled.
            bb_win (int): n period
            bb_dev (int): n factor standard deviation

        Returns:
            None

        """
        if (type(bb_win) != int) | (bb_win <= 0):
            raise ValueError('Volatility Class - Parameter bb_win must be int, positive')

        if (type(bb_dev) != int) | (bb_dev <= 0) | (bb_dev >= bb_win):
            raise ValueError('Volatility Class - Parameter bb_dev must be int, positive, less than bb_win')

        self.__data = data
        self.__ticker = (ticker+'_' if ticker!='' else '')
        self.__atr_win = atr_win
        self.__bb_win = bb_win
        self.__bb_dev = bb_dev

    @property
    def data(self):
        return self.__data

    @property
    def ticker(self):
        return self.__ticker

    def __getBeta(self, 
                  col_high: str,
                  col_low: str,
                  sl: int) -> pd.Series(dtype=float):
        """
        Calculates Beta, a measure of a stock's volatility in relation to the 
            overall market.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            sl (int): no. of days for rolling prices

        Returns:
            (pd.Series): Beta values

        """
        hl = self.__data[[col_high,col_low]].values
        hl = np.log(hl[:,0] / hl[:,1])**2
        hl = pd.Series(hl,index = self.__data.index)
        beta = hl.rolling(window = 2).sum()
        beta = beta.rolling(window = sl).mean()
        return beta # beta.dropna()
    
    def __getGamma(self,
                   col_high: str,
                   col_low: str    ) -> pd.Series(dtype=float):
        """
        Calculates Gamma, the rate of change for an option's delta based on a 
            single-point move in the delta's price. It is a second-order risk 
            factor, sometimes known as the delta of the delta. Gamma is at its 
            highest when an option is at the money and is at its lowest when it 
            is further away from the money.
            
        Args:
            self: object
                All entries in function __init__.        
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices

        Returns:
            (pd.Series): Gamma values

        """
        h2 = self.__data[col_high].rolling(window = 2).max()
        l2 = self.__data[col_low].rolling(window = 2).min()
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
            self: object
                All entries in function __init__.        
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
            self: object
                All entries in function __init__.        
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
    def cal_BollingerBands(self,
               col_close: str ) -> None:
        """
        Calculates Bollinger Bands of "CLOSE" prices passed in "col_close", 
            considering the windows "rol_win" and "win_dev" passed as arguments 
            and creates new columns named "BB"+rol_win+win_dev in the data-frame 
            passed as argument. For compatibility purposes, it was added the 
            ticker label in front of all columns created.
            
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
        
        # Initialize Bollinger Bands Indicator
        indicator_bb = BollingerBands(close=df_wrk["close"], window=self.__bb_win, window_dev=self.__bb_dev)
        
        # Add Bollinger Bands features
        self.__data[self.__ticker+"BBM_"+str(self.__bb_win)] = indicator_bb.bollinger_mavg().values
        self.__data[self.__ticker+"BBH_"+str(self.__bb_win)] = indicator_bb.bollinger_hband().values
        self.__data[self.__ticker+"BBL_"+str(self.__bb_win)] = indicator_bb.bollinger_lband().values
        
        # Add Bollinger Band high indicator
        self.__data[self.__ticker+"BBHI_"+str(self.__bb_win)] = indicator_bb.bollinger_hband_indicator().values
        
        # Add Bollinger Band low indicator
        self.__data[self.__ticker+"BBLI_"+str(self.__bb_win)] = indicator_bb.bollinger_lband_indicator().values
        
        # Add Width Size Bollinger Bands
        self.__data[self.__ticker+"BBW_"+str(self.__bb_win)] = indicator_bb.bollinger_wband().values
        
        # Add Percentage Bollinger Bands
        self.__data[self.__ticker+"BBP_"+str(self.__bb_win)] = indicator_bb.bollinger_pband().values
    
    def cal_ATR(self,
                col_high: str,
                col_low: str,
                col_close: str ) -> None:
        """
        Calculates Average True Range (ATR) to capture volatility of stocs windows 
            "atr_win" passed as argument And creates new columns named "ATR"+atr_win
            in the data-frame also passed as argument. For compatibility purposes, 
            it was added the ticker label in front of all columns created.
            
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
        indicator_atr = AverageTrueRange(high = df_wrk["high"], low = df_wrk["low"], 
                                         close = df_wrk["close"], window = self.__atr_win)
        
        # Add Bollinger Bands features
        self.__data[self.__ticker+"ATR_"+str(self.__atr_win)] = indicator_atr.average_true_range().values  
    
    #------------------------------------------------------------------------------------
    # IMPLEMENTATION OF THE CORWIN-SCHULTZ Algorithm
    def cal_CorwinSchultz(self, 
                      col_high: str,
                      col_low: str,
                      sl: int = 1):
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Corwin-Shultz volatility index over a series or stock prices.
            
        Args:
            self: object
                All entries in function __init__.        
            series (pd.Series): Series of data prices
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            sl (int): no. of days for rolling prices

        Returns:
            None.

        """
        # Note: S<0 iif alpha<0
        beta = self.__getBeta(col_high, col_low, sl)
        gamma = self.__getGamma(col_high, col_low)
        alpha = self.__getAlpha(beta, gamma)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        startTime = pd.Series(self.__data.index[0:spread.shape[0]], index = spread.index)
        spread = pd.concat([spread,startTime], axis = 1)
        spread.columns = ['Spread','Start_Time'] # 1st loc used to compute beta
        self.__data[self.__ticker+'CorwinSchultz'] = spread['Spread'].values
    
    def cal_GarmanKlass(self,
                        col_open: str,
                        col_high: str,
                        col_low: str,
                        col_close: str,
                        window: int = 30, 
                        trading_periods: int = 252, 
                        clean:bool = False):
    
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Garman-Klass volatility index over a series or stock prices.
            
        Args:
            self: object
                All entries in function __init__.        
            price_data (pd.DataFrame): DataFrame containing stock data
            col_open (str): name of the column with the "OPEN" data prices
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices
            window (int): no. of days for rolling prices
            trading_periods (int): no. of trading periods
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_hl = (self.__data[col_high] / self.__data[col_low]).apply(np.log)
        log_co = (self.__data[col_close] / self.__data[col_open]).apply(np.log)
    
        rs = 0.5 * log_hl **2 - (2 * math.log(2) - 1) * log_co **2
        
        def f(v):
            return (trading_periods * v.mean()) **0.5
        
        result = rs.rolling(window = window, center = False).apply(func=f)
        
        if clean:
            self.__data[self.__ticker+'GarmanKlass'] = result.dropna().values
        else:
            self.__data[self.__ticker+'GarmanKlass'] = result.values
        
    def cal_RogersSatchell(self, 
                           col_open: str,
                           col_high: str,
                           col_low: str,
                           col_close: str,
                           window: int = 30, 
                           trading_periods: int = 252, 
                           clean:bool = False):
        
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Rogers-Satchell volatility index over a series or stock prices.
            
        Args:
            self: object
                All entries in function __init__.        
            price_data (pd.DataFrame): DataFrame containing stock data
            col_open (str): name of the column with the "OPEN" data prices
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices
            window (int): no. of days for rolling prices
            trading_periods (int): no. of trading periods
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_ho = (self.__data[col_high] / self.__data[col_open]).apply(np.log)
        log_lo = (self.__data[col_low] / self.__data[col_open]).apply(np.log)
        log_co = (self.__data[col_close] / self.__data[col_open]).apply(np.log)
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
        def f(v):
            return trading_periods * v.mean()**0.5
        
        result = rs.rolling(window = window, center = False).apply(func=f)
        
        if clean:
            self.__data[self.__ticker+'RogersSatchell'] = result.dropna().values
        else:
            self.__data[self.__ticker+'RogersSatchell'] = result.values
        
    def cal_YangZhang(self, 
                      col_open: str,
                      col_high: str,
                      col_low: str,
                      col_close: str,
                      window: int = 30, 
                      trading_periods:int = 252, 
                      clean:bool = False):
    
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Yang-Zang volatility index over a series or stock prices.
            
        Args:
            self: object
                All entries in function __init__.        
            price_data (pd.DataFrame): DataFrame containing stock data
            col_open (str): name of the column with the "OPEN" data prices
            col_high (str): name of the column with the "HIGH" data prices
            col_low (str): name of the column with the "LOW" data prices
            col_close (str): name of the column with the "CLOSE" data prices
            window (int): no. of days for rolling prices
            trading_periods (int): no. of trading periods
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_ho = (self.__data[col_high] / self.__data[col_open]).apply(np.log)
        log_lo = (self.__data[col_low] / self.__data[col_open]).apply(np.log)
        log_co = (self.__data[col_close] / self.__data[col_open]).apply(np.log)
        
        log_oc = (self.__data[col_open] / self.__data[col_close].shift(1)).apply(np.log)
        log_oc_sq = log_oc **2
        
        log_cc = (self.__data[col_close] / self.__data[col_close].shift(1)).apply(np.log)
        log_cc_sq = log_cc **2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = log_cc_sq.rolling(window = window, center = False).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(window = window, center = False).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(window = window, center = False).sum() * (1.0 / (window - 1.0))
    
        k = 0.34 / (1 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)
    
        if clean:
            self.__data[self.__ticker+'YangZhang'] = result.dropna().values
        else:
            self.__data[self.__ticker+'YangZhang'] = result.values
        
    def cal_HodgesTompkins(self, 
                           col_close: str,
                           window: int = 30, 
                           trading_periods: int = 252, 
                           clean:bool = False):
        
        """
        Adapted from Chap. 2 of  "Volatility Trading", by
        - Euan Sinclair - Wiley, 1st. edition

        Calculates Hodges-Tompkins volatility index over a series or stock prices.
            
        Args:
            self: object
                All entries in function __init__.        
            price_data (pd.DataFrame): DataFrame containing stock data
            col_close (str): name of the column with the "CLOSE" data prices
            window (int): no. of days for rolling prices
            trading_periods (int): no. of trading periods
            clean (bool): if 'True', removes 'NaN's; otherwise don't remove

        Returns:
            None.

        """
        log_return = (self.__data[col_close] / self.__data[col_close].shift(1)).apply(np.log)
    
        vol = log_return.rolling(window = window, center = False).std() * math.sqrt(trading_periods)
    
        h = window
        n = (log_return.count() - h) + 1
    
        adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))
    
        result = vol * adj_factor
    
        if clean:
            self.__data[self.__ticker+'HodgesTompkins'] = result.dropna().values
        else:
            self.__data[self.__ticker+'HodgesTompkins'] = result.values
        

    def calculate_volatilities (self):
        """
        Calculates the volatility indicators of the dataframe provided. 
            For compatibility purposes, it was added the ticker label in front 
            of all columns created.
            
        Args:
            self: object
                All entries in function __init__.        

        Returns:
            None.

        """
        # Calculate Technical Indicators
        self.cal_BollingerBands (self.__ticker+"CLOSE")
        self.cal_ATR (self.__ticker+"HIGHT", self.__ticker+"LOW", 
                      self.__ticker+"CLOSE" )
        
        # Calculate Volatility Indicators
        self.cal_CorwinSchultz(self.__ticker+'HIGHT', self.__ticker+'LOW')
        self.cal_HodgesTompkins(self.__ticker+"CLOSE" )
        self.cal_YangZhang(self.__ticker+"OPEN", self.__ticker+"HIGHT", 
                           self.__ticker+"LOW", self.__ticker+"CLOSE" )
        self.cal_RogersSatchell(self.__ticker+"OPEN", self.__ticker+"HIGHT", 
                                self.__ticker+"LOW", self.__ticker+"CLOSE" )
        self.cal_GarmanKlass(self.__ticker+"OPEN", self.__ticker+"HIGHT", 
                             self.__ticker+"LOW", self.__ticker+"CLOSE" )
        
    def fit(self, 
            X: pd.DataFrame(dtype=float), 
            y = None):
               
        """
        Method defined for compatibility purposes.
            
        Args:
            self: object
                All entries in function __init__.        
    
            X (pd.DataFrame): Columns of dataframe containing the variables to be
                used to calculate volatilities.

        Returns:
            self (objext)

        """
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

        self.calculate_volatilities()
        
        return self.__data
    
    def fit_transform(self, 
                  X: pd.DataFrame(dtype=float), 
                  y = None                     ):
        """
        Fit and Transforms the dataframe containing all variables of our financial series
            calculating the volatility indicators available.
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
