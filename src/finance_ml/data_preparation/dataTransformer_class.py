"""
Created on Mon Nov 27 12:33:00 2023
@Group:
    Ahmed Ibrahim
    Brijesh Mandaliya
    Moritz Link
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

class DataTransformer():
    def __init__(self, assets: dict, DOLLAR_PRICE_COL: str = "CLOSE"):
        """
        Initialize DataTransformer.
            
        Args:
            assets (dict): contains the different assets in the portfolio
            DOLLAR_PRICE_COL (str): Sets the price column for the dollar calculation: price * volume
        Returns:
            None
        """   
        self._asset_names = list(assets.keys())
        self._asset_columns = {}
        self._return_df = pd.DataFrame()
        self._dollar_price_col = DOLLAR_PRICE_COL

        if self._dollar_price_col not in ["CLOSE", "LOW", "HEIGHT", "OPEN"]:
            raise ValueError('DataTransformer Class - Parameter must be of ["CLOSE", "LOW", "HEIGHT", "OPEN"] ')
        if (type(self._dollar_price_col) != str):
            raise ValueError('DataTransformer Class - Parameter DOLLAR_PRICE_COL must be a string')
    def transform(self, assets_df:pd.DataFrame, bar_type:str, bar_parameter: float): 
        """
        Transforms the dataframe into the chosen format.
            
        Args:
            assets_df (pd.DataFrame): dataframe to transform
            bar_type (str): param to choose the bar method
            bar_parameter (float): param for the bar method to calculate bars
        Returns:
            pd.DataFrame : The transformed dataframe
        """
        if (type(assets_df) != pd.DataFrame):
            raise ValueError('DataTransformer Class - Parameter assets_df must be a pd.DataFrame')
        if (type(bar_type) != str):
            raise ValueError('DataTransformer Class - Parameter bar_type must be a string')
        if bar_type not in ["tick", "dollar", "volume"]:
            raise ValueError('DataTransformer Class - bar_type must be of ["tick", "dollar", "volume"] ')
        if (type(bar_parameter) not in [float, int]):
            raise ValueError('DataTransformer Class - Parameter bar_parameter must be a float or int')

        # Create a dict self._asset_columns with the assets as keys and the corresponding columns as values
        assets_df._columns = assets_df.columns
        for asset_name in self._asset_names:
            asset_columns = [col for col in assets_df._columns if col.__contains__(asset_name+"_")]
            self._asset_columns[asset_name]= asset_columns 
            
        #For each asset in the dict apply the bar transformation function
        # result: one return dataframe of all transformed assets
        for ind, asset in enumerate(self._asset_columns.keys()):
            print("=" *5)
            print(f'asset: {asset}')
            asset_df = assets_df[self._asset_columns[asset]]
            # preprocessing
            asset_df = asset_df.dropna()
            #bar_calculation
            if bar_type == "tick":
                asset_df = self.tick_bars(asset_df, bar_parameter, asset)
            elif bar_type == "volume":
                asset_df = self.volume_bars(asset_df, bar_parameter, asset)
            elif bar_type == "dollar":
                 asset_df = self.dollar_bars(asset_df, bar_parameter, asset)
            else:
                raise ValueError("unknown bar_type choose one of tick, volume , dollar")

            #Create the return dataframe with outer joins
            if ind ==0:
                self._return_df = asset_df.set_index('DATE')
                
            else:
                 self._return_df = self._return_df.join(asset_df.set_index('DATE'), how = "outer")
            
        # replace the NaN values in the result dataframe by the previous value
        if self._return_df.iloc[0].isna().sum() > 0:
            fist_row_values = self._return_df.iloc[0]
            date = fist_row_values.name
            for asset in self._asset_columns.keys():
                asset_col_name = asset + "_OPEN"
                value_1 = fist_row_values[asset_col_name]
                if np.isnan(value_1):
                    asset_columns = self._asset_columns[asset]
                    self._return_df.loc[date,asset_columns ] = np.zeros(len(asset_columns), dtype = int).tolist()
                
            self._return_df = self._return_df.fillna(method='ffill')    
            
        return self._return_df
    def tick_bars(self, df:pd.DataFrame, bar_parameter: float, asset:str):
        """
        Transforms the dataframe into the tick_bar format.
            
        Args:
            df (pd.DataFrame): The asset dataframe to transform
            bar_parameter (float): number of rows to summarize into one tick
            asset (str): name of the asset
             
        Returns:
            pd.DataFrame : The transformed dataframe
        """
        asset_name = asset
        volume_col = df[asset_name+"_VOLUME"]
        hight_col = df[asset_name+"_HIGHT"]
        close_col = df[asset_name+"_CLOSE"]
        open_col = df[asset_name+"_OPEN"]
        low_col = df[asset_name+"_LOW"]
        times_col = list(df.index)
        standard_cols = [asset_name+"_OPEN", asset_name+"_HIGHT",asset_name+"_LOW",asset_name+"_CLOSE",asset_name+"_VOLUME"]
        indicator_cols = [col for col in df.columns if col not in standard_cols]
        res = np.zeros(shape=(len(range(bar_parameter, len(low_col), bar_parameter)), len(df.columns)))
        it = 0
        time_l = []
        
        # Calculate bar
        for i in tqdm(range(bar_parameter, len(low_col), bar_parameter)):
            time_l.append(times_col[i-1])                           # time
            res[it][0] = open_col[i-bar_parameter]                  # open
            res[it][1] = np.max(hight_col[i-bar_parameter:i])       # high
            res[it][2] = np.min(low_col[i-bar_parameter:i])         # low
            res[it][3] = close_col[i-1]                             # close
            res[it][4] = np.sum(volume_col[i-bar_parameter:i])      # volume
            #Other Indicators
            for ind, indicator in enumerate(indicator_cols) :
                res[it][ind+5] = np.sum(df[indicator][i-bar_parameter:i])

            it += 1
        data_frame_cols = standard_cols +  indicator_cols
        return_df = pd.DataFrame(res, columns = data_frame_cols)
        
        return_df["DATE"] = time_l
        return return_df   
    def volume_bars(self,df:pd.DataFrame, bar_parameter: float, asset:str):
        """
        Transforms the dataframe into the volume bar format.
            
        Args:
            df (pd.DataFrame): The asset dataframe to transform
            bar_parameter (float): volume threshold for one bar
            asset (str): name of the asset
             
        Returns:
            pd.DataFrame : The transformed dataframe
        """
        asset_name = asset
        volume_col = df[asset_name+"_VOLUME"]
        hight_col = df[asset_name+"_HIGHT"]
        close_col = df[asset_name+"_CLOSE"]
        open_col = df[asset_name+"_OPEN"]
        low_col = df[asset_name+"_LOW"]
        times_col = list(df.index)
        standard_cols = [asset_name+"_OPEN", asset_name+"_HIGHT",asset_name+"_LOW",asset_name+"_CLOSE",asset_name+"_VOLUME"]
        indicator_cols = [col for col in df.columns if col not in standard_cols]
        ans = np.zeros(shape=(len(low_col), len(df.columns)))
        candle_counter = 0
        vol = 0
        lasti = 0
        time_l = []
        # calculate the running sum until a threshold.
        # Calculate bar
        for i in tqdm(range(len(low_col))):
            vol += volume_col[i]
            if vol >= bar_parameter:
                
                time_l.append(times_col[i])                                     # time
                ans[candle_counter][0] = open_col[lasti]                        # open
                ans[candle_counter][1] = np.max(hight_col[lasti:i+1])           # high
                ans[candle_counter][2] = np.min(low_col[lasti:i+1])             # low
                ans[candle_counter][3] = close_col[i]                           # close
                ans[candle_counter][4] = np.sum(volume_col[lasti:i+1])          # volume
                #Other Indicators
                for ind, indicator in enumerate(indicator_cols) :
                    ans[candle_counter][ind+5] = np.sum(df[indicator][lasti:i+1])    

                candle_counter += 1
                lasti = i+1
                vol = 0
                
        ans = ans[:candle_counter]
        data_frame_cols = standard_cols +  indicator_cols
        return_df = pd.DataFrame(ans, columns = data_frame_cols)
        
        return_df["DATE"] = time_l
        return return_df
    def dollar_bars(self, df:pd.DataFrame, bar_parameter: float, asset:str):
        """
        Transforms the dataframe into the dollar bar format.
            
        Args:
            df (pd.DataFrame): The asset dataframe to transform
            bar_parameter (float): dollar threshold for one bar
            asset (str): name of the asset
             
        Returns:
            pd.DataFrame : The transformed dataframe
        """   
        asset_name = asset
        volume_col = df[asset_name+"_VOLUME"]
        hight_col = df[asset_name+"_HIGHT"]
        close_col = df[asset_name+"_CLOSE"]
        open_col = df[asset_name+"_OPEN"]
        low_col = df[asset_name+"_LOW"]
        price_col = df[asset_name+"_" + self._dollar_price_col]
        standard_cols = [asset_name+"_OPEN", asset_name+"_HIGHT",asset_name+"_LOW",asset_name+"_CLOSE",asset_name+"_VOLUME"]
        indicator_cols = [col for col in df.columns if col not in standard_cols]
        times_col = list(df.index)
        ans = np.zeros(shape=(len(close_col), len(df.columns)))
        candle_counter = 0
        dollars = 0
        lasti = 0
        time_l = []
        # calculate the running sum until a threshold.
        # Calculate bar
        for i in tqdm(range(len(price_col))):
            dollars += volume_col[i]*price_col[i]
            if dollars >= bar_parameter:
                time_l.append(times_col[i])                                 # time
                ans[candle_counter][0] = open_col[lasti]                    # open
                ans[candle_counter][1] = np.max(hight_col[lasti:i+1])       # high
                ans[candle_counter][2] = np.min(low_col[lasti:i+1])         # low
                ans[candle_counter][3] = close_col[i]                       # close
                ans[candle_counter][4] = np.sum(volume_col[lasti:i+1])
                #Other Indicators
                for ind, indicator in enumerate(indicator_cols) :
                    ans[candle_counter][ind+5] = np.sum(df[indicator][lasti:i+1])                 
                candle_counter += 1
                lasti = i+1
                dollars = 0


        ans = ans[:candle_counter]
        data_frame_cols = standard_cols +  indicator_cols
        return_df = pd.DataFrame(ans, columns = data_frame_cols)
        
        return_df["DATE"] = time_l
        return return_df
