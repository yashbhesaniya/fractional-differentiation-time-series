import pandas as pd
from typing import List
from src.finance_ml.data_preparation.asset_class import Asset

class Portfolio:
    def __init__(self,
                 assets: List[Asset]):
        self.assets = {a.ticker:a for a in assets}

    def add_asset(self,
                  asset: Asset):
        if asset.ticker not in self.assets.keys():
            self.assets[asset.ticker] = asset
        else:
            raise ValueError("Asset with same ticker already in portfolio")

    def update_asset(self,
                     asset: Asset):
        if asset.ticker in self.assets.keys():
            self.assets[asset.ticker] = asset
        else:
            raise ValueError("Asset not in portfolio")

    def join_data(self,
                  tickers: List[str],
                  join: str = "outer") -> pd.DataFrame:
        '''
        Returns unified databases indexed by the timestamp.

        Args:
        -----
        tickers: List[str]
            A list containing the tickers of the assets whose data want to join.
            All assets must be in self.assets and have the same index_name.
        join: str = ["outer", "inner"]
            Join method for pandas concat() method. Defaults to "outer".

        Returns:
        -----
        dataset: pandas DataFrame
            The unified dataset containing data from all provided assets.
        '''
        try:
            if not all(self.assets[t].index_name == self.assets[tickers[0]].index_name for t in tickers):
                raise ValueError("Not all index names are equal")

            return(pd.concat(objs = [self.assets[t].data for t in tickers],
                             join = join))

        except KeyError:
            raise KeyError("Ticker not in assets")

    def __repr__(self):
        return f"Portfolio(assets={list(self.assets.values())})"



