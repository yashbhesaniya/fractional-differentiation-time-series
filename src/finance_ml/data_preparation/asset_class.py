import warnings
import pandas as pd

class Asset:
    def __init__(self,
                 ticker: str,
                 data: pd.DataFrame,
                 index_name: str):

        self.ticker = ticker
        self.index_name = index_name

        if data.index.name != index_name:
            warnings.warn("Provided index_name and data.index.name are different. Setting data.index.name to index_name")
            data.index.name = index_name

        self.data = data

    def __repr__(self):
        return f"Asset(ticker='{self.ticker}', index_name='{self.index_name}')"

