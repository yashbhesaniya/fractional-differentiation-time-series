# import warnings
import pytest

import pandas as pd

from src.finance_ml.data_preparation.data_preparation import DataLoader
from src.finance_ml.data_preparation.asset_class import Asset
from src.finance_ml.data_preparation.portfolio_class import Portfolio

def test_asset():
    '''
    Tests consistency of Asset.index_name on instantiation and the warning on inconsistent data entry.
    '''
    dataloader = DataLoader(time_index_col = 'DATE',
                            keep_cols = ['VOLUME','OPEN', 'CLOSE', 'LOW', 'TRANSACTIONS'])

    asset_a = Asset(ticker = "GOLD",
                    # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                    data = pd.DataFrame({"a": [1,2], "b": [3,4]}).rename_axis(index=dataloader.time_index_col),
                    index_name = dataloader.time_index_col)
    assert isinstance(asset_a.data, pd.DataFrame)
    assert asset_a.data.index.name == asset_a.index_name

    with pytest.warns(Warning):
        asset_b = Asset(ticker = "GOLD",
                        # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                        data = pd.DataFrame({"a": [1,2], "b": [3,4]}).rename_axis(index=dataloader.time_index_col),
                        index_name = "WRONG_INDEX")
    assert asset_b.data.index.name == asset_b.index_name

def test_portfolio():
    '''
    Tests consistency of Portfolio.assets.keys() on instantiation and the Portfolio.join_data() method.
    '''
    dataloader = DataLoader(time_index_col = 'DATE',
                            keep_cols = ['VOLUME','OPEN', 'CLOSE', 'LOW', 'TRANSACTIONS'])
    asset_a = Asset(ticker = "GOLD",
                    # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                    data = pd.DataFrame({"a": [1,2], "b": [3,4]}).rename_axis(index=dataloader.time_index_col),
                    index_name = dataloader.time_index_col)
    asset_b = Asset(ticker = "BTC",
                    # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                    data = pd.DataFrame({"c": [1,2], "d": [3,4]}).rename_axis(index=dataloader.time_index_col),
                    index_name = dataloader.time_index_col)
    portfolio = Portfolio([asset_a, asset_b])

    assert list(portfolio.assets.keys()) == ["GOLD", "BTC"]
    assert isinstance(portfolio.join_data(["GOLD", "BTC"]), pd.DataFrame)
    with pytest.raises(KeyError):
        portfolio.join_data(["GOLD", "BTC", "WRONG_TICKER"])

    with pytest.raises(ValueError):
        portfolio.add_asset(Asset(ticker = "GOLD",
                                  # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                                  data = pd.DataFrame({"a": [1,2], "b": [3,4]}).rename_axis(index=dataloader.time_index_col),
                                  index_name = dataloader.time_index_col))
    with pytest.raises(ValueError):
        portfolio.update_asset(Asset(ticker = "WRONG_TICKER",
                                     # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                                     data = pd.DataFrame({"a": [1,2], "b": [3,4]}).rename_axis(index=dataloader.time_index_col),
                                     index_name = dataloader.time_index_col))

    with pytest.warns(Warning):
        portfolio.update_asset(Asset(ticker = "GOLD",
                                     # data = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'}),
                                     data = pd.DataFrame({"a": [1,2], "b": [3,4]}).rename_axis(index=dataloader.time_index_col),
                                     index_name = "WRONG_INDEX"))
    with pytest.raises(ValueError):
        portfolio.join_data(["GOLD", "BTC"])


