import sys

sys.path.append(".")
from src.finance_ml.backtests.strategy import Rule, Strategy, Operator, MarketAction
import pandas as pd
import numpy as np
import pytest


def test_strategy_class():
    strategy = Strategy("Some test Strategy")
    assert isinstance(strategy, Strategy)


def test_add_single_rule():
    strategy = Strategy("Some test Strategy")

    data = pd.DataFrame(columns=["date", "SMA_30", "SMA_50", "EMA", "CLOSE"])
    data["date"] = pd.date_range(start="2021-01-01", periods=100)
    data["SMA_30"] = np.linspace(100, 200, 100)
    data["SMA_50"] = np.random.randint(80, 150, 100)
    data["SMA_10"] = np.linspace(100, 120, 100)
    data["CLOSE"] = np.random.randint(100, 200, 100)
    strategy.load_data(data=data, date_column="date")

    rule = strategy.add_single_rule(
        "SMA_30[N-1]>SMA_50[N-10]",
        operator=Operator.OR,
        action_quantity="30%",
        action=MarketAction.SELL,
    )

    assert isinstance(rule, Rule)
    assert rule.operator == Operator.OR.value
    assert rule.action_quantity == "30%"
    assert rule.action == MarketAction.SELL


def test_add_refine_rules():
    strategy = Strategy("Some test Strategy")

    data = pd.DataFrame(columns=["date", "SMA_30", "SMA_50", "EMA", "CLOSE"])
    data["date"] = pd.date_range(start="2021-01-01", periods=100)
    data["SMA_30"] = np.linspace(100, 200, 100)
    data["SMA_50"] = np.random.randint(80, 150, 100)
    data["SMA_10"] = np.linspace(100, 120, 100)
    data["CLOSE"] = np.random.randint(100, 200, 100)
    strategy.load_data(data=data, date_column="date")

    rule1 = strategy.add_single_rule(
        "SMA_30[N-1]>SMA_50[N-10]",
        operator=Operator.AND,
        action_quantity="30%",
        action=MarketAction.SELL,
    )
    rule2 = strategy.add_single_rule(
        "SMA_30[N-1]>SMA_50[N-1]",
        operator=Operator.OR,
        action_quantity="10%",
        action=MarketAction.SELL,
    )

    strategy._refine_rules()
    assert len(strategy.rules) == 2
    assert strategy._eval_rules[MarketAction.SELL].replace(
        " ", ""
    ) == "(self._curr_data['SMA_30'].iloc[-1] > self._curr_data['SMA_50'].iloc[-10]) & (self._curr_data['SMA_30'].iloc[-1] > self._curr_data['SMA_50'].iloc[-1])".replace(
        " ", ""
    )
    assert strategy._eval_rules[MarketAction.BUY] == "False"


def test_load_data():
    strategy = Strategy("Some test Strategy")
    data = pd.DataFrame(columns=["date", "SMA_30", "SMA_50", "EMA", "CLOSE"])
    data["date"] = pd.date_range(start="2021-01-01", periods=100)
    data["SMA_30"] = np.linspace(100, 200, 100)
    data["SMA_50"] = np.random.randint(80, 150, 100)
    data["EMA"] = np.linspace(100, 120, 100)
    data["CLOSE"] = np.random.randint(100, 200, 100)

    strategy.load_data(
        data=data,
        date_column="date",
    )
    assert strategy.data.shape == (100, 5)
    assert strategy._date_column == "date"


def test_simulate():
    st = Strategy("Some test Strategy")
    data = pd.DataFrame(columns=["date", "CLOSE"])
    data["date"] = pd.date_range(start="2021-01-01", periods=10)
    data["CLOSE"] = np.array([100, 90, 100, 99, 105, 100, 110, 90, 100, 110])

    st.load_data(
        data=data,
        date_column="date",
    )
    st.add_single_rule(
        "CLOSE[N-1]>CLOSE[N-2]",
        operator=Operator.OR,
        action_quantity="ALL",
        action=MarketAction.SELL,
    )
    st.add_single_rule(
        "CLOSE[N-1]<CLOSE[N-2]", action_quantity="ALL", action=MarketAction.BUY
    )
    # based on above close, we buy when price decrease and sell when price increase.
    # so we should have 4 buy and 4 sell
    st.simulate()

    assert len(st.history_df.query("action=='BUY'")) == 4
    assert len(st.history_df.query("action=='SELL'")) == 4
    assert st.history_df["action"].iloc[0] == MarketAction.BUY.value
    assert st.history_df["action"].iloc[1] == MarketAction.SELL.value


def test_strategy_name():
    with pytest.raises(ValueError):
        Strategy(123, "invalid action")


def test_strategy_rule_before_load():
    st = Strategy("some st")
    with pytest.raises(Exception):
        st.add_single_rule("some rule", action="invalid action")


def test_strategy_invalid_column():
    st = Strategy("some st")
    with pytest.raises(ValueError):
        df = pd.DataFrame(columns=["date", "CLOSE"])
        df["date"] = pd.date_range(start="2021-01-01", periods=10)
        st.load_data(data=df, date_column="datee")


def test_strategy_invalid_data():
    st = Strategy("some st")
    with pytest.raises(ValueError):
        st.load_data(data="Wrong data", date_column=123)


def test_strategy_invalid_rule_params():
    st = Strategy("some st")
    with pytest.raises(Exception):
        df = pd.DataFrame(columns=["date", "CLOSE"])
        df["date"] = pd.date_range(start="2021-01-01", periods=10)
        st.load_data(data=df, date_column="date")
        st.add_single_rule(
            "CLOSE[N-1]>CLOSE[N-2]",
            operator="or",
            action_quantity="ALLLLLL",
            action="SELL",
        )


def test_plot_history():
    st = Strategy("some st")
    df = pd.DataFrame(columns=["date", "CLOSE"])
    df["date"] = pd.date_range(start="2021-01-01", periods=10)
    df["CLOSE"] = np.array([100, 90, 100, 99, 105, 100, 110, 90, 100, 110])
    st.load_data(data=df, date_column="date")
    st.add_single_rule(
        "CLOSE[N-1]>CLOSE[N-2]",
        operator=Operator.OR,
        action_quantity="ALL",
        action=MarketAction.SELL,
    )
    st.add_single_rule(
        "CLOSE[N-1]<CLOSE[N-2]", action_quantity="ALL", action=MarketAction.BUY
    )
    st.simulate()

    with pytest.raises(ValueError):
        st.plot_history(columns=["date", "CLOSEe"])
