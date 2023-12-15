import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum
import re
import warnings

warnings.filterwarnings("ignore")


class MarketAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


class Rule(BaseModel):
    id: str = Field(...)
    condition: str = Field(...)
    operator: Optional[str] = Field(None)
    action: MarketAction = Field(MarketAction.BUY)
    action_quantity: Union[int, float, str] = Field(1)
    equation: Optional[str] = Field(None)


class Operator(Enum):
    AND = "&"
    OR = "|"
    NOT = "~"


class Strategy:
    def __init__(
        self,
        name: str,
        cash: float = 10000,
        commission: float = 0.01,
        min_positions: float = 0.1,
    ):
        """
        Creates a strategy object.

        Parameters
        ----------
        name : str,
                name of a strategy. Required.
        cash : float,
                Starting cash amount. Default is 10000.
        commission : float,
                Commission per trade. Default is 0.01. Total commission is calculated as commission * quantity.
        min_positions : float,
                Minimum possible quantity of the asset to be bought or sold. Default is 0.1.
        """

        if not isinstance(name, str):
            raise ValueError(f"{name} is not a valid name")
        if not isinstance(cash, (int, float)):
            raise ValueError(f"{cash} is not a valid cash amount")
        if not isinstance(commission, (int, float)):
            raise ValueError(f"{commission} is not a valid commission amount")
        if not isinstance(min_positions, (int, float)):
            raise ValueError(f"{min_positions} is not a valid min_positions amount")

        self.name = name
        self._data = None
        self._rules = {MarketAction.BUY: "", MarketAction.SELL: ""}
        self._raw_rules = {MarketAction.BUY: [], MarketAction.SELL: []}
        self._eval_rules = {MarketAction.BUY: "", MarketAction.SELL: ""}
        self._date_column = "date"
        self._curr_data = None
        self._curr_result = None
        self._starting_cash = cash
        self._cash = cash
        self._commission = commission
        self._min_data_needed = 0
        self._positions = 0
        self._curr_pos = []
        self._history = []
        self.min_positions = min_positions

    @property
    def positions(self):
        return self._positions

    @property
    def history_df(self):
        hist_cols = [
            "date",
            "action",
            "quantity",
            "close_price",
            "commission",
            "amount",
            "positions",
            "cash",
            "portfolio_value",
        ]
        hdf = pd.DataFrame(data=self._history, columns=hist_cols)
        return hdf

    def __repr__(self):
        return f"Strategy: {self.name}"

    def load_data(self, data: pd.DataFrame, date_column: str = "date"):
        """
        Loads the data into the strategy.
        Parameters
        ----------
        data : pd.DataFrame
            The data to be loaded into the strategy.
        date_column : str, optional
            The column name of the date column, by default 'date'.

        Returns
        -------
        None
        """
        # convert all columns to upper case
        # data.columns = data.columns.str.upper()
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{data} is not a valid dataframe")
        if date_column not in data.columns:
            raise ValueError(f"{date_column} is not a valid date column")

        self._date_column = date_column
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def rules(self) -> Dict[MarketAction, List[Rule]]:
        return self._rules

    def _refine_rules(self):
        """
        Refine rule by removing last operator if needed, make the rule easy to run in the next method
        as easy as self._curr_data['SMA_30'][-1] > self._curr_data['SMA_50'][-1].

        """

        for action, rule in self._raw_rules.items():
            if len(rule) == 0:
                self._eval_rules[action] = "False"

                continue
            self._eval_rules[action] += "("
            for i, r in enumerate(rule):
                if i == len(rule) - 1:
                    self._eval_rules[action] += f"{r.equation})"
                else:
                    self._eval_rules[action] += f"{r.equation}) {r.operator} ("

        # print(self._eval_rules)

    def add_single_rule(
        self,
        condition: str = "CLOSE[N-1]>CLOSE_30[N-10]",
        operator: Optional[Operator] = Operator.AND,
        action: MarketAction = MarketAction.BUY,
        action_quantity: Union[int, float, str] = 1,
    ) -> Rule:
        """
        Creates a single rule and adds it to the strategy.

        Parameters
        ----------
        condition : str
            The condition to be added, by default 'SMA_30[N-1]>SMA_50[N-10]', example: 'CLOSE[N-1]>=100'
            Where, SMA_30 is a column available in the data and N-1 is the last row of the data.
            Possible symbols are >, <, >=, <=, ==, !=
        operator : Optional[Operator], optional
            The operator to be used to combine the condition, by default Operator.AND.
            When this rule is the last rule, this parameter is ignored. But when there are multiple rules, this parameter is used to combine the rules.
        action : MarketAction, optional
            The action to be taken if the condition is true, by default MarketAction.BUY.
            This parameter puts rule into one of the action group.
        action_quantity : Union[int, float, str], optional
            The quantity of the asset to be bought or sold, by default 1. If used str, it should be in the format of '30%', 'ALL', '20%'.
            Only needed when this rule is the last rule in the action group.

        **Note**:
        - The rules will be combined and only the quantity of the last rule will be used.
        - First the rules are grouped by the action and then the rules are combined using the operator in FIFO order.
        - When action_quantity is in %, it is calculated based on the cash available at that time.

        Returns
        -------
        rule : Rule
        """
        # ensure that the column is present in the data, else raise error
        # first split by ]
        # then split by [
        # get column names from the condition

        columns = re.findall(r"[\w.]+", condition)
        symbol = re.findall(r"[<>=!]+", condition)[0]
        first_part, second_part = [c.strip() for c in condition.split(symbol)]
        col1 = first_part.split("[N-")[0]
        ind1 = first_part.split("[N-")[1].split("]")[0]

        # check if the column is present in the data
        if self._data is None:
            raise ValueError("Data is not loaded into the strategy")
        if col1 not in self._data.columns:
            raise ValueError(f"{col1} is not present in the data")
        # check if the index is number
        if not ind1.isdigit():
            raise ValueError(f"{ind1} is not a valid index")
        ind1 = int(ind1)
        self._min_data_needed = max(self._min_data_needed, int(ind1))

        col2 = None
        if second_part.isdigit():
            value = second_part
        else:
            col2 = second_part.split("[N-")[0]
            ind2 = second_part.split("[N-")[1].split("]")[0]
            self._min_data_needed = max(self._min_data_needed, int(ind2))
        if col2:
            if col2 not in self._data.columns:
                raise ValueError(f"{col2} is not present in the data")
            if not ind2.isdigit():
                raise ValueError(f"{ind2} is not a valid index")
            ind2 = int(ind2)
            equation = f"self._curr_data['{col1}'].iloc[-{ind1}] {symbol} self._curr_data['{col2}'].iloc[-{ind2}]"
        else:
            value = columns[3]
            equation = f"self._curr_data['{col1}'].iloc[-{ind1}] {symbol} {value}"
        # print(col1, ind1, symbol)

        # check if the symbol is valid
        if symbol not in [">", "<", ">=", "<=", "==", "!="]:
            raise ValueError(f"{symbol} is not a valid symbol")
        # check if the operator is valid
        if operator not in [Operator.AND, Operator.OR, Operator.NOT]:
            raise ValueError(f"{operator} is not a valid operator")
        # check if the action is valid
        if action not in [MarketAction.BUY, MarketAction.SELL]:
            raise ValueError(f"{action} is not a valid action")
        # check if the action quantity is valid
        if not isinstance(action_quantity, (int, float, str)):
            raise ValueError(f"{action_quantity} is not a valid action quantity")
        if isinstance(action_quantity, (str)):
            if not action_quantity.endswith("%") and action_quantity != "ALL":
                raise ValueError(f"{action_quantity} is not a valid action quantity")

        # create the rule
        rule_id = sum([len(self._raw_rules[a]) for a in self._raw_rules]) + 1
        rule = Rule(
            id=f"{self.name}_rule_{rule_id}",
            condition=condition,
            equation=equation.replace("AND", "&")
            .replace("OR", "|")
            .replace("NOT", "~"),
            operator=operator,
            action=action,
            action_quantity=action_quantity,
        )

        self._raw_rules[action].append(rule)
        return rule

    def _buy(self):
        """
        Performs the buy action based on the quantity specified in the rule.
        """
        quantity = self._raw_rules[MarketAction.BUY][-1].action_quantity
        close_price = self._curr_data["CLOSE"].iloc[-1]
        close_date = self._curr_data[self._date_column].iloc[-1]
        buyable_quantity_without_commission = self._cash / close_price

        if self._cash < 0:
            return None

        if quantity == "ALL":
            commission_needed = self._commission * buyable_quantity_without_commission
            new_cash = self._cash - commission_needed
            quantity = new_cash / close_price
            commission = self._commission * quantity

        elif isinstance(quantity, int) or isinstance(quantity, float):
            quantity = int(quantity)
            commission = self._commission * quantity

        elif quantity.endswith("%"):
            use_amt = self._cash * float(quantity[:-1]) / 100
            quantity = use_amt / close_price
            commission = self._commission * quantity

        else:
            raise ValueError(
                f"{self._rules[MarketAction.BUY].action_quantity} is not a valid quantity."
            )

        if quantity < self.min_positions:
            return None
        self._positions += quantity
        amount = commission + quantity * close_price
        self._cash = self._cash - amount
        portfolio_value = self._cash + self._positions * close_price

        self._curr_pos = [
            close_date,
            "BUY",
            quantity,
            close_price,
            commission,
            amount,
            self._positions,
            self._cash,
            portfolio_value,
        ]
        self._history.append(self._curr_pos)

    def _sell(self):
        """
        When the rule is true, this method is called.
        It sells the asset based on the quantity specified in the rule.
        """
        if self._positions == 0:
            return
        close_price = self._curr_data["CLOSE"].iloc[-1]
        quantity = self._raw_rules[MarketAction.SELL][-1].action_quantity
        close_date = self._curr_data[self._date_column].iloc[-1]

        if quantity == "ALL":
            commission = self._commission * self._positions
            amount = self._positions * close_price - commission
            quantity = self._positions

        elif isinstance(quantity, int) or isinstance(quantity, float):
            quantity = int(quantity)
            if quantity > self._positions:
                quantity = self._positions
            commission = self._commission * quantity
            amount = quantity * close_price - commission

        elif quantity.endswith("%"):
            quantity = float(quantity[:-1]) / 100
            quantity = quantity * self._positions
            commission = self._commission * quantity
            amount = quantity * close_price - commission

        if self.min_positions > quantity:
            return None
        self._cash = self._cash + amount
        self._positions -= quantity
        portfolio_value = self._cash + self._positions * close_price

        self._curr_pos = [
            close_date,
            "SELL",
            quantity,
            close_price,
            commission,
            amount,
            self._positions,
            self._cash,
            portfolio_value,
        ]

        self._history.append(self._curr_pos)

    def _hold(self):
        """
        Holds the position. Does nothing.
        """
        pass

    def next(self):
        """
        Goes to the next row of the data and checks if the rule is true.
        """
        is_buy = eval(self._eval_rules[MarketAction.BUY])
        is_sell = eval(self._eval_rules[MarketAction.SELL])
        if is_buy:
            self._buy()
        elif is_sell:
            self._sell()
        else:
            self._hold()

    def simulate(self):
        """
        Simulates the strategy on the data provided.
        """
        self._curr_data = self._data.iloc[: self._min_data_needed]
        self._refine_rules()

        # loop through the data
        # for each row, check if the rule is true
        # curr data should contain only the data needed for the rule
        # try to make it deque

        for i in range(self._min_data_needed, len(self._data)):
            self._curr_data = self._data.iloc[:i]
            self.next()
        pass

    def summary(self):
        """
        Prints the summary of the strategy.
        """
        if len(self.history_df) == 0:
            print("No trades were made.")
            return None
        first_trade_date = self.history_df["date"].iloc[0]
        last_trade_date = self.history_df["date"].iloc[-1]
        total_trades = len(self.history_df)
        final_pnl = self.history_df["portfolio_value"].iloc[-1] - self._starting_cash
        final_portfolio_value = self.history_df["portfolio_value"].iloc[-1]
        total_commission = self.history_df["commission"].sum()
        total_buy_commission = self.history_df[self.history_df["action"] == "BUY"][
            "commission"
        ].sum()
        total_sell_commission = self.history_df[self.history_df["action"] == "SELL"][
            "commission"
        ].sum()
        total_buy_amount = self.history_df[self.history_df["action"] == "BUY"][
            "amount"
        ].sum()
        total_sell_amount = self.history_df[self.history_df["action"] == "SELL"][
            "amount"
        ].sum()
        total_buy_quantity = self.history_df[self.history_df["action"] == "BUY"][
            "quantity"
        ].sum()
        total_sell_quantity = self.history_df[self.history_df["action"] == "SELL"][
            "quantity"
        ].sum()
        total_buy_trades = len(self.history_df[self.history_df["action"] == "BUY"])
        total_sell_trades = len(self.history_df[self.history_df["action"] == "SELL"])

        print(f"Strategy Name: {self.name}")
        print(f"First Trade Date: {first_trade_date}")
        print(f"Last Trade Date: {last_trade_date}")
        print(f"Total Trades: {total_trades}")
        print(f"Final PnL: {final_pnl}")
        print(f"Starting Cash: {self._starting_cash}")
        print(f"Final Cash: {self._cash}")
        print(f"Final Positions: {self._positions}")
        print(f"Final Portfolio Value: {final_portfolio_value}")
        print(f"Total Commission: {total_commission}")
        print(f"Total Buy Commission: {total_buy_commission}")
        print(f"Total Sell Commission: {total_sell_commission}")
        print(f"Total Buy Amount: {total_buy_amount}")
        print(f"Total Sell Amount: {total_sell_amount}")
        print(f"Total Buy Quantity: {total_buy_quantity}")
        print(f"Total Sell Quantity: {total_sell_quantity}")
        print(f"Total Buy Trades: {total_buy_trades}")
        print(f"Total Sell Trades: {total_sell_trades}")

    def plot_history(
        self,
        title: str = "Trades",
        yaxis_title: str = "Stock's Price in USD",
        xaxis_title: str = "Date",
        plot_columns: Optional[List[str]] = None,
    ) -> "plotly.graph_objects.Figure  ":
        """
        Plots the history of the strategy. The plot contains the market data, trades and the portfolio value.

        Parameters
        ----------
        title : str, optional
            Title of the plot, by default 'Trades'.
        yaxis_title : str, optional
            Y axis title of the plot, by default 'Stock's Price in USD'.
        xaxis_title : str, optional
            X axis title of the plot, by default 'Date'.
        plot_columns : Optional[List[str]], optional
            List of columns to be plotted, by default None. If None, no columns are plotted.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The plotly figure object.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except:
            print("Please install plotly before plotting.")
            return None

        if len(self.history_df) == 0:
            print("No trades were made.")
            return None

        if plot_columns:
            for col in plot_columns:
                if col not in self.data.columns:
                    raise ValueError(f"{col} is not present in the data")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        hist = self.history_df.copy()
        hist["hover_text"] = (
            hist.action
            + " "
            + hist.quantity.round(4).astype(str)
            + " at "
            + hist.close_price.round(4).astype(str)
            + " Amount: "
            + hist.amount.round(4).astype(str)
            + " Cash: "
            + hist.cash.round(4).astype(str)
        )

        ldf = self.data.copy()
        fig.add_trace(
            go.Candlestick(
                x=ldf["DATE"],
                open=ldf["OPEN"],
                high=ldf["HIGHT"],
                low=ldf["LOW"],
                close=ldf["CLOSE"],
                name="OHLC Market Data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=[ldf["CLOSE"].max() + 10] * len(hist),
                mode="markers",
                name="Trades",
                marker=dict(
                    symbol=hist["action"].apply(
                        lambda x: "triangle-up" if x == "BUY" else "triangle-down"
                    ),
                    size=10,
                    color=hist["action"].apply(
                        lambda x: "green" if x == "BUY" else "red"
                    ),
                ),
                hovertext=hist.hover_text,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["portfolio_value"],
                name="Portfolio Value",
                mode="lines",
            ),
            secondary_y=True,
        )
        if plot_columns:
            for col in plot_columns:
                fig.add_trace(
                    go.Scatter(
                        x=ldf["DATE"],
                        y=ldf[col],
                        name=col,
                        mode="lines",
                    ),
                    secondary_y=False,
                )
        fig.update_layout(title=title, yaxis_title=yaxis_title, xaxis_title=xaxis_title)

        # fig.show()
        self._figure = fig

        return fig


if __name__ == "__main__":
    st = Strategy("test", cash=1000, commission=0.01, min_positions=0.5)
    # create dummy dataframe with data
    data = pd.DataFrame(columns=["date", "SMA_30", "SMA_50", "EMA", "CLOSE"])
    data["date"] = pd.date_range(start="2021-01-01", periods=100)
    data["SMA_30"] = np.linspace(100, 200, 100)
    data["SMA_50"] = np.random.randint(80, 150, 100)
    data["SMA_10"] = np.linspace(100, 120, 100)
    data["CLOSE"] = np.random.randint(100, 200, 100)

    st.load_data(
        data=data,
        date_column="date",
    )
    st.add_single_rule(
        "SMA_30[N-1]>SMA_50[N-10]",
        operator=Operator.OR,
        action_quantity="30%",
        action=MarketAction.SELL,
    )
    st.add_single_rule(
        "CLOSE[N-1]<SMA_10[N-10]", action_quantity="20%", action=MarketAction.BUY
    )
    st.simulate()

    print(st.history_df)
