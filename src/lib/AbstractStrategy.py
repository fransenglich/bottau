from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from lib import common


class AbstractStrategy(ABC):
    """Abstract base class for all strategies."""

    data: pd.DataFrame
    """Contains the in-sample data and is populated with features. See
    README.md for documentation on standardised feature and data names."""

    parameters: dict[str, Any]
    """Holds the parameters. The key is the name."""

    def __init__(self, df: pd.DataFrame, parameters: dict[str, Any]):
        self.data = df
        self.parameters = parameters

    @abstractmethod
    def prepare_features(self) -> None:
        """Computes the features and stores them as member data."""
        pass

    @abstractmethod
    def display(self) -> None:
        """Displays graphs or plots and/or prints data for the purpose of
        diagnostics.

        This is typically done by showing graphs with Matplotlib. Typically you
        need to call prepare_features() before running this diagnostic tool."""
        pass

    def backtest(self) -> None:
        """Does a traditional backtest and displays/outputs it.
        
        It expects the column 'returns', typically computed in
        prepare_features(), which hence needs to be called before."""

        # TODO whether to simulate position sizing, slippage, transaction costs, etc.
        # TODO drawdown
        plt.figure(figsize=common.FIG_SIZE)

        # plt.plot(self.data['close'], label='Closing Price', color='black')

        pass