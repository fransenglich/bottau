from typing import Any
import logging
import logging.config
import os

import pandas as pd
import matplotlib.pyplot as plt

# Set up logging. We do this before importing our own modules.
logging.config.fileConfig(os.path.join(os.path.dirname(__file__),
                                       "../logging.conf"))
logger = logging.getLogger('BotTau')

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
        logger.info(f"Initialized strategy {self.__class__.__name__}")

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

        if not "returns" in self.data.columns:
            raise ValueError("No 'returns' column.")

        # TODO whether to simulate position sizing, slippage, transaction costs, etc.
        # TODO drawdown

        # Copied from backtest.py.
        plt.figure(figsize=common.FIG_SIZE)
        plt.plot(self.data['returns'], label='Returns')
        plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
        plt.title("Returns")
        plt.ylabel("Returns")
        plt.legend()
        plt.grid()
        plt.show()
        # common.savefig(plt, "returns", sn)

        # plt.plot(self.data['close'], label='Closing Price', color='black')

        pass