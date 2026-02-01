from typing import Any

import pandas as pd

from abc import ABC, abstractmethod


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
