import pandas as pd

from abc import ABC, abstractmethod

class AbstractStrategy(ABC):
    """Abstract base class for all strategies."""

    data: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.data = df


    @abstractmethod
    def compute_features(self):
        """Computes the features and stores them as member data."""
        pass
