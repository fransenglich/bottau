from abc import ABC, abstractmethod

class AbstractStrategy(ABC):
    """Abstract base class for all strategies."""

    @abstractmethod
    def compute_features(self):
        """Computes the features and stores them as member data."""
        pass
