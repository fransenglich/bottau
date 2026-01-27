import ta

from lib.AbstractStrategy import AbstractStrategy


class StrategySMARSI(AbstractStrategy):
    """A SMA and RSI technical analysis-strategy."""

    # FEATURE 1: SMA 30
    def compute_features(self):
        self.data["SMA_30"] = ta.trend.SMAIndicator(self.data["close"], 30) \
                                       .sma_indicator()
