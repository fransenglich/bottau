import ta

from lib.AbstractStrategy import AbstractStrategy


class StrategySMARSI(AbstractStrategy):
    """A SMA and RSI technical analysis-strategy."""

    def compute_features(self):
        self.data["SMA_30"] = ta.trend.SMAIndicator(self.data["close"], 30).sma_indicator()