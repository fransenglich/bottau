import ta

from lib.AbstractStrategy import AbstractStrategy


class StrategySMA_RSI(AbstractStrategy):
    """A SMA and RSI technical analysis-strategy."""

    def compute_features(self):
        self.data["SMA_30"] = ta.trend.SMAIndicator(self.data["close"], 30) \
            .sma_indicator()
        self.data["SMA_5"] = ta.trend.SMAIndicator(self.data["close"], 5) \
            .sma_indicator()
        self.data["RSI"] = ta.momentum.RSIIndicator(self.data["close"], 10) \
            .rsi()  # TODO RSI constant

        condition = self.data['SMA5'] > self.data['SMA30']

    def display(self):
        pass
