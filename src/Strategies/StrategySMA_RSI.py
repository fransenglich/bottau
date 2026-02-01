import ta
import matplotlib.pyplot as plt

from lib.AbstractStrategy import AbstractStrategy
from lib import common


class StrategySMA_RSI(AbstractStrategy):
    """A SMA and RSI technical analysis-strategy."""

    def prepare_features(self) -> None:
        self.data["SMA_30"] = ta.trend.SMAIndicator(self.data["close"], 30) \
            .sma_indicator()
        self.data["SMA_5"] = ta.trend.SMAIndicator(self.data["close"], 5) \
            .sma_indicator()
        self.data["RSI"] = ta.momentum.RSIIndicator(self.data["close"], 10) \
            .rsi()  # TODO RSI constant

        # condition = self.data['SMA5'] > self.data['SMA30']

    def display(self) -> None:
        plt.figure(figsize=common.FIG_SIZE)
        plt.plot(self.data['close'], label='Closing Price', linestyle='dotted', color='black')
        plt.plot(self.data['SMA_5'], label='SMA 5', linestyle='dotted', color='red')
        plt.plot(self.data['SMA_30'], label='SMA 30', linestyle='dotted', color='green')
        plt.xticks(rotation=70)
        plt.legend()
        plt.grid()
        plt.show()
