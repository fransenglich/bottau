import pandas as pd
import ta
import matplotlib.pyplot as plt

from lib.AbstractStrategy import AbstractStrategy
from lib import common


class StrategySMA_RSI(AbstractStrategy):
    """An SMA and RSI technical analysis-strategy."""

    def prepare_features(self) -> None:
        self.data["SMA_30"] = ta.trend.SMAIndicator(self.data["close"], 30) \
            .sma_indicator()
        self.data["SMA_5"] = ta.trend.SMAIndicator(self.data["close"], 5) \
            .sma_indicator()
        self.data["RSI"] = ta.momentum.RSIIndicator(self.data["close"], 10) \
            .rsi()  # TODO RSI constant

        # df["pct_close_futur"] = (df["close"].shift(-2) - df["close"]) / df["close"]
        self.data["pct_close_futur"] = self.data["close"].pct_change()

        # Logic: when SMA 5 crosses above SMA 30, buy. I.e, 
        condition = self.data['SMA_5'] > self.data['SMA_30']
        self.data.loc[condition, "signal"] = 1

        self.data['returns'] = self.data['signal'] * self.data['pct_close_futur']

    def display(self) -> None:
        plt.figure(figsize=common.FIG_SIZE)
        plt.plot(self.data['close'], label='Closing Price', color='black')
        plt.plot(self.data['SMA_5'], label='SMA 5', linestyle='dotted', color='red')
        plt.plot(self.data['SMA_30'], label='SMA 30', linestyle='dotted', color='green')
        plt.xticks(rotation=70)
        plt.legend()
        plt.grid()
        plt.show()
