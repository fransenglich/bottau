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

        # Shifts forward, calling it "retarded" makes no sense.
        self.data["RSI_retarded"] = self.data["RSI"].shift(1)

        # df["pct_close_futur"] = (df["close"].shift(-2) - df["close"]) / df["close"]
        self.data["pct_close_futur"] = self.data["close"].pct_change()

        condition_1_buy = self.data["SMA_5"] < self.data["SMA_30"]
        condition_1_sell = self.data["SMA_5"] > self.data["SMA_30"]

        condition_2_buy = self.data["RSI"] > self.data["RSI_retarded"]
        condition_2_sell = self.data["RSI"] < self.data["RSI_retarded"]

        # For security we only buy if two conditions are true, same for sell.
        self.data.loc[condition_1_buy & condition_2_buy, "signal"] = 1
        self.data.loc[condition_1_sell & condition_2_sell, "signal"] = -1

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
