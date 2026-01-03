import os

import pandas as pd
import matplotlib.pyplot as plt

import lib.common
from lib import backtest


def strategy_sma(df: pd.DataFrame) -> pd.DataFrame:
    """A moving average crossover strategy."""

    df['SMA5'] = df["Close"].rolling(window = 5).mean()
    df['SMA30'] = df["Close"].rolling(window = 30).mean()

    df['signal'] = 0
    condition: pd.Series = df['SMA5'] > df['SMA30']

    df.loc[condition, "signal"] = 1

    df['returns'] = df['signal'] * df['pct_close_futur']

    plt.figure(figsize=lib.common.FIG_SIZE)
    plt.plot(df['Close'], label='Closing Price', linestyle='dotted', color='black')
    plt.plot(df['SMA5'], label='SMA 5', linestyle='dotted', color='red')
    plt.plot(df['SMA30'], label='SMA 30', linestyle='dotted', color='green')
    plt.xticks(rotation=70)
    plt.legend()
    plt.grid()

    plt.show()

    return df


def main() -> None:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../Tickers/AAPL.csv"),
                     index_col="Date",
                     parse_dates=True)

    # Dropping NaNs.
    len_before = len(df)
    df = df.dropna()
    print(f"Dropped from `df': {len_before - len(df)} rows, out of total {len_before}.")

    # Standardize and drop features.
    df = df.rename(columns={"Date":         "time",
                            "Open":         "open",
                            "High":         "high",
                            "Low":          "low",
                            "Adj Close":    "close"})
    df.drop(("Ticker", "Volume", "Close"))

    df["pct_close_futur"] = (df["close"].shift(-2) - df["close"]) / df["close"]

    df = strategy_sma(df)

    backtest.backtest(df, "sma")


if __name__ == "__main__":
    main()