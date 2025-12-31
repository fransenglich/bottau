import pandas as pd
import matplotlib.pyplot as plt

import common


def strategy_sma(df: pd.DataFrame) -> pd.DataFrame:
    """A moving average crossover strategy."""

    df['SMA5'] = df["Close"].rolling(window = 5).mean()
    df['SMA30'] = df["Close"].rolling(window = 30).mean()

    df['signal'] = 0
    condition: pd.Series = df['SMA5'] > df['SMA30']

    df.loc[condition, "signal"] = 1

    df['returns'] = df['signal'] * df['pct_close_futur']

    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['Close'], label='Closing Price', linestyle='dotted', color='black')
    plt.plot(df['SMA5'], label='SMA 5', linestyle='dotted', color='red')
    plt.plot(df['SMA30'], label='SMA 30', linestyle='dotted', color='green')
    plt.xticks(rotation=70)
    plt.legend()
    plt.grid()

    plt.show()

    return df