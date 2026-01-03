import pandas as pd
import ta
import matplotlib.pyplot as plt

from lib import common


def strategy_Bollinger_RSI(df: pd.DataFrame,
                           param_window: int = 20) -> pd.DataFrame:
    """A strategy built on conditions involving Bollinger bands and RSI."""

    # Bollinger Bands
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'],
                                                   window=int(param_window))
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'],
                                                   window=int(param_window))
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'],
                                                   window=int(param_window))

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Create our signal
    df["signal"] = 0

    # Define our conditions
    condition_1_buy = df["Close"] < df["BB_Lower"]
    condition_1_sell = df["BB_Upper"] < df["Close"]

    condition_2_buy = df["RSI"] < 30
    condition_2_sell = df["RSI"] > 70

    # Apply our conditions
    df.loc[condition_1_buy & condition_2_buy, "signal"] = 1
    df.loc[condition_1_sell & condition_2_sell, "signal"] = -1

    # Compute our exit signal
    df["pct_close_futur"] = (df["Close"].shift(-2) - df["Close"])/df["Close"]

    # Compute the returns of each position
    # This is our computed trading signal applied to the returns
    df["returns"] = df["signal"] * df["pct_close_futur"]

    # Plot price with moving averages and Bollinger Bands
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['Close'], label='Closing Price', color='black')
    plt.plot(df['BB_Upper'], label='BB Upper', linestyle='dotted', color='red')
    plt.plot(df['BB_Lower'], label='BB Lower', linestyle='dotted',
             color='green')

    # Plot Buy/Sell Signals
    plt.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1],
                marker='v', color='red', label='Sell Signal', s=50)
    plt.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1],
                marker='^', color='green', label='Buy Signal', s=50)

    plt.legend()
    plt.title("Bollinger Bands on Closing Prices - AAPL")
    plt.ylabel("Price")
    plt.grid()
    common.savefig(plt, "feature_BollingerBands")

    # Plot RSI
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='dashed', color='red', alpha=0.5)
    plt.axhline(30, linestyle='dashed', color='green', alpha=0.5)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    common.savefig(plt, "feature_RSI")

    return df