import yfinance as yf
import matplotlib.pyplot as plt
import sys
import ta
import pandas as pd
from datetime import datetime, timedelta

tickers = ['AAPL', 'TSLA']

# Data Frames of tickers, comes from Yahoo Finance and are in classic OHLC(Adj)V
# format.
df_tickers = []

def initialDownload() -> None:
    """ Fetches for the tickers in `tickers` and writes them out to CSV-files in Tickers/.

    We build simple DataFrames, one header like what is typical.
    Discussed here:
    https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801
    """
    for ticker in tickers:
        #df = yf.download(ticker, period='1y', interval='1d', auto_adjust=False)
        df = yf.download(ticker, period='1y', group_by='Ticker', interval='1d', auto_adjust=False)
        df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        #df['ticker'] = ticker
        df.to_csv(f'Tickers/{ticker}.csv')
        df_tickers.append(df)


def fetchNewTicks() -> None:
    """Updates the tickers to today."""
    for ticker in tickers:
        df = pd.read_csv(f'Tickers/{ticker}.csv')

        date_item = df['Date'].tail(1).item()
        startDate = (datetime.strptime(date_item, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

        print("FOO: " + str(startDate))

        updated_df = yf.download(ticker, period='1y', group_by='Ticker', interval='1d', auto_adjust=False, start=startDate)
        updated_df = updated_df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        #updated_df['ticker'] = ticker

        print(df)

        print(updated_df)

        merged = pd.concat([df, updated_df])

        #print(merged)

        merged.to_csv(f'Tickers/{ticker}_merged.csv')

        df_tickers.append(merged)

# TODO:
# - Fix fetchNewTicks()

def downloadToFile():
    df = yf.download("AAPL", interval='1d').loc["2015":]
    df.to_csv("Tickers/AAPL.manual.csv")

def main() -> int:
    global df_tickers

    if len(sys.argv) == 2:
        if sys.argv[1] == "i":
            initialDownload()
        elif sys.argv[1] == "c":
            fetchNewTicks()
        else:
            raise Exception("No or wrong commandline argument passed.")
    else:
        df = pd.read_csv("Tickers/IBM.csv")
        df_tickers.append(df)

    df = df_tickers[0]


    # For some reason the name differs.
    df = df.rename(columns = {"4. close": "Close",
                              "1. open": "Open",
                              "2. high": "High",
                              "3. low": "Low",
                              "5. volume": "Volume"})

    df

    # Bollinger Bands
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'], window=20)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20)

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Create our signal
    df["signal"] = 0

    # Define our conditions
    condition_1_buy = df["Close"] < df["BB_Lower"]
    condition_1_sell = df["BB_Upper"] < df["Close"]

    condition_2_buy = df["RSI"] < 30
    condition_2_sell = df[f"RSI"] > 70

    # Apply our conditions
    df.loc[condition_1_buy & condition_2_buy, "signal"] = 1
    df.loc[condition_1_sell & condition_2_sell, "signal"] = -1

    # Compute our exit signal
    df["pct_close_futur"] = (df["Close"].shift(-2)-df["Close"])/df["Close"]

    # Compute the returns of each position
    df["returns"] = df["signal"]*df["pct_close_futur"]

    df

    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot price with moving averages and Bollinger Bands
    ax1.plot(df['Close'], label='Closing Price', color='black')
    ax1.plot(df['BB_Upper'], label='BB Upper', linestyle='dotted', color='red')
    ax1.plot(df['BB_Lower'], label='BB Lower', linestyle='dotted', color='green')
    # Plot Buy/Sell Signals
    ax1.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1], marker='v', color='red', label='Sell Signal', s=50)
    ax1.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1], marker='^', color='green', label='Buy Signal', s=50)

    ax1.legend()
    ax1.set_title("Bollinger Bands on Closing Prices - AAPL")
    ax1.set_ylabel("Price")
    ax1.grid()

    # Plot RSI
    ax2.plot(df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, linestyle='dashed', color='red', alpha=0.5)
    ax2.axhline(30, linestyle='dashed', color='green', alpha=0.5)
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid()

    # Plot Returns
    ax3.plot(df['returns'].cumsum(), label='Returns', color='blue')
    ax3.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    ax3.set_title("Cumulative Returns")
    ax3.set_ylabel("Cumulative Returns")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.show()


    return 0


main()
