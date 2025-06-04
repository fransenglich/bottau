import yfinance as yf
import matplotlib.pyplot as plt
import sys
import ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

tickers = ['AAPL', 'TSLA']

def_figsize = (4.7, 4)
transaction_commission = 0.02

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
    global def_figsize

    if len(sys.argv) == 2:
        if sys.argv[1] == "i":
            initialDownload()
        elif sys.argv[1] == "c":
            fetchNewTicks()
        else:
            raise Exception("No or wrong commandline argument passed.")
    else:
        df = pd.read_csv("Tickers/IBM.csv")

        # Reverse, get increasing dates. Specific to IBM.csv.
        df = df[::-1]

        df_tickers.append(df)

    df: pd.DataFrame = df_tickers[0]


    # For some reason the name differs.
    df = df.rename(columns = {"1. open":    "Open",
                              "2. high":    "High",
                              "3. low":     "Low",
                              "4. close":   "Close",
                              "5. volume":  "Volume"})

    df["date"] = pd.to_datetime(df["date"]) #, format='%Y-%b-%d')

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

    print(df)

    # Plot price with moving averages and Bollinger Bands
    plt.figure(figsize=def_figsize)
    plt.plot(df['Close'], label='Closing Price', color='black')
    plt.plot(df['BB_Upper'], label='BB Upper', linestyle='dotted', color='red')
    plt.plot(df['BB_Lower'], label='BB Lower', linestyle='dotted', color='green')

    # Plot Buy/Sell Signals
    plt.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1], marker='v', color='red', label='Sell Signal', s=50)
    plt.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1], marker='^', color='green', label='Buy Signal', s=50)

    plt.legend()
    plt.title("Bollinger Bands on Closing Prices - AAPL")
    plt.ylabel("Price")
    plt.grid()
    plt.savefig("generated/feature_BollingerBands.png")

    # Plot RSI
    plt.figure(figsize=def_figsize)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='dashed', color='red', alpha=0.5)
    plt.axhline(30, linestyle='dashed', color='green', alpha=0.5)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    plt.savefig("generated/feature_RSI.png")


    # ---- Drawdown ----
    # 1 + & cumprod() because 'returns' are not log returns.
    df['comp_cumulative_returns'] = (1 + df['returns']).cumprod()
    df['cumulative_max'] = df['comp_cumulative_returns'].cummax()
    df['drawdown'] = ((df['comp_cumulative_returns'] - df['cumulative_max']) / df['cumulative_max']) * 100

    plt.figure(figsize=def_figsize)
    plt.plot(df['drawdown'], label="Drawdown")
    plt.title("Drawdown")
    plt.ylabel("Drawdown %")
    plt.legend()
    plt.savefig("generated/drawdown.png")


    # ---- Drawdown Histogram----
    plt.figure(figsize=def_figsize)
    plt.hist(df['drawdown'], bins='auto')
    plt.title("Drawdown Distribution")
    plt.savefig("generated/drawdown_dist.png")


    # ---- Returns ----
    plt.figure(figsize=def_figsize)
    plt.plot(df['returns'], label='Returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Returns")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    plt.savefig("generated/returns.png")


    # ---- Returns Histogram----
    plt.figure(figsize=def_figsize)
    plt.hist(df['returns'], bins='auto')
    plt.title("Returns Distribution")
    plt.savefig("generated/returns_dist.png")


    # ---- Cumulative Returns ----
    df['cumulative_returns'] = df['returns'].cumsum()
    plt.figure(figsize=def_figsize)
    plt.plot(df['cumulative_returns'], label='Returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Cumulative Returns")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    plt.savefig("generated/cumulative_returns.png")

    # ---- Cumulative Returns Minus Transcation Costs ----
    def transaction_cost(trade):
        global transaction_commission
        return trade - (transaction_commission + trade/2)

    df['cum_with_trans'] = df['cumulative_returns'].map(transaction_cost)

    plt.figure(figsize=def_figsize)
    plt.plot(df['cum_with_trans'], label='Netted returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Cumulative Returns Minus Transaction Costs")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    plt.savefig("generated/cumulative_returns_except_trans_costs.png")


    # ---- Write constants ----
    drawdown_max = round(abs(df['drawdown'].min()), 2) # Percent
    print(f"Max drawdown: {drawdown_max}")

    with open("generated/constants.tex", "w") as f:
        f.write(f"\def\constantMaxdrawdown{{{drawdown_max}}}")
        f.write(f"\n\def\constantStartdate{{{df['date'].min()}}}")
        f.write(f"\n\def\constantEnddate{{{df['date'].max()}}}")
        f.write(f"\n\def\constantTransactionCommission{{{transaction_commission}}}")

        rmean = round(df['returns'].mean() * 100, 4)
        std = round(df['returns'].std(), 4)
        sr = round(rmean/std, 4)
        f.write(f"\n\def\constantRMean{{{rmean}}}")
        f.write(f"\n\def\constantSharpeRatio{{{sr}}}")
        f.write(f"\n\def\constantStd{{{std}}}")

    df.to_csv("generated/df.csv")

    return 0


if __name__ == "__main__":
    main()