import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def initialDownload(tickers) -> None:
    """ Fetches for the tickers in `tickers` and writes them out to CSV-files in Tickers/.

    We build simple DataFrames, one header like what is typical.
    Discussed here:
    https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801
    """

    dfs = []

    for ticker in tickers:
        #df = yf.download(ticker, period='1y', interval='1d', auto_adjust=False)
        df = yf.download(ticker, period='1y', group_by='Ticker', interval='1d', auto_adjust=False)
        df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        #df['ticker'] = ticker
        df.to_csv(f'Tickers/{ticker}.csv')
        dfs.append(df)

    return dfs

def fetchNewTicks(tickers) -> None:
    """Updates the tickers to today."""

    dfs = []

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

        dfs.append(merged)

    return dfs

# TODO:
# - Fix fetchNewTicks()

def downloadToFile() -> None:
    df = yf.download("AAPL", interval='1d').loc["2015":]
    df.to_csv("Tickers/AAPL.manual.csv")
