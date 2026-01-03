import yfinance as yf
import pandas as pd


def main2() -> None:
    df = pd.read_csv("Tickers/AAPL orig.csv")
    print(df)


def main() -> None:
    df: pd.DataFrame = yf.download(("AAPL", "NVDA"), group_by="ticker", multi_level_index=True)

    print(df)
    df.to_csv("Tickers/multiindex2.csv")

    #df2 = df["Close"]["AAPL"]
    df2 = df["AAPL"]["Close"]
    print(df2)
    df2.to_csv("Tickers/df2.csv")


if __name__ == "__main__":
    main()
