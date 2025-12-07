import yfinance as yf
import pandas as pd


def main2() -> None:
    df = pd.read_csv("Tickers/AAPL orig.csv")
    print(df)


def main() -> None:
    df: pd.DataFrame = yf.download(("AAPL", "NVDA"), multi_level_index=True)

    print(df)
    df.to_csv("Tickers/multiindex.csv")

    df2 = df["Close"]["AAPL"]
    print(df2)
    df2.to_csv("Tickers/df.csv")


if __name__ == "__main__":
    main()
