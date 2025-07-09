
import yfinance as yf
import pandas as pd

def main() -> None:
    df: pd.DataFrame = yf.download("^GSPC", multi_level_index=False)

    df.to_csv("Tickers/GSPC.csv")

if __name__ == "__main__":
    main()