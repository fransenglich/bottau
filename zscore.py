import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


def main() -> None:
    """Plots the Z-score for closing prices."""

    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df = df[::-1]

    df["zscore"] = scipy.stats.zscore(df["4. close"])

    df["zscore_rolling30"] = np.nan
    col = df.columns.get_loc("zscore_rolling30")

    for i in range(len(df)):
        df.iloc[i, col] = scipy.stats.zscore(df["4. close"].iloc[i:(i + 30)])[-1]

    plt.figure(figsize=(15, 8))
    plt.plot(df["zscore"])
    plt.plot(df["zscore_rolling30"])
    plt.legend(["Z-score", "Z-score 30 days"])

    plt.title("Z-scores for closing prices")
    plt.grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')
    plt.show()


if __name__ == "__main__":
    main()
