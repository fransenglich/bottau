import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


def main() -> None:
    """Plots the Z-score for closing prices."""

    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df = df[::-1]

    df["zscore"] = scipy.stats.zscore(df["4. close"])

    plt.figure(figsize=(15, 8))
    plt.plot(df["zscore"])
    plt.legend(["Z-scores"])
    plt.df = df[::-1]
    plt.title("Z-scores for closing prices")
    plt.grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')
    plt.show()


if __name__ == "__main__":
    main()
