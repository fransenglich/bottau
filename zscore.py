import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


def main() -> None:
    """Plots rolling z-scores for closing prices.

    Notice that z-scores doesn't assume normally distributed, but if you want
    to look up p-values they have to be. Interpretation should be affected by
    actual distribution.
    """

    WINDOW_SIZE = 30

    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df = df[::-1]

    # ---- z-score
    df["zscore"] = scipy.stats.zscore(df["4. close"])
    plt.figure(figsize=(15, 8))
    plt.plot(df["zscore"])

    # ---- z-score rolling scipy
    df["scipy_zscore_rolling30"] = np.nan
    col = df.columns.get_loc("scipy_zscore_rolling30")

    for i in range(len(df)):
        df.iloc[i, col] = scipy.stats.zscore(df["4. close"]
                                             .iloc[i:(i + WINDOW_SIZE)],
                                             ddof=1)[-1]

    plt.plot(df["scipy_zscore_rolling30"])

    # ---- z-score rolling Pandas
    rolling_mean = df["4. close"].rolling(WINDOW_SIZE).mean()
    rolling_std = df["4. close"].rolling(WINDOW_SIZE).std()
    df["pandas_zscore_rolling30"] = (df["4. close"] - rolling_mean)/rolling_std
    plt.plot(df["pandas_zscore_rolling30"])

    # ---- Plot
    plt.legend(("z-score",
                f"scipy z-score {WINDOW_SIZE} days",
                f"Pandas rolling z-score {WINDOW_SIZE} days"))
    plt.title("z-scores for closing prices")
    plt.grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')
    plt.show()


def rolling_using_apply() -> None:
    """Plots rolling z-scores for closing prices.

    Notice that z-scores doesn't assume normally distributed, but if you want
    to look up p-values they have to be. Interpretation should be affected by
    actual distribution.
    """

    WINDOW_SIZE = 30

    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df = df[::-1]

    plt.figure(figsize=(15, 8))

    # ---- z-score
    df["zscore"] = scipy.stats.zscore(df["4. close"])
    plt.plot(df["zscore"])

    # ---- z-score rolling scipy
    def func_zs(data):
        return scipy.stats.zscore(data, ddof=1)[-1]

    df["scipy_zscore_rolling30"] = df["4. close"] \
        .rolling(window=WINDOW_SIZE) \
        .apply(func_zs)

    plt.plot(df["scipy_zscore_rolling30"])

    # ---- z-score rolling Pandas
    rolling_mean = df["4. close"].rolling(WINDOW_SIZE).mean()
    rolling_std = df["4. close"].rolling(WINDOW_SIZE).std()
    df["pandas_zscore_rolling30"] = (df["4. close"] - rolling_mean)/rolling_std
    # plt.plot(df["pandas_zscore_rolling30"])

    # ---- Plot
    plt.legend(("z-score",
                f"scipy z-score {WINDOW_SIZE} days",
                f"Pandas rolling z-score {WINDOW_SIZE} days"))
    plt.title("z-scores for closing prices")
    plt.grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')
    plt.show()


if __name__ == "__main__":
    main()
