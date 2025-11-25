import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


def main() -> None:
    """Plots rolling Z-scores for closing prices.

    Notice that z-scores doesn't assume normally distributed, but if you want
    to look up p-values they have to be. Interpretation should be affected by
    actual distribution.
    """

    WINDOW = 30

    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df = df[::-1]

    # ---- Z-score
    df["zscore"] = scipy.stats.zscore(df["4. close"])
    plt.figure(figsize=(15, 8))
    plt.plot(df["zscore"])

    # ---- Z-score scipy rolling
    df["scipy_zscore_rolling30"] = np.nan
    col = df.columns.get_loc("scipy_zscore_rolling30")

    for i in range(len(df)):
        df.iloc[i, col] = scipy.stats.zscore(df["4. close"]
                                             .iloc[i:(i + WINDOW)],
                                             ddof=1)[-1]

    plt.plot(df["scipy_zscore_rolling30"])

    # ---- Z-score manual
    rolling_mean = df["4. close"].rolling(WINDOW).mean()
    rolling_std = df["4. close"].rolling(WINDOW).std()
    df["manual_zscore_rolling30"] = (df["4. close"] - rolling_mean)/rolling_std
    plt.plot(df["manual_zscore_rolling30"])

    # ---- Plot
    plt.legend(("Z-score",
                f"scipy Z-score {WINDOW} days",
                f"Manual rolling Z-score {WINDOW} days"))
    plt.title("Z-scores for closing prices")
    plt.grid(axis='y', linestyle='-', alpha=0.5, color='lightgrey')
    plt.show()


if __name__ == "__main__":
    main()
