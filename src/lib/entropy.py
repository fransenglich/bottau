import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

from lib import common


def main() -> None:
    """
    Computes and plots the Shannon entropy for discrete returns, volatility and
    skewness in the returns passed in the argument.

    The function shows a plot using Matplotlib.

    Entropy essentially shows how much information is in the data, as opposed
    to randomness.

    A time series with low entropy has potential for having
    patterns/information/predictability. High entropy implies
    noise/unpredictability.

    In the context of trading, the entropy is a quantification to how
    "learnable" a data set is. Useful indicator to whether there's something to
    predict at all. A model can't solve a too high entropy.

    See:
    - https://en.wikipedia.org/wiki/Entropy_(information_theory)
    - https://stackoverflow.com/questions/15450192/\
            fastest-way-to-compute-entropy-in-python
    - https://www.linkedin.com/posts/lucas-inglese-\
            75574817b_what-entropy-really-tells-you-about-a-\
            time-activity-7401568412436561922-wO7K/
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  "../../Tickers/IBM.csv"),
                     index_col="date",
                     parse_dates=True)

    # This CSV is in reverse order.
    df = df[::-1]

    # Standardize feature names
    df.rename(columns={"4. close": "Adj Close"}, inplace=True)

    df.dropna(inplace=True)

    df["returns"] = df["Adj Close"].pct_change()  # NOTE inconsistent naming

    def rolling_entropy(window_data):
        # 1. Create a histogram to get frequencies.
        counts, _ = np.histogram(window_data, common.WINDOW_SIZE)

        # 2. Calculate Shannon entropy on the counts.
        return scipy.stats.entropy(counts)

    # Returns
    df["entropy_returns"] = df["returns"].rolling(window=common.WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Volatility
    df["vol_returns"] = df["returns"].rolling(window=common.WINDOW_SIZE).std()

    df["entropy_vol_returns"] = df["vol_returns"].rolling(window=common.WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Skewness
    df["skewness"] = df["returns"].rolling(window=common.WINDOW_SIZE).skew()
    df["entropy_skewness"] = df["skewness"].rolling(window=common.WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Plot
    plt.figure()
    plt.plot(df["entropy_returns"], label="Entropy Returns")
    plt.plot(df["entropy_vol_returns"], label="Volatility")
    plt.plot(df["entropy_skewness"], label="Skewness")

    plt.legend(["Entropy of Returns",
                "Entropy of Volatility",
                "Entropy of Skewness"])
    plt.title("Entropy of Moments of IBM Returns")

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
