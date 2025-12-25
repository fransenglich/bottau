import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

WINDOW_SIZE = 30


def plot_entropy(df: pd.DataFrame) -> None:
    """
    Computes and plots the Shannon entropy for discrete returns, volatility and
    skewness in the returns passed in the argument.

    The function shows a plot using Matplotlib.

    Entropy essentially shows how much information is in the data, as opposed
    to randomness.

    A time series with low entropy has potential for having
    patterns/information/predicatability. High entropy implies
    noise/unpredictability.

    See:
    - https://en.wikipedia.org/wiki/Entropy_(information_theory)
    - https://stackoverflow.com/questions/15450192/\
            fastest-way-to-compute-entropy-in-python
    - https://www.linkedin.com/posts/lucas-inglese-\
            75574817b_what-entropy-really-tells-you-about-a-\
            time-activity-7401568412436561922-wO7K/
    """

    # See the code in main(). It does the function, but it hasn't been
    # abstracted/factorised.
    pass


def main() -> None:
    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)  # .to_period('D')
    df = df[::-1]

    # Standardize feature names
    df.rename(columns={"4. close": "Adj Close"}, inplace=True)

    df = df[0:300]

    df.dropna(inplace=True)

    df["returns"] = df["Adj Close"].pct_change()

    def rolling_entropy(window_data):
        # 1. Create a histogram to get frequencies.
        counts, _ = np.histogram(window_data, WINDOW_SIZE)

        # 2. Calculate Shannon entropy on the counts.
        return scipy.stats.entropy(counts)

    df["entropy_returns"] = df["returns"].rolling(window=WINDOW_SIZE) \
        .apply(rolling_entropy)

    df["vol_returns"] = df["returns"].rolling(window=WINDOW_SIZE).std()

    df["entropy_vol_returns"] = df["vol_returns"].rolling(window=WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Plot
    plt.figure()
    plt.plot(df["entropy_returns"], label="Entropy Returns")
    plt.plot(df["entropy_vol_returns"], label="Volatility")
    plt.legend(["Entropy Returns", "Entropy for Volatility"])

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
