import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

WINDOW_SIZE = 30


def plot_entropy(df: pd.DataFrame) -> None:
    """
    Plots the Shannon entropy for returns, volatlity and skewness in the
    returns passed in the argument.

    The function shows a plot using Matplotlib.

    Entropy essentially shows how much information is in the data, as opposed
    to randomness.

    A time series with low entropy has potential for having patterns,
    information. High entropy, implies noises.

    See:
    - https://en.wikipedia.org/wiki/Entropy_(information_theory)
    - https://stackoverflow.com/questions/15450192/\
            fastest-way-to-compute-entropy-in-python
    - https://www.linkedin.com/posts/lucas-inglese-\
            75574817b_what-entropy-really-tells-you-about-a-\
            time-activity-7401568412436561922-wO7K/
    """

    # Get entropy from count. We pass in count, which is ok because entropy()
    # accepts unnormalized values
    df["entropy_returns"] = df["returns"].rolling(window=10) \
                                         .apply(scipy.stats.entropy)

    print(df["entropy_returns"])

    # vol = returns.stdev()
    # p_vol = vol.value_counts()
    # e_vol = scipy.stats.entropy(p_vol)

    # Plot
    plt.figure()
    plt.plot(df["entropy_returns"], label="Entropy Returns", color="green")
    # plt.plot(e_vol, label="Volatility", color="blue")
    plt.legend(["Entropy Returns", "Entropy for Volatility"])
    plt.grid()
    plt.show()

    # TODO axis descriptions


def main() -> None:
    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)  # .to_period('D')
    df = df[::-1]

    # Standardize feature names
    df.rename(columns={"4. close": "Adj Close"}, inplace=True)

    df = df[0:300]

    df.dropna(inplace=True)

    df["returns"] = df["Adj Close"].pct_change()

    # plt.figure()
    # plt.plot(df["returns"], label="Returns")
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(df["returns"], label="Foo", color="Black")
    # plt.show()

    # plot_entropy(df)
    # df["entropy_returns"] = df["returns"].rolling(window=10) \
    #                                     .apply(scipy.stats.entropy)

    def rolling_entropy(window_data, bins=WINDOW_SIZE):
        # 1. Create a histogram to get frequencies
        counts, _ = np.histogram(window_data, bins=bins)

        # 2. Calculate Shannon entropy on the counts
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
