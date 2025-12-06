import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


def plot_entropy(returns: pd.Series) -> None:
    """
    Plots the entropy for returns, volatlity and skewness in the returns passed
    in the argument.

    See:
    - https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """

    # 1. We need the probability for each value in `returns'

    # Counts occurrence of each value
    p_returns = returns.value_counts()

    # Get entropy from count. We pass in count, which is ok because entropy()
    # accepts unnormalized values
    e_returns = scipy.stats.entropy(p_returns)

    vol = returns.stdev()
    p_vol = vol.value_counts()
    e_vol = scipy.stats.entropy(p_vol)

    # Plot
    plt.figure()
    plt.plot(e_returns, label="Returns", color="green")
    plt.plot(e_vol, label="Volatility", color="blue")
    plt.legend(["Returns", "Volatility"])
    plt.grid()
    plt.show()


def main() -> None:
    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).to_period('D')
    df = df[::-1]

    # Standardize feature names
    df.rename(columns={"4. close": "Adj Close"}, inplace=True)

    df = df[0:100]

    plot_entropy(df["Adj Close"])


if __name__ == "__main__":
    main()
