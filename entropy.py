import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


def plot_entropy(df: pd.DataFrame) -> None:
    """
    Plots the entropy for returns, volatlity and skewness in the returns passed
    in the argument.

    The function shows a plot using Matplotlib.

    See:
    - https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    # TODO docs on interpretation

    # 1. We need the probability for each value in `returns'

    # Counts occurrence of each value
    p_returns = df["returns"].value_counts()

    print(len(p_returns))

    # Get entropy from count. We pass in count, which is ok because entropy()
    # accepts unnormalized values
    df["entropy_returns"] = scipy.stats.entropy(p_returns)

    print(df["entropy_returns"])

    # vol = returns.stdev()
    # p_vol = vol.value_counts()
    # e_vol = scipy.stats.entropy(p_vol)

    # Plot
    plt.figure()
    plt.plot(df["entropy_returns"], label="Returns", color="green")
    # plt.plot(e_vol, label="Volatility", color="blue")
    plt.legend(["Returns", "Volatility"])
    plt.grid()
    plt.show()

    # TODO axis descriptions


def main() -> None:
    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)  # .to_period('D')
    df = df[::-1]

    # Standardize feature names
    df.rename(columns={"4. close": "Adj Close"}, inplace=True)

    df = df[0:100]

    df["returns"] = df["Adj Close"].pct_change()

    # plt.figure()
    # plt.plot(df["returns"], label="Foo", color="Black")
    # plt.show()

    plot_entropy(df)


if __name__ == "__main__":
    main()
