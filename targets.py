import matplotlib.pyplot as plt
import pandas as pd
import quantreo.target_engineering as te


def plot_future_returns(df: pd.DataFrame, close_col='returns') -> None:
    df["future_returns"] = te.magnitude.future_returns(df, close_col,
                                                       window_size=10)

    plt.figure(figsize=(15, 6))
    plt.plot(df["future_returns"])
    plt.title("Future Returns Target", size=15)
    plt.show()


def main() -> None:
    """
    A dummy function for testing the plot.
    """

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

    plot_future_returns(df)


if __name__ == "__main__":
    main()
