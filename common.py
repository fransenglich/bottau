import matplotlib
import numpy as np
import pandas as pd

FIG_SIZE = (15, 6)
WINDOW_SIZE = 30
TRANSACTION_COMMISSION = 0.01 # TODO

def savefig(plt: matplotlib.figure.Figure, basename: str) -> None:
    plt.savefig(f"generated/{basename}.png")
    matplotlib.pyplot.close()


def sharpe_ratio(portfolio: pd.Series, timeframe: int = 252) -> float:
    """Computes the Sharpe Ratio for the returns in the passed Series."""

    mean = portfolio.mean() * timeframe
    std = portfolio.std() * np.sqrt(timeframe)

    return mean / std
