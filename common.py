import matplotlib
import numpy as np
import pandas as pd

FIG_SIZE = (8, 4)
WINDOW_SIZE = 30
TRANSACTION_COMMISSION = 0.02

# See Lucas' book, p. 295.
# Take-profit
TP = 0.021

# Stop loss
SL = 0.09


def savefig(plt: matplotlib.figure.Figure, basename: str) -> None:
    plt.savefig(f"generated/{basename}.png")
    matplotlib.pyplot.close()


def sharpe_ratio(portfolio: pd.Series,
                 timeframe: int = 252,
                 rf: float=0.01) -> float:
    """Computes the Sharpe Ratio for the returns in the passed Series.
    
    See:
    - https://www.codearmo.com/blog/sharpe-sortino-and-calmar-ratios-python
    """

    mean = portfolio.mean() * timeframe - rf
    std = portfolio.std() * np.sqrt(timeframe)

    return mean / std


def sortino_ratio(series: pd.Series, N: int, rf: float):
    """Computes the Sortino ratio and returns it."""
    mean = series.mean() * N - rf
    std_neg = series[series < 0].std() * np.sqrt(N)

    return mean / std_neg


def max_drawdowns(returns: pd.Series):
    comp_ret = (returns + 1).cumprod()
    peak = comp_ret.expanding(min_periods = 1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()


def calmar_ratio(series: pd.Series, N: int=255):
    return series.mean() * N / abs(max_drawdowns(series))


def cagr(first: float, last: float, periods: int) -> float:
    """
    Computes and returns CAGR value.

    Nicked from https://feliperego.github.io/blog/2016/08/10/CAGR-Function-In-Python
    """
    return (last / first) ** (1 / periods) - 1