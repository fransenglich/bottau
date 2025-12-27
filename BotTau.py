import sys

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantreo.features_engineering as fe
import quantreo.target_engineering as te
import statsmodels.api as sm
import ta

import backtest
import common
import download
import heatmap

# Constants
# ----------------------------------------------
TICKERS = ['AAPL', 'TSLA']

# Data Frames of tickers, comes from Yahoo Finance and are in classic OHLC(Adj)V
# format.
df_tickers = []


def cagr(first: float, last: float, periods: int) -> float:
    """
    Computes and returns CAGR value.

    Nicked from https://feliperego.github.io/blog/2016/08/10/CAGR-Function-In-Python
    """
    return (last / first) ** (1 / periods) - 1


def strategy_Bollinger_RSI(df: pd.DataFrame,
                           param_window: int = 20) -> pd.DataFrame:
    """A strategy built on conditions involving Bollinger bands and RSI."""

    # Bollinger Bands
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'],
                                                   window=int(param_window))
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'],
                                                   window=int(param_window))
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'],
                                                   window=int(param_window))

    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Create our signal
    df["signal"] = 0

    # Define our conditions
    condition_1_buy = df["Close"] < df["BB_Lower"]
    condition_1_sell = df["BB_Upper"] < df["Close"]

    condition_2_buy = df["RSI"] < 30
    condition_2_sell = df["RSI"] > 70

    # Apply our conditions
    df.loc[condition_1_buy & condition_2_buy, "signal"] = 1
    df.loc[condition_1_sell & condition_2_sell, "signal"] = -1

    # Compute our exit signal
    df["pct_close_futur"] = (df["Close"].shift(-2) - df["Close"])/df["Close"]

    # Compute the returns of each position
    # This is our computed trading signal applied to the returns
    df["returns"] = df["signal"] * df["pct_close_futur"]

    # Plot price with moving averages and Bollinger Bands
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['Close'], label='Closing Price', color='black')
    plt.plot(df['BB_Upper'], label='BB Upper', linestyle='dotted', color='red')
    plt.plot(df['BB_Lower'], label='BB Lower', linestyle='dotted',
             color='green')

    # Plot Buy/Sell Signals
    plt.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1],
                marker='v', color='red', label='Sell Signal', s=50)
    plt.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1],
                marker='^', color='green', label='Buy Signal', s=50)

    plt.legend()
    plt.title("Bollinger Bands on Closing Prices - AAPL")
    plt.ylabel("Price")
    plt.grid()
    common.savefig(plt, "feature_BollingerBands")

    # Plot RSI
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='dashed', color='red', alpha=0.5)
    plt.axhline(30, linestyle='dashed', color='green', alpha=0.5)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    common.savefig(plt, "feature_RSI")

    return df


def plot_and_write(df: pd.DataFrame, featurenames: tuple[str]) -> pd.DataFrame:
    """Writes data and graphs that are strategy-agnostic."""

    # This is our design matrix.
    designmatrix: pd.DataFrame = pd.DataFrame()

    for name in featurenames:
        designmatrix[name] = df[name]

    # We do 3 things: add standard features, produce correlation matrix and VIF.
    # ---- corr matrix ----

    designmatrix['pct_close_futur'] = df['pct_close_futur']
    df['var'] = df['pct_close_futur'].rolling(window=common.WINDOW_SIZE).var()
    df['parkinsons_var'] = fe.volatility.parkinson_volatility(df,
                                                              window_size=common.WINDOW_SIZE,
                                                              high_col="High",
                                                              low_col="Low")
    designmatrix['var'] = df['var']
    designmatrix['parkinsons_var'] = df['parkinsons_var']

    flen = len(designmatrix.columns)
    in_range = range(flen)
    pearsonmatrix = np.zeros((flen, flen), dtype=float)
    spearmanmatrix = np.zeros((flen, flen), dtype=float)

    for i in in_range:
        # This works: for l in range(ilen - (ilen - i) + 1):
        for length in in_range:
            pearsonmatrix[i, length] = designmatrix.iloc[:, i].corr(designmatrix.iloc[:, length]) # TODO why to column length?
            spearmanmatrix[i, length] = designmatrix.iloc[:, i].corr(designmatrix.iloc[:, length], method='spearman')

    cm_labels = designmatrix.columns

    # - Pearson
    fig, ax = plt.subplots()
    pm, _ = heatmap.heatmap(pearsonmatrix, cm_labels, cm_labels, ax=ax,
                            cmap="YlGn", cbarlabel="Pearson correlation coefficient")
    heatmap.annotate_heatmap(pm)

    ax.set_title("Heatmap of Pearson correlation matrix of features")
    fig.tight_layout()
    common.savefig(fig, "pearsonmatrix")

    # - Spearman
    fig, ax = plt.subplots()
    pm, _ = heatmap.heatmap(spearmanmatrix, cm_labels, cm_labels, ax=ax,
                            cmap="YlGn", cbarlabel="Spearman correlation coefficient")
    heatmap.annotate_heatmap(pm)

    ax.set_title("Heatmap of Spearman correlation matrix of features")
    fig.tight_layout()
    common.savefig(fig, "spearmanmatrix")

    # - Multicollinearity

    # variance_inflation_factor() needs this.
    designmatrix.dropna(inplace=True)

    vifs = [(designmatrix.columns.values[i], variance_inflation_factor(designmatrix, i)) for i in range(len(designmatrix.columns))]

    with open("generated/VIFs.tex", "w") as f:
        for name, vif in vifs:
            f.write(name.replace("_", "\\_") + " & " + str(round(vif, 2)) + " \\\\\n")

    return df


def test_opt_Bollinger_RSI(df: pd.DataFrame) -> None:

    def backtest(x) -> float:
        # We evaluate based on the Sharpe Ratio
        df_ret = strategy_Bollinger_RSI(df, x)

        # We want to maximize, so * -1
        return -1.0 * common.sharpe_ratio(df_ret['Close'])

    x0 = (30)  # window size
    bounds = ((2, 100),)

    res = minimize(backtest, x0, bounds=bounds, options={'disp': True})

    plot_vals = []

    for i in range(bounds[0][0], bounds[0][1]):
        v = backtest(i)
        plot_vals.append(v)

    print(res.message)
    print(res.x)

    plt.plot(plot_vals)
    # plt.show()


def investigate(df: pd.DataFrame) -> None:
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['pct_close_futur'], 'g')
    plt.xticks(rotation=70)
    # plt.plot(df['pct_close_futur'], label='pct_close_futur')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("pct_close_futur")
    plt.ylabel("pct_close_futur")
    plt.legend()
    plt.grid()
    plt.show()


def strategy_sma(df: pd.DataFrame) -> pd.DataFrame:
    """A moving average crossover strategy."""

    df['SMA5'] = df["Close"].rolling(window = 5).mean()
    df['SMA30'] = df["Close"].rolling(window = 30).mean()

    df['signal'] = 0
    condition: pd.Series = df['SMA5'] > df['SMA30']

    df.loc[condition, "signal"] = 1

    df['returns'] = df['signal'] * df['pct_close_futur']

    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['Close'], label='Closing Price', linestyle='dotted', color='black')
    plt.plot(df['SMA5'], label='SMA 5', linestyle='dotted', color='red')
    plt.plot(df['SMA30'], label='SMA 30', linestyle='dotted', color='green')
    plt.xticks(rotation=70)
    plt.legend()
    plt.grid()

    plt.show()

    return df


def main() -> int:

    plt.ioff()

    if len(sys.argv) == 2:
        if sys.argv[1] == "i":
            df_tickers.append(download.initialDownload())
        elif sys.argv[1] == "c":
            df_tickers.append(download.fetchNewTicks())
        else:
            raise Exception("No or wrong commandline argument passed.")
    else:
        df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)

        # Reverse, get increasing dates. Specific to IBM.csv.
        df = df[::-1]

        df_tickers.append(df)

    df: pd.DataFrame = df_tickers[0]

    # For some reason the name differs.
    df = df.rename(columns={"1. open":    "Open",
                            "2. high":    "High",
                            "3. low":     "Low",
                            "4. close":   "Close",
                            "5. volume":  "Volume"})

    # df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')

    df

    df = df.head(100)

    # ---------- Drop NaNs ---------
    len_before = len(df)
    df = df.dropna()
    print(f"Dropped from `df': {len_before - len(df)} rows, out of total {len_before}.")

    df["pct_close_futur"] = (df["Close"].shift(-2) - df["Close"]) / df["Close"]

    df = strategy_Bollinger_RSI(df)
    #test_opt_Bollinger_RSI(df)

    # df = strategy_sma(df)
    # investigate(df)

    # -------------- Target ---------------
    df['target_future_returns_sign'] = te.directional.future_returns_sign(df,
                                                                          close_col="Close")

    # -------------- Regression ---------------
    # scikit-learn
    sklearn_reg = LinearRegression()
    sklearn_reg.fit(pd.DataFrame(df['target_future_returns_sign']),
                    pd.DataFrame(df['signal']))
    # Model is now trained.
    print(f"coef: {sklearn_reg.coef_}")
    print(f"intercept: {sklearn_reg.intercept_}")
    # reg.predict()

    # statsmodels
    # We copy the DataFrame because add_constant() adds a column.
    X = pd.DataFrame(df['signal'])
    y = df['target_future_returns_sign']

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    with open("generated/ols_conditions.txt", "w") as f:
        # Possibly write est2.pvalues['signal']
        f.write(str(est2.summary()))


    df.to_csv("generated/df.csv")
    plot_and_write(df, ())

    backtest.backtest(df)

    # Skip the other columns, backtest_static_portfolio() expects this.
    # df = pd.DataFrame(df["returns"])

    # weights = (1)
    # backtest.backtest_static_portfolio(weights, df)

    # ---------- Split ---------
    # split_point = int(0.80 * len(df))

    # TODO in/out of sample
    # in_sample = # before
    # out_sample = # after

    # "returns" is created in strategy()
    # y_train = df[["returns"]].iloc[:split_point]
    # X_train = df[["feature1", "feature2"]].iloc[:split_point]

    # broker: BrokerABC.BrokerABC = Broker.init()

    return 0

    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    # ChatGPT: "I'm stuck in my financial quant project. What shall I do?":
#        from sklearn.model_selection import train_test_split
#        from sklearn.ensemble import RandomForestClassifier
#        from sklearn.metrics import classification_report

#        X = df[['MACD', 'RSI']]  # or other features
#        y = df['Direction']
#
#        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
#
#        model = RandomForestClassifier()
#        model.fit(X_train, y_train)
#
#        preds = model.predict(X_test)
#        print(classification_report(y_test, preds))

# Final function for the strategy takes at least the `symbol' as argument, and
# returns a tuple with 2 Bools, for buy and sell. The function is pre-trained,
# loads the model with joblib.load() or so, does a prediction and returns.


if __name__ == "__main__":
    main()
