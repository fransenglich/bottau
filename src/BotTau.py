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

from lib import backtest
from lib import common
from lib import download
from lib import heatmap
from strategies.bollinger_rsi import strategy_Bollinger_RSI

# Data Frames of tickers, comes from Yahoo Finance and are in classic OHLC(Adj)V
# format.
df_tickers = []


def ml_investigate(df: pd.DataFrame, featurenames: tuple[str]) -> pd.DataFrame:
    """Writes data and graphs that are strategy-agnostic.
    
    This is specific to features used in regressions."""

    # This is our design matrix.
    designmatrix: pd.DataFrame = pd.DataFrame()

    for name in featurenames:
        designmatrix[name] = df[name]

    # We do 3 things: add standard features, produce correlation matrix and VIF.
    # ---- corr matrix ----

    designmatrix['pct_close_futur'] = df['pct_close_futur']

    df['var'] = df['pct_close_futur'].rolling(window=common.WINDOW_SIZE).var()
    designmatrix['var'] = df['var']

    df['vol_parkinson_30'] = fe.volatility.parkinson_volatility(df,
                                                                high_col="High",
                                                                low_col="Low",
                                                                window_size=30)
    designmatrix['vol_parkinson_30'] = df['vol_parkinson_30']

    df['vol_parkinson_60'] = fe.volatility.parkinson_volatility(df,
                                                                high_col="High",
                                                                low_col="Low",
                                                                window_size=60)
    designmatrix['vol_parkinson_60'] = df['vol_parkinson_60']

    df['vol_ctc_30'] = fe.volatility.close_to_close_volatility(df,
                                                               high_col="High",
                                                               low_col="Low",
                                                               window_size=30)
    designmatrix['vol_ctc_30'] = df['vol_ctc_30']

    df['vol_ctc_60'] = fe.volatility.close_to_close_volatility(df,
                                                               high_col="High",
                                                               low_col="Low",
                                                               window_size=60)
    designmatrix['vol_ctc_60'] = df['vol_ctc_60']

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

    vifs = [(designmatrix.columns.values[i],
             variance_inflation_factor(designmatrix, i))
             for i in range(len(designmatrix.columns))]

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
    ml_investigate(df, ())

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
