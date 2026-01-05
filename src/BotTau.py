import sys

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import quantreo.target_engineering as te
import statsmodels.api as sm

from lib import backtest
from lib import common
from lib import ml_modelling
from strategies.bollinger_rsi import strategy_Bollinger_RSI


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

    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)

    # Reverse, get increasing dates. Specific to IBM.csv.
    df = df[::-1]

    # For some reason the name differs.
    df = df.rename(columns={"date":     "time",
                            "1. open":  "open",
                            "2. high":  "high",
                            "3. low":   "low",
                            "4. close": "close"})
    df.drop(["5. volume"], axis=1)

    # df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')

    df

    df = df.head(100)

    # ---------- Drop NaNs ---------
    len_before = len(df)
    df = df.dropna()
    print(f"Dropped from `df': {len_before - len(df)} rows, out of total {len_before}.")

    df["pct_close_futur"] = (df["close"].shift(-2) - df["close"]) / df["close"]

    df = strategy_Bollinger_RSI(df)
    #test_opt_Bollinger_RSI(df)

    # df = strategy_sma(df)
    # investigate(df)

    # -------------- Target ---------------
    df['target_future_returns_sign'] = te.directional.future_returns_sign(df)

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

    with open(common.generated_file("ols_conditions.txt", "ibm"), "w") as f:
        # Possibly write est2.pvalues['signal']
        f.write(str(est2.summary()))

    # df.to_csv("generated/df.csv")
    ml_modelling.investigate(df, (), "ibm")

    backtest.backtest(df, "ibm")

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
