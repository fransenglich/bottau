import matplotlib.pyplot as plt
import matplotlib
import heatmap
import sys
import download
import ta
import quantreo.features_engineering as fe
import pandas as pd
import numpy as np

import BrokerABC
import Broker

# Constants
# ----------------------------------------------
TICKERS = ['AAPL', 'TSLA']
DEFAULT_FIGSIZE = (20, 4)
TRANSACTION_COMMISSION = 0.02
ROLLING_WINDOW_SIZE = 30

# See Lucas' book, p. 295.
# Take-profit
TP = 0.021
# Stop loss
SL = 0.09

# Data Frames of tickers, comes from Yahoo Finance and are in classic OHLC(Adj)V
# format.
df_tickers = []

#  Nicked from https://feliperego.github.io/blog/2016/08/10/CAGR-Function-In-Python
def CAGR(first: float, last: float, periods: int) -> float:
    return (last/first)**(1/periods)-1


def sharpe_function(portfolio: pd.DataFrame, timeframe: int = 252) -> float:
    mean = portfolio.mean() * timeframe
    std = portfolio.std() * np.sqrt(timeframe)

    return mean/std


def backtest_static_portfolio(weights, database, ben="^GSPC", timeframe: int = 252, CR: bool =False):
    """
    -----------------------------------------------------------------------------
    | Output: Backtest static portfolio                                         |
    -----------------------------------------------------------------------------
    | Inputs: - weights (type 1d array numpy): weights of the portfolio         |
    |         - database (type dataframe pandas): Returns of the asset          |
    |         - ben (type string): Name of the benchmark                        |
    |         - timeframe (type int): annualization factor                      |
    -----------------------------------------------------------------------------
    """
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    #plt.style.use('seaborn')


    # Compute the portfolio
    portfolio = np.multiply(database, np.transpose(weights))
    portfolio = portfolio.sum(axis=1)
    columns = database.columns
    columns = [col for col in columns]

    ######################### COMPUTE THE BETA ##################################
    # Import the benchmark
    # Lucas: benchmark = yf.download(ben)["Adj Close"].pct_change(1).dropna()
    benchmark = pd.read_csv("Tickers/GSPC.csv")["Close"].pct_change(1).dropna()

    # Concat the asset and the benchmark
    join = pd.concat((portfolio, benchmark), axis=1).dropna()

    # Covariance between the asset and the benchmark
    cov = np.cov(join, rowvar=False)[0][1]

    # Compute the variance of the benchmark
    var = np.cov(join, rowvar=False)[1][1]

    beta = cov/var


    ######################### COMPUTE THE ALPHA #################################
    # Mean of returns for the asset
    mean_stock_return = join.iloc[:,0].mean()*timeframe

    # Mean of returns for the market
    mean_market_return = join.iloc[:,1].mean()*timeframe

    # Alpha
    alpha = mean_stock_return - beta*mean_market_return


    ######################### COMPUTE THE SHARPE ################################
    mean = portfolio.mean() * timeframe
    std = portfolio.std() * np.sqrt(timeframe)
    Sharpe = mean/std


    ######################### COMPUTE THE SORTINO ###############################
    downward = portfolio[portfolio<0]
    std_downward = downward.std() * np.sqrt(timeframe)
    Sortino = mean/std_downward


    ######################### COMPUTE THE DRAWDOWN ###############################
    # Compute the cumulative product returns
    cum_rets = (portfolio+1).cumprod()

    # Compute the running max
    running_max = np.maximum.accumulate(cum_rets.dropna())
    running_max[running_max < 1] = 1

    # Compute the drawdown
    drawdown = ((cum_rets)/running_max - 1)
    min_drawdown = -drawdown.min()


    ######################### COMPUTE THE VaR ##################################
    theta = 0.01
    # Number of simulations
    n = 100000

    # Find the values for theta% error threshold
    t = int(n*theta)

    # Create a vector with n simulations of the normal law
    vec = pd.DataFrame(np.random.normal(mean, std, size=(n,)),
                      columns = ["Simulations"])

    # Orderer the values and find the theta% value
    VaR = -vec.sort_values(by="Simulations").iloc[t].values[0]


    ######################### COMPUTE THE cVaR #################################
    cVaR = -vec.sort_values(by="Simulations").iloc[0:t,:].mean().values[0]

    ######################### COMPUTE THE RC ###################################
    if CR:
    # Find the number of the asset in the portfolio
        l = len(weights)

        # Compute the risk contribution of each asset
        crs = []
        for i in range(l):
            # Importation of benchmark
            # Lucas: benchmark = yf.download(ben)["Adj Close"].pct_change(1).dropna()
            benchmark = pd.read_csv("Tickers/GSPC.csv")["Close"].pct_change(1).dropna()

            # Concat the asset and the benchmark
            join_bis = pd.concat((database.iloc[:,i], benchmark), axis=1).dropna()

            # Covariance between the asset and the benchmark
            cov = np.cov(join_bis, rowvar=False)[0][1]

            # Compute the variance of the benchmark
            var = np.cov(join_bis, rowvar=False)[1][1]

            beta_s = cov/var
            cr = beta_s * weights[i]
            crs.append(cr)

        crs = crs/np.sum(crs) # Normalizing by the sum of the risk contribution

    ######################### PLOT THE RESULTS #################################
    print(f"""
    -----------------------------------------------------------------------------
    Portfolio: {columns}
    -----------------------------------------------------------------------------
    Beta: {np.round(beta, 3)} \t Alpha: {np.round(alpha*100, 2)} %\t \
    Sharpe: {np.round(Sharpe, 3)} \t Sortino: {np.round(Sortino, 3)}
    -----------------------------------------------------------------------------
    VaR: {np.round(VaR*100, 2)} %\t cVaR: {np.round(cVaR*100, 2)} % \t \
    VaR/cVaR: {np.round(cVaR/VaR, 3)} \t drawdown: {np.round(min_drawdown*100, 2)} %
    -----------------------------------------------------------------------------
    """)


    plt.figure(figsize=(15,8))
    plt.plot(join.iloc[:,0].cumsum()*100, color="#035593", linewidth=3)
    plt.plot(join.iloc[:,1].cumsum()*100, color="#068C72", linewidth=3)
    plt.title("CUMULTATIVE RETURN", size=15)
    plt.ylabel("Cumulative return %", size=15)
    plt.xticks(size=15,fontweight="bold")
    plt.yticks(size=15,fontweight="bold")
    plt.legend(["Strategy", "Benchmark"])
    plt.show()

    plt.figure(figsize=(15,8))
    plt.fill_between(drawdown.index, drawdown*100, 0, color="#CE5151")
    plt.plot(drawdown.index,drawdown*100, color="#930303", linewidth=1.5)
    plt.title("DRAWDOWN", size=15)
    plt.ylabel("Drawdown %", size=15)
    plt.xticks(size=15,fontweight="bold")
    plt.yticks(size=15,fontweight="bold")
    plt.show()



    if CR:
        plt.figure(figsize=(15,8))
        plt.scatter(columns, crs, linewidth=3, color = "#B96553")
        plt.axhline(0, color="#53A7B9")
        plt.grid(axis="x")
        plt.title("RISK CONTRIBUTION PORTFOLIO", size=15)
        plt.xlabel("Assets")
        plt.ylabel("Risk contribution")
        plt.xticks(size=15,fontweight="bold")
        plt.yticks(size=15,fontweight="bold")
        plt.show()
        plt.show()


def savefig(plt: matplotlib.figure.Figure, basename: str) -> None:
    plt.savefig(f"generated/{basename}.png")


def backtest(df: pd.DataFrame) -> None:
    """A function by Lucas Inglese modified by me that plots and prints a backtest.
    
    The passed DataFrame must have a column named returns, which is the
    returns of the strategy to be backtested."""
    # ---- Drawdown ----
    # 1 + & cumprod() because 'returns' are not log returns.
    df['comp_cumulative_returns'] = (1 + df['returns']).cumprod()
    df['cumulative_max'] = df['comp_cumulative_returns'].cummax()
    df['drawdown'] = ((df['comp_cumulative_returns'] - df['cumulative_max']) / df['cumulative_max']) * 100

    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['drawdown'], label="Drawdown")
    plt.title("Drawdown")
    plt.ylabel("Drawdown %")
    plt.legend()
    savefig(plt, "drawdown")


    # ---- Drawdown Histogram----
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.hist(df['drawdown'], bins='auto')
    plt.title("Drawdown Distribution")
    savefig(plt, "drawdown_dist")


    # ---- Returns ----
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['returns'], label='Returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Returns")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    savefig(plt, "returns")


    # ---- Returns Histogram----
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.hist(df['returns'], bins='auto')
    plt.title("Returns Distribution")
    savefig(plt, "returns_dist")


    # ---- Cumulative Returns ----
    df['cumulative_returns'] = df['returns'].cumsum()
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['cumulative_returns'], label='Returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Cumulative Returns")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    savefig(plt, "cumulative_returns")


    # ---- Cumulative Returns Minus Transaction Costs ----
    def transaction_cost(trade: float) -> float:
        return trade - (TRANSACTION_COMMISSION + trade/2)

    df['cum_with_trans'] = df['cumulative_returns'].map(transaction_cost)

    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['cum_with_trans'], label='Netted returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Cumulative Returns Minus Transaction Costs")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    savefig(plt, "cumulative_returns_except_trans_costs")


    # ---- Drawdown ----
    drawdown_max = np.round(abs(df['drawdown'].min()), 2) # Percent
    print(f"Max drawdown: {drawdown_max}")


    # ---- Calmar Ratio ----
    def calmar_ratio(returns: pd.Series, maxdrawdown: float) -> float:
        """Returns the Calmar Ratio. Does not take into account the risk-free return."""
        cagr: float = (returns * 100).sum() / len(returns)
        return cagr / maxdrawdown

    cr = calmar_ratio(df["returns"], drawdown_max)
    cr = np.round(cr, 4)


    # ---- Write constants ----
    with open("generated/constants.tex", "w") as f:
        f.write(f"\def\constantMaxdrawdown{{{drawdown_max}}}")
        f.write(f"\n\def\constantStartdate{{{df['date'].min()}}}")
        f.write(f"\n\def\constantEnddate{{{df['date'].max()}}}")
        f.write(f"\n\def\constantTransactionCommission{{{TRANSACTION_COMMISSION}}}")

        rmean = np.round(df['returns'].mean() * 100, 4)
        std = np.round(df['returns'].std(), 4)
        sr = np.round(sharpe_function(df['returns']), 4)
        f.write(f"\n\def\constantRMean{{{rmean}}}")
        f.write(f"\n\def\constantSharpeRatio{{{sr}}}")
        f.write(f"\n\def\constantStd{{{std}}}")

        f.write(f"\n\def\constantCalmarRatio{{{cr}}}")


def strategy_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """A somewhat random function that does:
    * Bollinger bands
    * RSI
    * Parkinson's volatility
    * Heatmap of correlation matrix of features
    """
    # Bollinger Bands
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'], window=20)
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'], window=20)

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
    df["pct_close_futur"] = (df["Close"].shift(-2)-df["Close"])/df["Close"]

    # Compute the returns of each position
    # This is our computed trading signal applied to the returns
    df["returns"] = df["signal"]*df["pct_close_futur"]

    print(df)

    # Plot price with moving averages and Bollinger Bands
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['Close'], label='Closing Price', color='black')
    plt.plot(df['BB_Upper'], label='BB Upper', linestyle='dotted', color='red')
    plt.plot(df['BB_Lower'], label='BB Lower', linestyle='dotted', color='green')

    # Plot Buy/Sell Signals
    plt.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1], marker='v', color='red', label='Sell Signal', s=50)
    plt.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1], marker='^', color='green', label='Buy Signal', s=50)

    plt.legend()
    plt.title("Bollinger Bands on Closing Prices - AAPL")
    plt.ylabel("Price")
    plt.grid()
    savefig(plt, "feature_BollingerBands")

    # Plot RSI
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='dashed', color='red', alpha=0.5)
    plt.axhline(30, linestyle='dashed', color='green', alpha=0.5)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    savefig(plt, "feature_RSI")


    # ---- corr matrix ----
    indicators: pd.DataFrame = []
    indicators.append(df['pct_close_futur'])
    df['var'] = df['pct_close_futur'].rolling(window = ROLLING_WINDOW_SIZE).var()
    df['parkinsons_var'] = fe.volatility.parkinson_volatility(df, window_size=ROLLING_WINDOW_SIZE,
                                                              high_col = "High", low_col="Low")
    indicators.append(df['var'])
    indicators.append(df['parkinsons_var'])

    ilen = len(indicators)
    in_range = range(ilen)
    corrmatrix = np.zeros((ilen, ilen), dtype = float)

    for i in in_range:
        # This works: for l in range(ilen - (ilen - i) + 1):
        for l in in_range:
            corrmatrix[i, l] = indicators[i].corr(indicators[l])
    
    cm_labels = [i.name for i in indicators]

    fig, ax = plt.subplots()
    im, cbar = heatmap.heatmap(corrmatrix, cm_labels, cm_labels, ax=ax,
                               cmap="YlGn", cbarlabel="Correlation coefficient")
    texts = heatmap.annotate_heatmap(im)

    ax.set_title("Heatmap of correlation matrix of features")

    fig.tight_layout()

    savefig(fig, "corrmatrix")

    return df
    

def investigate(df: pd.DataFrame) -> None:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df["date"], df['pct_close_futur'], 'g')
    plt.xticks(rotation=70)
    #plt.plot(df['pct_close_futur'], label='pct_close_futur')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("pct_close_futur")
    plt.ylabel("pct_close_futur")
    plt.legend()
    plt.grid()
    plt.show()


def strategy_sma(df: pd.DataFrame) -> pd.DataFrame:
    """A moving average crossover strategy. """

    df['SMA5'] = df["Close"].rolling(window = 5).mean()
    df['SMA30'] = df["Close"].rolling(window = 30).mean()

    df['signal'] = 0
    condition: pd.Series = df['SMA5'] > df['SMA30']

    df.loc[condition, "signal"] = 1

    df['returns'] = df['signal'] * df['pct_close_futur']

    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(df['date'], df['Close'], label='Closing Price', linestyle='dotted', color='black')
    plt.plot(df['date'], df['SMA5'], label='SMA 5', linestyle='dotted', color='red')
    plt.plot(df['date'], df['SMA30'], label='SMA 30', linestyle='dotted', color='green')
    plt.xticks(rotation=70)
    plt.legend()
    plt.grid()

    plt.show()

    return df


def main() -> int:
    if len(sys.argv) == 2:
        if sys.argv[1] == "i":
            df_tickers.append(download.initialDownload())
        elif sys.argv[1] == "c":
            df_tickers.append(download.fetchNewTicks())
        else:
            raise Exception("No or wrong commandline argument passed.")
    else:
        df = pd.read_csv("Tickers/IBM.csv")

        # Reverse, get increasing dates. Specific to IBM.csv.
        df = df[::-1]

        df_tickers.append(df)

    df: pd.DataFrame = df_tickers[0]

    # For some reason the name differs.
    df = df.rename(columns = {"1. open":    "Open",
                              "2. high":    "High",
                              "3. low":     "Low",
                              "4. close":   "Close",
                              "5. volume":  "Volume"})

    df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')

    df

    df = df.head(100)

    # ---------- Drop NaNs ---------
    len_before = len(df)
    df = df.dropna()
    print(f"Dropped from `df': {len_before - len(df)} rows, out of total {len_before}.")

    df["pct_close_futur"] = (df["Close"].shift(-2)-df["Close"])/df["Close"]

    # df = strategy_bollinger(df)
    #investigate(df)

    df = strategy_sma(df)

    df.to_csv("generated/df.csv")

    # Skip the other columns, backtest_static_portfolio() expects this.
    df = pd.DataFrame(df["returns"])

    weights = (1)
    backtest_static_portfolio(weights, df)

    # ---------- Split ---------
    split_point = int(0.80 * len(df))

    # TODO in/out of sample
    #in_sample = # before
    #out_sample = # after

    # "returns" is created in strategy()
    #y_train = df[["returns"]].iloc[:split_point]
    #X_train = df[["feature1", "feature2"]].iloc[:split_point]

    broker: BrokerABC.BrokerABC = Broker.init()

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