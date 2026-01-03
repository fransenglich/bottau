import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib import common


# Copied from Lucas.
def backtest_static_portfolio(weights,
                              database,
                              ben="^GSPC",
                              timeframe: int = 252,
                              CR: bool = False):
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
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn')

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
    print(drawdown)


    ######################### COMPUTE THE VaR ##################################
    theta = 0.01
    # Number of simulations
    n = 100000

    # Find the values for theta% error threshold
    t = int(n*theta)

    # Create a vector with n simulations of the normal law
    vec = pd.DataFrame(np.random.normal(mean, std, size=(n,)),
                       columns=["Simulations"])

    # Orderer the values and find the theta% value
    VaR = -vec.sort_values(by="Simulations").iloc[t].values[0]


    ######################### COMPUTE THE cVaR #################################
    cVaR = -vec.sort_values(by="Simulations").iloc[0:t,:].mean().values[0]

    ######################### COMPUTE THE RC ###################################
    if CR:
        # Find the number of the asset in the portfolio
        length = len(weights)

        # Compute the risk contribution of each asset
        crs = []
        for i in range(length):
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

    plt.figure(figsize=(15, 8))
    plt.plot(join.iloc[:, 0].cumsum()*100, color="#035593", linewidth=3)
    plt.plot(join.iloc[:, 1].cumsum()*100, color="#068C72", linewidth=3)
    plt.title("CUMULTATIVE RETURN", size=15)
    plt.ylabel("Cumulative return %", size=15)
    plt.xticks(size=15, fontweight="bold")
    plt.yticks(size=15, fontweight="bold")
    plt.legend(["Strategy", "Benchmark"])
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.fill_between(drawdown.index, drawdown*100, 0, color="#CE5151")
    plt.plot(drawdown.index, drawdown*100, color="#930303", linewidth=1.5)
    plt.title("DRAWDOWN", size=15)
    plt.ylabel("Drawdown %", size=15)
    plt.xticks(size=15, fontweight="bold")
    plt.yticks(size=15, fontweight="bold")
    plt.show()

    if CR:
        plt.figure(figsize=(15, 8))
        plt.scatter(columns, crs, linewidth=3, color="#B96553")
        plt.axhline(0, color="#53A7B9")
        plt.grid(axis="x")
        plt.title("RISK CONTRIBUTION PORTFOLIO", size=15)
        plt.xlabel("Assets")
        plt.ylabel("Risk contribution")
        plt.xticks(size=15, fontweight="bold")
        plt.yticks(size=15, fontweight="bold")
        plt.show()


def backtest(df: pd.DataFrame, sn: str) -> None:
    """A function by Lucas Inglese modified by me that plots and prints a
    backtest.

    Argument sn is strategy name.

    The passed DataFrame must have a column named returns, which is the returns
    of the strategy to be backtested."""

    if "returns" not in df.columns:
        raise ValueError("The passed DataFrame must have a 'returns' column.")

    # ---- Drawdown ----
    # 1 + & cumprod() because 'returns' are not log returns.
    df['comp_cumulative_returns'] = (1 + df['returns']).cumprod()
    df['cumulative_max'] = df['comp_cumulative_returns'].cummax()
    df['drawdown'] = ((df['comp_cumulative_returns'] - df['cumulative_max']) / df['cumulative_max']) * 100

    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['drawdown'], label="Drawdown")
    plt.title("Drawdown")
    plt.ylabel("Drawdown %")
    plt.legend()
    common.savefig(plt, "drawdown", sn)

    # ---- Drawdown Histogram ----
    plt.figure(figsize=common.FIG_SIZE)
    plt.hist(df['drawdown'], bins='auto')
    plt.title("Drawdown Distribution")
    common.savefig(plt, "drawdown_dist", sn)

    # ---- Returns ----
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['returns'], label='Returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Returns")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    common.savefig(plt, "returns", sn)

    # ---- Returns Histogram ----
    plt.figure(figsize=common.FIG_SIZE)
    plt.hist(df['returns'], bins='auto')
    plt.title("Returns Distribution")
    common.savefig(plt, "returns_dist", sn)

    # ---- Cumulative Returns ----
    df['cumulative_returns'] = df['returns'].cumsum()
    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['cumulative_returns'], label='Returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Cumulative Returns")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    common.savefig(plt, "cumulative_returns", sn)

    # ---- Cumulative Returns Minus Transaction Costs ----
    def transaction_cost(trade: float) -> float:
        return trade - (common.TRANSACTION_COMMISSION + trade/2)

    df['cum_with_trans'] = df['cumulative_returns'].map(transaction_cost)

    plt.figure(figsize=common.FIG_SIZE)
    plt.plot(df['cum_with_trans'], label='Netted returns')
    plt.axhline(0, linestyle='dashed', color='black', alpha=0.5)
    plt.title("Cumulative Returns Minus Transaction Costs")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid()
    common.savefig(plt, "cumulative_returns_except_trans_costs", sn)
    # TODO simulate slippage
    # TODO fx risk

    # ---- Drawdown ----
    max_drawdown = common.max_drawdowns(df["returns"])
    max_drawdown = np.round(max_drawdown, 2)
    print(f"Max drawdown: {max_drawdown}")

    cr = common.calmar_ratio(df["returns"])
    cr = np.round(cr, 4)

    # ---- Sortino Ratio ----
    sr = common.sortino_ratio(df["returns"])
    sr = np.round(sr, 4)

    # ---- Write constants ----
    path = os.path.join(os.path.dirname(__file__),
                        f"../strategies/generated_{sn}")
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/constants.tex", "w") as f:
        f.write(f"\\def\\constantMaxdrawdown{{{max_drawdown}}}")
        f.write(f"\n\\def\\constantStartdate{{{df.index.min()}}}")
        f.write(f"\n\\def\\constantEnddate{{{df.index.max()}}}")
        f.write(f"\n\\def\\constantTransactionCommission{{{common.TRANSACTION_COMMISSION}}}")

        rmean = np.round(df['returns'].mean() * 100, 4)
        std = np.round(df['returns'].std(), 4)
        sr = np.round(common.sharpe_ratio(df['returns']), 4)
        f.write(f"\n\\def\\constantRMean{{{rmean}}}")
        f.write(f"\n\\def\\constantSharpeRatio{{{sr}}}")
        f.write(f"\n\\def\\constantStd{{{std}}}")
        f.write(f"\n\\def\\constantCalmarRatio{{{cr}}}")
        f.write(f"\n\\def\\constantSortinoRatio{{{sr}}}")


def backtest2(sn: str,
              df: pd.DataFrame) -> None:
    pass
