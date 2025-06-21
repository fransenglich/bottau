import matplotlib.pyplot as plt
import matplotlib
import heatmap
import sys
import download
import ta
import quantreo.features_engineering as fe
import pandas as pd
import numpy as np

# Constants
# ----------------------------------------------
TICKERS = ['AAPL', 'TSLA']
DEFAULT_FIGSIZE = (6, 4)
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

def main() -> int:

    def savefig(plt: matplotlib.figure.Figure, basename: str) -> None:
        plt.savefig(f"generated/{basename}.png")

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

    # ---------- Drop NaNs ---------
    len_before = len(df)
    df = df.dropna()
    print(f"Dropped from `df': {len_before - len(df)} rows, out of total {len_before}.")

    # ---------- Split ---------
    split_point = int(0.80 * len(df))

    y_train = df[["returns"]].iloc[:split_point]
    X_train = df[["feature1", "feature2"]].iloc[:split_point]

    # ---- Write constants ----
    drawdown_max = round(abs(df['drawdown'].min()), 2) # Percent
    print(f"Max drawdown: {drawdown_max}")

    with open("generated/constants.tex", "w") as f:
        f.write(f"\def\constantMaxdrawdown{{{drawdown_max}}}")
        f.write(f"\n\def\constantStartdate{{{df['date'].min()}}}")
        f.write(f"\n\def\constantEnddate{{{df['date'].max()}}}")
        f.write(f"\n\def\constantTransactionCommission{{{TRANSACTION_COMMISSION}}}")

        rmean = round(df['returns'].mean() * 100, 4)
        std = round(df['returns'].std(), 4)
        sr = round(rmean/std, 4)
        f.write(f"\n\def\constantRMean{{{rmean}}}")
        f.write(f"\n\def\constantSharpeRatio{{{sr}}}")
        f.write(f"\n\def\constantStd{{{std}}}")

    df.to_csv("generated/df.csv")

    return 0


if __name__ == "__main__":
    main()