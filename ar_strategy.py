from statsmodels.tsa.arima.model import ARIMA
import backtest
import numpy as np
import pandas as pd


def AR_forecast(train_set: pd.DataFrame):
    """
    """

    p = 1
    model = ARIMA(train_set, order=(p, 0, 0))

    model_fit = model.fit()
    #print(model_fit.summary())
    forecast = model_fit.forecast()

    return forecast.iloc[0]


def AR_strategy(df: pd.DataFrame, returns: bool):
    """
    """

    splitpoint = int(len(df) * 0.70)

    df['predicted'] = df['Adj Close'].rolling(splitpoint) \
                                     .apply(AR_forecast)

    if returns:
        df['returns'] = df['Adj Close']
        df['signal'] = np.where(df['predicted'] > 0, 1, -1)
    else:
        df['returns'] = df['Adj Close'].pct_change(1)
        df['signal'] = np.where(df['predicted'] > df['Adj Close'], 1, -1)

    df['strategy'] = df['signal'].shift(1) * df['returns']

    return df


def main():
    df = pd.read_csv("Tickers/IBM.csv", index_col="date", parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).to_period('D')
    df = df[::-1]

    # Standardize feature names
    df.rename(columns={"4. close": "Adj Close"}, inplace=True)

    df = df[0:100]

    df = AR_strategy(df, False)

    backtest.backtest_static_portfolio((1), df)

if __name__ == "__main__":
    main()
