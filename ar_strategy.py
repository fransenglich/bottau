from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import pandas as pd


def AR_forecast(train_set):
    p = 1
    model = ARIMA(train_set, order=(p, 0, 0))

    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast()

    return forecast[0][0]


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
    pass


if __name__ == "__main__":
    main()
