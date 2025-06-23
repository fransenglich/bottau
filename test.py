

import pandas as pd
import yfinance as yf

#df = pd.read_csv("df.csv")
#df = pd.read_csv("Tickers/IBM.csv")

#print(df)

#print(df.loc[6348]["2. high"])

#list_tickers = ["META", "NFLX", "TSLA"]
#database = yf.download(list_tickers)
#database.to_csv("Tickers/db3assets.csv")

df = pd.read_csv("Tickers/db3assets.csv")

print(df['Close'])
print(df['Close'].describe())
df['Close'].to_csv("Tickers/only_close.csv")


# Importation of data
list_tickers = ["META", "NFLX", "TSLA"]
database = yf.download(list_tickers)

# Take only the adjusted stock price
database = database["Close"]

# Drop missing values
data = database.dropna().pct_change(1).dropna()