

import pandas as pd

#df = pd.read_csv("df.csv")
df = pd.read_csv("Tickers/IBM.csv")

print(df)

print(df.loc[6348]["2. high"])