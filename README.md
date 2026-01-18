
Bot Tau is a financial quantitative trading bot written with Pandas & friends.

# Usage

The easiest way to use this code is to in bottau/ issue:

`pip install -e .`

This installs the code as an editable package, meaning symlinks are created to
the actual files.


# Development Notes

## Feature Names

Standardized feature names are as follows.

| Feature Name	| Pandas column name |
| ----------- | ----------- |
| Date/Timestamp, which is the DataFrame index	| time              |
|                                               | open              |
|                                               | high              |
|                                               | low               |
| Close/Adjusted close                          | close             |
| Volatility features                           | vol_*             |
| The strategy’s PnL                            | returns           |
| The closes (input data) as returns            | pct_close_futur   | 
| Volatility STDEV                              | vol_std           |
| Other volatility                              | features	vol_*   |

## File Naming Conventions for Strategies

In folder Strategies/ for strategy X:

| Filename | Description |
| ----------- | ----------- |
| backtest_X.py	    | Generates X_description.pdf and shows plots/prints to stdout. |
| live_X.py	        | Runs the strategy. |
| research_X.py	    | Essentially a playground. |
| generated_X/	    | Generated files. |
| input_X.csv       | The input data used as input, typically OHLCV. |
