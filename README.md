
Bot Tau is a financial quantitative trading bot written with Pandas & friends.

# Usage

The easiest way to use this code is to in bottau/ (this folder) issue:

`pip install -e .`

This installs the code as an editable package, meaning symlinks are created to
the actual files.


# Development Notes

## Feature Names

Standardized feature names are as follows.

| Feature Name	| Pandas column name |
| ----------- | ----------- |
| time              | Date/Timestamp, which is the DataFrame index      |
| open              |                                                   |
| high              |                                                   |
| low               |                                                   |
| close             | Close/Adjusted close                              |
| returns           | The strategy’s PnL                                |
| pct_close_futur   | The closes (input data) as returns                |
| vol_std           | Volatility STDEV                                  |
| features vol_*    | Other volatility features                         |
| *_futur           | Target variables/future                           |
| signal            | The strategy's advice/signal. 1 = buy, -1 = sell  |

## File Naming Conventions for Strategies

In folder Strategies/ for strategy X:

| Filename | Description |
| ----------- | ----------- |
| backtest_X.py	    | Generates X_description.pdf and shows plots/prints to stdout. |
| live_X.py	        | Runs the strategy. |
| research_X.py	    | Essentially a playground. |
| generated_X/	    | Generated files. |
| input_X.csv       | The input data used as input, typically OHLCV. |
