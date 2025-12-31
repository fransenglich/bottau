
Bot Tau is a financial quantitative trading bot written with Pandas & friends.

# Development Notes

## Feature Names

Standardized feature names are as follows.

| Feature Name	| Pandas column name |
| ----------- | ----------- |
| Date/Timestamp, which is the DataFrame index	| time      |
|                                               | open      |
|                                               | high      |
|                                               | low       |
| Close/Adjusted close                          | close     |
| The strategy’s PnL                            | returns   |

## File Naming Conventions for Strategies

In folder Strategies/ for strategy X:

| Filename | Description |
| ----------- | ----------- |
| backtest_X.py	    | Generates X_description.pdf and shows plots/prints to stdout. |
| live_X.py	        | Runs the strategy. |
| research_X.py	    | Essentially a playground. |
| generated_X/	    | Generated files. |