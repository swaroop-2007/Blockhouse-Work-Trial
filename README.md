# Blockhouse Work Trial Repo

This repository consists of the following components:

1. **Benchmarks** - Code for TWAP and VWAP strategies.
2. **Cost Calculations** - Code for calculating various costs within the rewards component.
3. **Reinforcement Learning Model based on a research paper** - Model to minimize transaction costs.
4. **Report and Results** - Comparison of Results of the model and TWAP, VWAP strategies.
5. **Datasets** - Example datasets that can be used for this process.

## Run model.py file to train the model. 
## Run backtest.py file to compare the results of TWAP, VWAP and model, specifically on Total Slippage and Total Market Impact. 

### Backtest.py file uses merged_bid_ask_ohlcv_data (1).csv file, present in data folder. 

### Make sure to change the path in data_path in backtest.py, and model.py -> main function. 








