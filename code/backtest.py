import pandas as pd
import numpy as np
import json
from benchmark_costs import Benchmark  
from model import TradingEnvironment, QLearningTrader  

data_path = 'C:/Users/swaro/Desktop/blockhouse/Blockhouse-Work-Trial/data/merged_bid_ask_ohlcv_data (1).csv'
market_data = pd.read_csv(data_path)

benchmark = Benchmark(data=market_data)
initial_inventory = 1000
preferred_timeframe = 390  

#Run TWAP strategy 
twap_trades = benchmark.get_twap_trades(data=market_data, initial_inventory=initial_inventory, preferred_timeframe=preferred_timeframe)
twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, market_data, preferred_timeframe=preferred_timeframe)

# Run VWAP strategy
vwap_trades = benchmark.get_vwap_trades(data=market_data, initial_inventory=initial_inventory, preferred_timeframe=preferred_timeframe)
vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, market_data, preferred_timeframe=preferred_timeframe)

# Run the Q-learning environment and agent
env = TradingEnvironment(dataset=market_data)
trader = QLearningTrader(env)
trader.train(episodes=1000)
qlearning_schedule = trader.get_optimal_schedule()
qlearning_schedule_df = pd.DataFrame(qlearning_schedule)
model_slippage, model_market_impact = benchmark.simulate_strategy(qlearning_schedule_df, market_data, preferred_timeframe=preferred_timeframe)


comparison_df = pd.DataFrame({
    "Strategy": ["TWAP", "VWAP", "Q-Learning Model"],
    "Total Slippage": [np.sum(twap_slippage), np.sum(vwap_slippage), np.sum(model_slippage)],
    "Total Market Impact": [np.sum(twap_market_impact), np.sum(vwap_market_impact), np.sum(model_market_impact)]
})

print("TWAP Total Slippage:", np.sum(twap_slippage))
print("TWAP Total Market Impact:", np.sum(twap_market_impact))
print("VWAP Total Slippage:", np.sum(vwap_slippage))
print("VWAP Total Market Impact:", np.sum(vwap_market_impact))
print("Q-Learning Model Total Slippage:", np.sum(model_slippage))
print("Q-Learning Model Total Market Impact:", np.sum(model_market_impact))

print("\nDetailed Comparison DataFrame:")
print(comparison_df)

qlearning_schedule_json = json.dumps(qlearning_schedule, indent=4)
with open("qlearning_optimal_schedule.json", "w") as file:
    file.write(qlearning_schedule_json)

print("\nOptimal Q-Learning Schedule saved to 'qlearning_optimal_schedule.json'")
