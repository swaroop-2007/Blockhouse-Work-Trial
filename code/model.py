import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import random
import json

class TradingEnvironment:
    def __init__(
        self,
        dataset: pd.DataFrame,
        total_shares: int = 1000,
        price_impact_factor: float = 0.1,
        spread_cost: float = 0.01
    ):
        self.dataset = dataset
        self.total_shares = total_shares
        self.price_impact_factor = price_impact_factor
        self.spread_cost = spread_cost
        self.timestamps = self.dataset['timestamp'].values
        self.base_price = self.dataset['ask_price_1'].iloc[0]  
        self.num_intervals = len(self.timestamps)
        self.remaining_shares = total_shares
        self.current_interval = 0
        self.current_price = self.base_price
        self.execution_history = []

    def calculate_trading_costs(self, shares_to_trade: int):
        temp_impact = (self.spread_cost + 
                      self.price_impact_factor * (shares_to_trade / self.total_shares)) * shares_to_trade
        perm_impact = 0.5 * self.price_impact_factor * (shares_to_trade / self.total_shares) * shares_to_trade
        return temp_impact, perm_impact

    def step(self, shares_to_trade: int):
        if shares_to_trade > self.remaining_shares:
            shares_to_trade = self.remaining_shares
            
        temp_impact, perm_impact = self.calculate_trading_costs(shares_to_trade)
        total_cost = temp_impact + perm_impact
        self.current_price = self.dataset['ask_price_1'].iloc[self.current_interval]

        reward = -total_cost
        self.remaining_shares -= shares_to_trade
        self.current_interval += 1

        self.execution_history.append({
            'timestamp': self.timestamps[self.current_interval - 1],
            'shares_traded': shares_to_trade
        })

        done = (self.current_interval >= self.num_intervals) or (self.remaining_shares <= 0)
        new_state = (self.remaining_shares, self.current_interval)

        return new_state, reward, done

    def reset(self):
        self.remaining_shares = self.total_shares
        self.current_interval = 0
        self.current_price = self.base_price
        self.execution_history = []
        return (self.remaining_shares, self.current_interval)


class QLearningTrader:
    def __init__(
        self,
        env: TradingEnvironment,
        learning_rate: float = 0.5,
        discount_factor: float = 0.95,
        epsilon: float = 0.005,
        min_trade: int = 0,
        max_trade: int = 1000
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_trade = min_trade
        self.max_trade = max_trade
        self.q_table = {}

    def get_possible_actions(self, remaining_shares: int):
        max_possible = min(remaining_shares, self.max_trade)
        step_size = max(1, max_possible // 10)
        return list(range(self.min_trade, max_possible + step_size, step_size))

    def get_action(self, state: tuple):
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state[0]))
        
        if state not in self.q_table:
            return random.choice(self.get_possible_actions(state[0]))
            
        return max(self.get_possible_actions(state[0]), 
                  key=lambda a: self.q_table.get((state, a), 0))

    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                
                old_value = self.q_table.get((state, action), 0)
                next_max = max([self.q_table.get((next_state, a), 0) 
                              for a in self.get_possible_actions(next_state[0])])
                
                new_value = (1 - self.learning_rate) * old_value + \
                           self.learning_rate * (reward + self.discount_factor * next_max)
                
                self.q_table[(state, action)] = new_value
                state = next_state

    def get_optimal_schedule(self):
        state = self.env.reset()
        done = False
        schedule = []
        
        while not done:
            action = self.get_action(state)
            state, _, done = self.env.step(action)
            schedule.append({
                'timestamp': self.env.execution_history[-1]['timestamp'],
                'shares': action
            })
            
        return schedule


def main():
    data = pd.read_csv("C:/Users/swaro/Desktop/blockhouse/Blockhouse-Work-Trial/data/AAPL_Quotes_Data.csv")
    
    env = TradingEnvironment(dataset=data)
    trader = QLearningTrader(env)
    trader.train(episodes=1000)
    schedule = trader.get_optimal_schedule()
    schedule_json = json.dumps(schedule, indent=4)
    
    return schedule_json


if __name__ == "__main__":
    optimal_schedule_json = main()
    print("\nOptimal Trading Schedule:")
    print(optimal_schedule_json)
