# market_making_strategies.py

"""
Market Making Strategies Module for Automated Market-Making Systems

This module contains functions for implementing and testing various market-making strategies using trained reinforcement learning models.

Techniques Used:
- Spread setting
- Inventory management
- Order execution algorithms

Libraries/Tools:
- numpy
- pandas
- tensorflow
- gym

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from rl_models import RLModels, MarketMakingEnv
import os

class MarketMakingStrategies:
    def __init__(self, model, env):
        """
        Initialize the MarketMakingStrategies class.
        
        :param model: trained reinforcement learning model
        :param env: gym.Env, custom trading environment
        """
        self.model = model
        self.env = env

    def execute_trade(self, state):
        """
        Execute a trade based on the current state using the trained model.
        
        :param state: array, current state of the environment
        :return: int, action to be taken (0: Hold, 1: Buy, 2: Sell)
        """
        q_values = self.model.predict(state[np.newaxis])
        action = np.argmax(q_values[0])
        return action

    def simulate_trading(self, num_steps=1000):
        """
        Simulate trading using the market-making strategy.
        
        :param num_steps: int, number of steps to simulate
        :return: dict, simulation results including total reward and final balance
        """
        state = self.env.reset()
        total_reward = 0
        balance_history = []
        positions_history = []
        reward_history = []

        for step in range(num_steps):
            action = self.execute_trade(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            balance_history.append(self.env.balance)
            positions_history.append(self.env.positions)
            reward_history.append(reward)

            if done:
                break

            state = next_state

        return {
            'total_reward': total_reward,
            'final_balance': self.env.balance,
            'balance_history': balance_history,
            'positions_history': positions_history,
            'reward_history': reward_history
        }

    def evaluate_strategy(self, simulation_results, output_dir):
        """
        Evaluate and visualize the performance of the market-making strategy.
        
        :param simulation_results: dict, results from the trading simulation
        :param output_dir: str, directory to save the evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        balance_history = simulation_results['balance_history']
        positions_history = simulation_results['positions_history']
        reward_history = simulation_results['reward_history']

        # Plot balance history
        plt.figure(figsize=(10, 6))
        plt.plot(balance_history, label='Balance')
        plt.xlabel('Step')
        plt.ylabel('Balance')
        plt.title('Balance History')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'balance_history.png'))
        plt.show()

        # Plot positions history
        plt.figure(figsize=(10, 6))
        plt.plot(positions_history, label='Positions')
        plt.xlabel('Step')
        plt.ylabel('Positions')
        plt.title('Positions History')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'positions_history.png'))
        plt.show()

        # Plot reward history
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label='Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward History')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'reward_history.png'))
        plt.show()

        # Save simulation results
        with open(os.path.join(output_dir, 'simulation_results.txt'), 'w') as f:
            f.write(f"Total Reward: {simulation_results['total_reward']}\n")
            f.write(f"Final Balance: {simulation_results['final_balance']}\n")
        print(f"Simulation results saved to {os.path.join(output_dir, 'simulation_results.txt')}")

if __name__ == "__main__":
    # Example data loading and environment setup
    data_filepath = 'data/processed/processed_data.csv'
    data = pd.read_csv(data_filepath)
    env = MarketMakingEnv(data)

    # Load the trained RL model
    model_dir = 'models/'
    rl_model = RLModels(env)
    model = rl_model.load_model(model_dir)

    # Initialize and test market-making strategy
    strategy = MarketMakingStrategies(model, env)
    simulation_results = strategy.simulate_trading(num_steps=1000)
    strategy.evaluate_strategy(simulation_results, 'results/market_making')
