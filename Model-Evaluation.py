# model_evaluation.py

"""
Model Evaluation Module for Automated Market-Making Systems

This module contains functions for evaluating the performance of the market-making strategies and reinforcement learning models.

Techniques Used:
- Profit and Loss (P&L) Analysis
- Sharpe Ratio Calculation
- Maximum Drawdown Calculation
- Win Rate Calculation

Libraries/Tools:
- numpy
- pandas
- matplotlib

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def profit_and_loss(self, balance_history):
        """
        Calculate the Profit and Loss (P&L) from the balance history.
        
        :param balance_history: list, history of balances during trading
        :return: float, total profit and loss
        """
        return balance_history[-1] - balance_history[0]

    def sharpe_ratio(self, balance_history, risk_free_rate=0.0):
        """
        Calculate the Sharpe Ratio from the balance history.
        
        :param balance_history: list, history of balances during trading
        :param risk_free_rate: float, risk-free rate for Sharpe ratio calculation
        :return: float, Sharpe ratio
        """
        returns = np.diff(balance_history) / balance_history[:-1]
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    def maximum_drawdown(self, balance_history):
        """
        Calculate the Maximum Drawdown from the balance history.
        
        :param balance_history: list, history of balances during trading
        :return: float, maximum drawdown
        """
        peak = balance_history[0]
        max_drawdown = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def win_rate(self, reward_history):
        """
        Calculate the Win Rate from the reward history.
        
        :param reward_history: list, history of rewards during trading
        :return: float, win rate
        """
        wins = np.sum(np.array(reward_history) > 0)
        total_trades = len(reward_history)
        return wins / total_trades

    def evaluate(self, simulation_results, output_dir):
        """
        Evaluate and visualize the performance of the trading strategy.
        
        :param simulation_results: dict, results from the trading simulation
        :param output_dir: str, directory to save the evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        balance_history = simulation_results['balance_history']
        reward_history = simulation_results['reward_history']

        # Calculate evaluation metrics
        total_pnl = self.profit_and_loss(balance_history)
        sharpe_ratio = self.sharpe_ratio(balance_history)
        max_drawdown = self.maximum_drawdown(balance_history)
        win_rate = self.win_rate(reward_history)

        # Print evaluation metrics
        print(f"Total P&L: {total_pnl}")
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Maximum Drawdown: {max_drawdown}")
        print(f"Win Rate: {win_rate}")

        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Total P&L: {total_pnl}\n")
            f.write(f"Sharpe Ratio: {sharpe_ratio}\n")
            f.write(f"Maximum Drawdown: {max_drawdown}\n")
            f.write(f"Win Rate: {win_rate}\n")
        print(f"Evaluation results saved to {os.path.join(output_dir, 'evaluation_results.txt')}")

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

if __name__ == "__main__":
    # Example usage
    simulation_results = {
        'balance_history': [10000 + np.random.normal(0, 1) for _ in range(1000)],
        'reward_history': [np.random.normal(0, 1) for _ in range(1000)]
    }
    output_dir = 'results/evaluation/'

    evaluator = ModelEvaluation()
    evaluator.evaluate(simulation_results, output_dir)
    print("Model evaluation completed and results saved.")
