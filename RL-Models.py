# rl_models.py

"""
Reinforcement Learning Models Module for Automated Market-Making Systems

This module contains functions for developing and training reinforcement learning models to optimize market-making strategies.

Techniques Used:
- Q-learning
- Deep Q-Networks (DQN)
- Policy Gradient Methods
- Actor-Critic Methods

Libraries/Tools:
- numpy
- pandas
- tensorflow
- keras
- gym

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import gym
import os
import joblib

class MarketMakingEnv(gym.Env):
    def __init__(self, data):
        super(MarketMakingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
        self.balance = 10000
        self.positions = 0

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.positions = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0

        if action == 1:  # Buy
            self.positions += 1
            self.balance -= current_price
        elif action == 2:  # Sell
            self.positions -= 1
            self.balance += current_price

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if done:
            reward = self.balance + self.positions * current_price

        next_state = self.data.iloc[self.current_step].values
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

class RLModels:
    def __init__(self, env):
        """
        Initialize the RLModels class.
        
        :param env: gym.Env, custom trading environment
        """
        self.env = env
        self.model = None

    def build_dqn_model(self, state_shape, num_actions):
        """
        Build a Deep Q-Network (DQN) model.
        
        :param state_shape: tuple, shape of the input state
        :param num_actions: int, number of possible actions
        :return: compiled DQN model
        """
        inputs = layers.Input(shape=state_shape)
        layer1 = layers.Dense(64, activation='relu')(inputs)
        layer2 = layers.Dense(64, activation='relu')(layer1)
        outputs = layers.Dense(num_actions, activation='linear')(layer2)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def train_dqn(self, num_episodes=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        """
        Train the DQN model.
        
        :param num_episodes: int, number of training episodes
        :param gamma: float, discount factor
        :param epsilon: float, initial exploration rate
        :param epsilon_min: float, minimum exploration rate
        :param epsilon_decay: float, exploration rate decay factor
        """
        state_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n
        self.model = self.build_dqn_model(state_shape, num_actions)

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                if np.random.rand() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.predict(state[np.newaxis])
                    action = np.argmax(q_values[0])

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                target = reward
                if not done:
                    target = reward + gamma * np.amax(self.model.predict(next_state[np.newaxis])[0])

                target_f = self.model.predict(state[np.newaxis])
                target_f[0][action] = target

                self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
                state = next_state

                if done:
                    print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
                    break

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

    def save_model(self, model_dir, model_name='dqn_model.h5'):
        """
        Save the trained model to a file.
        
        :param model_dir: str, directory to save the model
        :param model_name: str, name of the model file
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_dir, model_name='dqn_model.h5'):
        """
        Load a trained model from a file.
        
        :param model_dir: str, directory containing the model
        :param model_name: str, name of the model file
        :return: loaded model
        """
        model_path = os.path.join(model_dir, model_name)
        self.model = tf.keras.models.load_model(model_path)
        return self.model

if __name__ == "__main__":
    # Example data loading and environment setup
    data_filepath = 'data/processed/processed_data.csv'
    data = pd.read_csv(data_filepath)
    env = MarketMakingEnv(data)

    # Initialize and train the RL model
    rl_model = RLModels(env)
    rl_model.train_dqn(num_episodes=100)

    # Save the trained model
    model_dir = 'models/'
    rl_model.save_model(model_dir)
