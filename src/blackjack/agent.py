from collections import defaultdict
import gymnasium as gym
import numpy as np


import pickle

class BlackjackAgent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor=0.95, q_values=None):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

        def default_q_values():
            return np.zeros(env.action_space.n)

        if q_values is None:
            self.q_values = defaultdict(default_q_values)
        else:
            self.q_values = defaultdict(default_q_values, q_values)

    def save(self, path="blackjack/models/blackjack_agent.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_values), f)

    def load(self, path="blackjack/models/blackjack_agent.pkl"):
        with open(path, "rb") as f:
            q_values_dict = pickle.load(f)
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n), q_values_dict)

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)