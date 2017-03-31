import gym
import numpy as np
import random

class CustomGymClassicControl:
    def __init__(self, game_name, skip_actions=4):
        self.env = gym.make(game_name)
        self.skip_actions = skip_actions
        self.action_size = env.action_space.n
        self.action_space = range(self.action_size)
        self.observation_shape = env.observation_space.shape

        self.state = None

    def render(self):
        self.env.render()

    def reset(self):
        return np.reshape(self.env.reset(), [1,-1])

    def step(self, action_idx):
        action = self.action_space[action_idx]
        accum_reward = 0
        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break

        return np.reshape(s, [1, -1]), accum_reward, term, info
