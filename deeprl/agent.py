"""This file contains various classes that can play environments.
"""
import numpy as np


# Agents
class BaseAgent:
    def __init__(self):
        pass

    def act(self, state):
        """Chooses an action given a state.
        """
        pass


class DQNAgent(BaseAgent):
    def __init__(self, net, action_picker, sess):
        self.net = net
        self.action_picker = action_picker

    def act(self, state):
        return self.action_picker(self.net(state))


class RandomAgent(BaseAgent):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, state):
        return np.random.choice(self.num_actions)


# Action pickers
class BaseActionPicker:
    def __init__(self):
        pass

    def __call__(self, x):
        pass


class EpsilonGreedyActionPicker(BaseActionPicker):

    def __init__(self, start_epsilon, final_epsilon, epsilon_steps, num_actions):
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_steps = epsilon_steps
        self.num_actions = num_actions
        self.i = 0
        self.epsilon = start_epsilon

        super().__init__()

    def update_epsilon(self):
        if self.i > self.epsilon_steps:
            self.epsilon = self.final_epsilon
        a = float(self.i) / float(self.epsilon_steps)
        self.epsilon = (1.0 - a) * self.start_epsilon + a * self.final_epsilon

    def __call__(self, x):
        assert len(x.shape) == 1

        self.i += 1
        self.update_epsilon()

        # Play randomly epsilon of the time
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(x)
