import abc

import cv2
import gym


class Environment:

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass


class Pong(Environment):
    """Wrapper around gym atari Pong.

    Action space: 0 - move up
                  1 - move down
    """

    def __init__(self):
        self.env = gym.make('Pong-v0')

        # In the actual implementation 2 is UP and 3 is DOWN.
        self.action_map = {0: 2, 1: 3}

    @property
    def action_space(self):
        return [0, 1]

    def preprocess(self, state):
        state = cv2.resize(state, dsize=(64, 64),
                           interpolation=cv2.INTER_CUBIC)
        state = state / 255.0
        return state

    def reset(self):
        state = self.env.reset()
        return self.preprocess(state)

    def step(self, action):
        gym_action = self.action_map[action]
        state, reward, done, _ = self.env.step(gym_action)

        if reward != 0.0:
            done = True

        return self.preprocess(state), reward, done, None

    def render(self):
        self.env.render(mode='human')
