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

    def __init__(self):
        self.env = gym.make('Pong-v0')

    def preprocess(self, state):
        state = cv2.resize(state, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        state = state / 255.0
        return state

    def reset(self):
        state = self.env.reset()
        return self.preprocess(state)

    def step(self, action):
        state, reward, done, _ = self.env.step(action)

        return self.preprocess(state), reward, done, None
