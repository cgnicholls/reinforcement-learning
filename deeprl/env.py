"""This file defines an environment that an agent can try to learn.
"""
import gym

class BaseEnv:

    def __init__(self):
        pass

    def reset(self):
        """Resets the environment.
        """
        pass

    def step(self, action):
        """Takes the given action.
        """
        pass


class GymEnv(BaseEnv):

    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def preprocess(self, state):
        return state

    def reset(self):
        state = self.env.reset()
        return self.preprocess(state)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.preprocess(observation), reward, done

    def available_actions(self):
        return list(range(self.env.action_space.n))
