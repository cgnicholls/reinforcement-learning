import abc

import random

class Actor:

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, state):
        pass


class RandomActor(Actor):

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        return random.choice(self.action_space)
