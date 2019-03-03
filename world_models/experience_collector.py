import abc
from collections import deque, namedtuple
import random
import numpy as np
import pickle


class ExperienceCollector:
    """An ExperienceCollector collects experience using a given actor in a given environment.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def collect_experience(self, actor, environment, n):
        pass

    @abc.abstractmethod
    def sample_experience(self, n):
        pass

    @abc.abstractmethod
    def save_experience(self, file_name):
        pass

    @abc.abstractmethod
    def load_experience(self, file_name):
        pass

    @abc.abstractmethod
    def reset_experience(self):
        pass


StateActionTransition = namedtuple('StateActionTransition', ('state', 'action', 'next_state'))


class StateActionCollector(ExperienceCollector):
    """The StateActionCollector takes actions in the given environment according to the given agent, and stores
    observed StateActionTransitions.

    The transition (state, action, next_state) means the environment was in state 'state',, the agent took action
    'action', and then the environment transitioned to state 'next_state'.
    """

    def __init__(self):
        super().__init__()

        self.experience = []

    def collect_experience(self, actor, environment, num_episodes):
        for i in range(num_episodes):
            state = environment.reset()
            done = False
            while not done:
                action = actor.get_action(state)
                next_state, _, done, _ = environment.step(action)

                self.experience.append(StateActionTransition(state, action, next_state))

                state = next_state

    def sample_experience(self, num_transitions):
        return random.sample(self.experience, num_transitions)

    def save_experience(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.experience, f)

    def load_experience(self, file_name):
        with open(file_name, 'rb') as f:
            self.experience = pickle.load(f)

    def reset_experience(self):
        self.experience = []


Rollout = namedtuple('Rollout', ('states', 'actions'))

def rollout_to_array(rollout):
    pass

def save_rollouts(rollouts, f):
    states = np.stack(rollout.states)


def save_numpy_arrays(arr, f):
    """Saves a given dictionary of numpy arrays to the file object f."""
    pass

def load_numpy_arrays(f):
    """Loads a dictionary of numpy arrays from the file object f."""
    pass


class RolloutCollector(ExperienceCollector):
    """The RolloutCollector collects sequences of states and actions and saves and loads them efficiently using numpy.
    """

    def __init__(self):
        self.rollouts = []

    def collect_experience(self, actor, environment, num_episodes):
        for i in range(num_episodes):
            state = environment.reset()
            states = [state]
            actions = []
            done = False
            while not done:
                action = actor.get_action(state)
                next_state, _, done, _ = environment.step(action)

                states.append(state)
                actions.append(action)

                state = next_state
            self.rollouts.append(Rollout(states, actions))

    def sample_experience(self, num_transitions):
        return random.sample(self.rollouts, num_transitions)

    def save_experience(self, file_name):
        rollouts_dict = {'states_{}'.format(i): v[0] for i, v in enumerate(self.rollouts)}
        rollouts_dict.update({'actions_{}'.format(i): v[1] for i, v in enumerate(self.rollouts)})
            with open(file_name, 'wb') as f:
                save_numpy_arrays(rollouts_dict, f)

    def load_experience(self, file_name):
        with open(file_name, 'rb') as f:
            rollouts_dict = load_numpy_arrays(f)

        self.rollouts = 

    def reset_experience(self):
        self.rollouts = []
