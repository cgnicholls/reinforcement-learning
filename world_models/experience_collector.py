import abc
from collections import namedtuple
import random
import pickle
import deepdish as dd


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
    """The StateActionCollector takes actions in the given environment
    according to the given agent, and stores observed StateActionTransitions.

    The transition (state, action, next_state) means the environment was in
    state 'state', the agent took action 'action', and then the environment
    transitioned to state 'next_state'.
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


def save_numpy_arrays(arr, file_name):
    """Saves a given dictionary of numpy arrays to the given file."""
    dd.io.save(file_name, arr)


def load_numpy_arrays(file_name):
    """Loads a dictionary of numpy arrays from the given file."""
    return dd.io.load(file_name)


class RolloutCollector(ExperienceCollector):
    """The RolloutCollector collects sequences of states and actions and saves and loads them efficiently using numpy.
    """

    def __init__(self):
        self.rollouts = []

    def collect_experience(self, actor, environment, num_episodes):
        for i in range(num_episodes):
            state = environment.reset()
            states = []
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
        rollouts_dict = {'rollout_{}'.format(str(i)): (states, actions) for
                         i, (states, actions) in enumerate(self.rollouts)}

        save_numpy_arrays(rollouts_dict, file_name)

    def load_experience(self, file_name):
        try:
            loaded_rollouts = load_numpy_arrays(file_name)
            loaded_rollouts = [Rollout(states, actions) for (states, actions) in
                               loaded_rollouts.values()]
            self.rollouts += loaded_rollouts
        except ValueError as e:
            print("Could not load experience for {}".format(file_name))
            print(e)


    def reset_experience(self):
        self.rollouts = []


def get_rollout_states(rollouts):
    states = []
    for rollout in rollouts:
        states.extend(rollout.states)

    return states


class StatesServer:
    """This class reads in rollouts stored in the given files and serves all states once."""

    def __init__(self, file_names):
        self.file_names = file_names
        random.shuffle(self.file_names)

    def serve(self, batch_size):
        for file_name in self.file_names:
            print("Serving {}".format(file_name))
            experience_collector = RolloutCollector()
            experience_collector.load_experience(file_name)

            states = get_rollout_states(experience_collector.rollouts)
            random.shuffle(states)

            i = 0
            while i < len(states):
                yield states[i: i + batch_size]
                i += batch_size
