import numpy as np
from unittest import mock

from world_models.experience_collector import StateActionCollector, StateActionTransition, Rollout


def test_state_action_collector_can_collect_one_episode():

    agent = mock.MagicMock()
    environment = mock.MagicMock()

    def get_action(state):
        return state + 1

    def reset():
        return 0

    def step(action):
        if action > 3:
            return None, None, True, None
        else:
            return action + 1, None, False, None

    agent.get_action = get_action
    environment.reset = reset
    environment.step = step

    state_action_collector = StateActionCollector(agent, environment)

    state_action_collector.collect_experience(1)

    experience = state_action_collector.experience

    assert list(experience) == [
        StateActionTransition(0, 1, 2),
        StateActionTransition(2, 3, 4),
        StateActionTransition(4, 5, None)
    ]


def test_state_action_collector_can_collect_two_episodes():

    agent = mock.MagicMock()
    environment = mock.MagicMock()

    def get_action(state):
        return state + 1

    def reset():
        return 0

    def step(action):
        if action > 3:
            return None, None, True, None
        else:
            return action + 1, None, False, None

    agent.get_action = get_action
    environment.reset = reset
    environment.step = step

    state_action_collector = StateActionCollector(agent, environment)

    state_action_collector.collect_experience(2)

    experience = state_action_collector.experience

    assert list(experience) == [
        StateActionTransition(0, 1, 2),
        StateActionTransition(2, 3, 4),
        StateActionTransition(4, 5, None),
        StateActionTransition(0, 1, 2),
        StateActionTransition(2, 3, 4),
        StateActionTransition(4, 5, None)
    ]

def test_rollout_to_arrays():
    states = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [-1, -2, -3]])]
    actions = [2, 3]
    rollout = Rollout(np.array(states, actions))

    states_arr, actions_arr = rollout_to_arrays(rollout)

    states = rollout.states
    actions = rollout.actions
