from io import StringIO
import numpy as np
import tempfile
import pytest

from world_models.experience_collector import (StateActionCollector,
                                               StateActionTransition, Rollout,
                                               RolloutCollector,
                                               save_numpy_arrays,
                                               load_numpy_arrays)
from world_models.actor import Actor, RandomActor
from world_models.environment import Environment, Pong


class TestActor(Actor):

    def get_action(self, state):
        return state + 1


class TestEnvironment(Environment):

    def __init__(self, max_action):
        self.max_action = max_action

    def reset(self):
        return 0

    def step(self, action):
        if action >= self.max_action:
            return None, None, True, None
        else:
            return action + 1, None, False, None


def setup_module(module):
    np.random.seed(0)


def test_state_action_collector_can_collect_one_episode():

    actor = TestActor()
    environment = TestEnvironment(4)

    state_action_collector = StateActionCollector()

    state_action_collector.collect_experience(actor, environment, 1)

    experience = state_action_collector.experience

    assert list(experience) == [
        StateActionTransition(0, 1, 2),
        StateActionTransition(2, 3, 4),
        StateActionTransition(4, 5, None)
    ]


def test_state_action_collector_can_collect_two_episodes():

    actor = TestActor()
    environment = TestEnvironment(4)

    state_action_collector = StateActionCollector()

    state_action_collector.collect_experience(actor, environment, 2)

    experience = state_action_collector.experience

    assert list(experience) == [
        StateActionTransition(0, 1, 2),
        StateActionTransition(2, 3, 4),
        StateActionTransition(4, 5, None),
        StateActionTransition(0, 1, 2),
        StateActionTransition(2, 3, 4),
        StateActionTransition(4, 5, None)
    ]


def test_rollout_collector_can_collect_rollout():

    actor = TestActor()
    environment = TestEnvironment(4)

    rollout_collector = RolloutCollector()

    rollout_collector.collect_experience(actor, environment, 1)

    rollouts = rollout_collector.rollouts

    expected_states = [np.array(0), np.array(2), np.array(4)]
    expected_actions = [1, 3, 5]

    assert list(rollouts) == [
        Rollout(expected_states, expected_actions)
    ]


def test_rollout_collector_can_collect_many_rollouts():

    actor = TestActor()
    environment = TestEnvironment(4)

    rollout_collector = RolloutCollector()
    rollout_collector.collect_experience(actor, environment, 1)

    environment = TestEnvironment(5)
    rollout_collector.collect_experience(actor, environment, 1)

    environment = TestEnvironment(6)
    rollout_collector.collect_experience(actor, environment, 2)

    rollouts = rollout_collector.rollouts

    assert list(rollouts) == [
        Rollout([np.array(0), np.array(2), np.array(4)],
                [1, 3, 5]),
        Rollout([np.array(0), np.array(2), np.array(4)],
                [1, 3, 5]),
        Rollout([np.array(0), np.array(2), np.array(4), np.array(6)],
                [1, 3, 5, 7]),
        Rollout([np.array(0), np.array(2), np.array(4), np.array(6)],
                [1, 3, 5, 7])
    ]


def test_rollout_collector_can_save_and_load():
    actor = TestActor()
    environment = TestEnvironment(4)

    rollout_collector = RolloutCollector()
    rollout_collector.collect_experience(actor, environment, 3)

    rollout_collector_1 = RolloutCollector()
    rollout_collector_2 = RolloutCollector()

    rollout_collector_1.collect_experience(actor, environment, 4)
    rollout_collector_2.collect_experience(actor, environment, 5)

    f = tempfile.NamedTemporaryFile()
    rollout_collector.save_experience(f.name)

    rollout_collector_1.load_experience(f.name)
    rollout_collector_2.load_experience(f.name)
    f.close()

    expected_rollout = Rollout([np.array(0), np.array(2), np.array(4)],
                               [1, 3, 5])

    assert list(rollout_collector.rollouts) == [expected_rollout] * 3
    assert list(rollout_collector_1.rollouts) == [expected_rollout] * 7
    assert list(rollout_collector_2.rollouts) == [expected_rollout] * 8


def test_save_and_load_numpy_arrays():
    x1 = np.random.randn(5, 6, 7)
    x2 = np.random.randn(3, 2, 2)

    f = tempfile.NamedTemporaryFile()
    save_numpy_arrays({'x1': x1, 'x2': x2}, f.name)

    loaded = load_numpy_arrays(f.name)

    f.close()

    assert set(loaded.keys()) == {'x1', 'x2'}
    np.testing.assert_allclose(x1, loaded['x1'])
    np.testing.assert_allclose(x2, loaded['x2'])


def test_random_agent_can_collect_experience_on_pong():
    actor = RandomActor([0, 1])
